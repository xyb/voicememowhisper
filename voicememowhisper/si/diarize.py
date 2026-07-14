#!/usr/bin/env python3
"""PoC 02: Speaker diarization with pyannote.audio Community-1.

Goal
----
Given an audio file, produce a Diarization JSON containing speaker
time segments (anonymous labels like SPEAKER_00). This is the
prerequisite for identification and merging.

Usage
-----
    python 02_diarize_pyannote.py \\
        --audio samples/example.m4a \\
        --output runs

Outputs to runs/<recording_id>/:
    diarization_pyannote.json       (contracts.Diarization)
    diarization_pyannote.meta.json  (contracts.StageMeta)

Dependencies
------------
    pip install pyannote.audio==4.0.*
    Model "pyannote/speaker-diarization-community-1" must be
    accessible (either pre-cached or via HF token).

Notes
-----
- The torchcodec warning from pyannote.audio at import time is
  harmless on macOS; we preload audio via torchaudio/soundfile into
  an in-memory waveform dict to bypass it.
- CPU inference on Apple Silicon. No MPS for pyannote 4.x yet.
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

# Silence the torchcodec warning at import time — we work around it
# with an in-memory waveform dict.
warnings.filterwarnings("ignore", message="torchcodec is not installed")

from . import contracts
from .progress import DiarizeProgressHook, StageProgress


def peak_rss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def load_waveform(audio_path: Path, target_sr: int = 16000) -> dict:
    """Load audio into an in-memory {'waveform': Tensor, 'sample_rate': int}
    dict. pyannote accepts this format directly and we avoid torchaudio's
    broken torchcodec backend on this machine.

    Decodes via ffmpeg CLI → temp wav → wave stdlib → numpy → torch.
    """
    import subprocess
    import tempfile
    import wave

    import numpy as np
    import torch

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        # ffmpeg: any input → 16 kHz mono PCM_S16LE wav
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(audio_path),
                "-ac",
                "1",
                "-ar",
                str(target_sr),
                "-f",
                "wav",
                str(tmp_path),
            ],
            check=True,
        )
        with wave.open(str(tmp_path), "rb") as w:
            sr = w.getframerate()
            nframes = w.getnframes()
            sampwidth = w.getsampwidth()
            raw = w.readframes(nframes)
        dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
        dtype = dtype_map[sampwidth]
        arr = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        # Normalize integer PCM to [-1, 1]
        arr /= float(1 << (8 * sampwidth - 1))
        waveform = torch.from_numpy(arr).unsqueeze(0)  # shape (1, T)
        return {"waveform": waveform, "sample_rate": sr}
    finally:
        tmp_path.unlink(missing_ok=True)


def run_diarization(
    audio_path: Path,
    model_name: str,
    hf_token: str | None,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
    embeddings_out_path: Path | None = None,
    progress: StageProgress | None = None,
):
    """Run pyannote diarization.

    Returns (Diarization, duration_sec, speaker_label_order).
    Optionally dumps per-speaker centroid embeddings to `embeddings_out_path`
    as a .npz with keys = ordered speaker labels.

    When ``progress`` is supplied, pyannote's internal sub-steps
    (segmentation → embeddings → clustering) are surfaced through
    ``DiarizeProgressHook`` so the caller sees sub-step boundaries and
    throttled progress inside each sub-step.
    """
    from .._lock import acquire_compute_lock

    acquire_compute_lock(what="the pyannote diarization pipeline")
    # Lazy import so that --help works without ML stack loaded.
    import numpy as np

    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(model_name, token=hf_token)

    audio_dict = load_waveform(audio_path)
    duration = audio_dict["waveform"].shape[1] / audio_dict["sample_rate"]

    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers
    if progress is not None:
        kwargs["hook"] = DiarizeProgressHook(progress)

    result = pipeline(audio_dict, **kwargs)
    # pyannote 4.0 returns DiarizeOutput dataclass:
    #   - speaker_diarization:            Annotation (may overlap)
    #   - exclusive_speaker_diarization:  Annotation (non-overlapping, better for transcript merge)
    #   - speaker_embeddings:             np.ndarray (num_speakers, dim), may be None
    exclusive = result.exclusive_speaker_diarization
    speaker_labels_ordered = list(exclusive.labels())

    # Convert to contracts.SpeakerSegment using the exclusive version
    segments: list[contracts.SpeakerSegment] = []
    for turn, _track, speaker in exclusive.itertracks(yield_label=True):
        segments.append(
            contracts.SpeakerSegment(
                start=round(float(turn.start), 3),
                end=round(float(turn.end), 3),
                label=str(speaker),
                confidence=None,
            )
        )

    # Dump per-speaker embeddings to .npz if provided.
    # Keys are label strings, values are 1-D centroids.
    if embeddings_out_path is not None and result.speaker_embeddings is not None:
        emb = np.asarray(result.speaker_embeddings)
        # speaker_embeddings rows follow speaker_diarization.labels() order
        labels_for_emb = list(result.speaker_diarization.labels())
        if emb.shape[0] != len(labels_for_emb):
            print(
                f"[02_diarize] warn: embeddings rows={emb.shape[0]} "
                f"but labels={len(labels_for_emb)}; aligning by min length",
                file=sys.stderr,
            )
        n = min(emb.shape[0], len(labels_for_emb))
        arrays = {labels_for_emb[i]: emb[i] for i in range(n)}
        embeddings_out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(embeddings_out_path), **arrays)

    diar = contracts.Diarization(
        recording_id=audio_path.stem,
        backend="pyannote",
        model=model_name,
        num_speakers=len(speaker_labels_ordered),
        segments=segments,
    )
    return diar, duration, speaker_labels_ordered


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio", required=True, type=Path)
    ap.add_argument(
        "--model",
        default="pyannote/speaker-diarization-community-1",
        help="pyannote model identifier",
    )
    ap.add_argument("--output", default="runs", type=Path)
    ap.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"),
        help="HuggingFace access token; defaults to HF_TOKEN env var or cached ~/.cache/huggingface/token",
    )
    ap.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="pin exact speaker count (overrides min/max)",
    )
    ap.add_argument("--min-speakers", type=int, default=None)
    ap.add_argument("--max-speakers", type=int, default=None)
    args = ap.parse_args()

    if not args.audio.exists():
        print(f"error: audio not found: {args.audio}", file=sys.stderr)
        return 2

    recording_id = args.audio.stem
    out_dir = args.output / recording_id
    out_dir.mkdir(parents=True, exist_ok=True)
    diar_path = out_dir / "diarization_pyannote.json"
    meta_path = out_dir / "diarization_pyannote.meta.json"
    emb_path = out_dir / "diarization_pyannote.embeddings.npz"

    print(f"[02_diarize] audio:   {args.audio}")
    print(f"[02_diarize] model:   {args.model}")
    print(f"[02_diarize] output:  {out_dir}")
    if args.num_speakers:
        print(f"[02_diarize] num_speakers pinned to {args.num_speakers}")
    elif args.min_speakers or args.max_speakers:
        print(
            f"[02_diarize] speaker range: "
            f"min={args.min_speakers} max={args.max_speakers}"
        )
    print(f"[02_diarize] starting... (loading model and audio)")

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()

    try:
        diar, duration, speaker_labels_ordered = run_diarization(
            audio_path=args.audio,
            model_name=args.model,
            hf_token=args.hf_token,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            embeddings_out_path=emb_path,
        )
    except Exception as e:
        print(f"[02_diarize] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        raise

    elapsed = time.perf_counter() - t0
    peak = peak_rss_bytes()

    diar.to_json(diar_path)

    meta = contracts.StageMeta(
        stage_id="02_diarize_pyannote",
        backend="pyannote",
        model=args.model,
        started_at=started_at,
        elapsed_sec=round(elapsed, 2),
        peak_rss_bytes=peak,
        config={
            "duration_sec": round(duration, 3),
            "num_speakers_hint": args.num_speakers,
            "min_speakers": args.min_speakers,
            "max_speakers": args.max_speakers,
        },
    )
    meta.to_json(meta_path)

    print()
    print(f"[02_diarize] done in {elapsed:.1f}s, peak RSS {peak / 1e9:.2f} GB")
    print(f"[02_diarize] speakers detected: {diar.num_speakers}")
    print(f"[02_diarize] segments: {len(diar.segments)}")
    print(f"[02_diarize] diarization → {diar_path}")
    print(f"[02_diarize] meta        → {meta_path}")
    if emb_path.exists():
        print(f"[02_diarize] embeddings → {emb_path} (per-speaker centroids)")

    if diar.segments:
        print()
        print("[02_diarize] first 5 segments:")
        for seg in diar.segments[:5]:
            print(f"  [{seg.start:7.2f} → {seg.end:7.2f}] {seg.label}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
