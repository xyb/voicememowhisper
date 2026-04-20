#!/usr/bin/env python3
"""PoC 01: Transcribe audio with faster-whisper into a structured Transcript.

Goal
----
Replace the current "one big blob of text" output with a structured
Transcript JSON that carries segment-level timestamps, punctuation, and
optional word-level timestamps. This is the prerequisite for every other
stage in the speaker-id pipeline.

Usage
-----
    python 01_transcribe_faster_whisper.py \\
        --audio samples/example.m4a \\
        --model large-v3 \\
        --language zh \\
        --output runs

Outputs two files under runs/<recording_id>/:
    transcript_faster_whisper.json   (contracts.Transcript)
    transcript_faster_whisper.meta.json   (contracts.StageMeta)

Dependencies
------------
    pip install faster-whisper

Install notes
-------------
faster-whisper pulls CTranslate2 + onnxruntime + numpy. On Apple Silicon
it runs on CPU by default; large-v3 on M-series CPU is slow but accurate.
For PoC we accept the wait in exchange for quality.
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import contracts
from .progress import StageProgress


def human_duration(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m}m{s:02d}s"


def peak_rss_bytes() -> int:
    """Return peak RSS of current process in bytes (macOS: ru_maxrss is bytes)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def probe_audio_duration(audio_path: Path) -> float | None:
    """Best-effort: get duration via ffprobe, fall back to None."""
    import shutil
    import subprocess

    if not shutil.which("ffprobe"):
        return None
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(out.stdout.strip())
    except Exception:
        return None


def transcribe(
    audio_path: Path,
    model_name: str,
    language: str | None,
    compute_type: str,
    word_timestamps: bool,
    vad_filter: bool,
    progress: StageProgress | None = None,
) -> tuple[contracts.Transcript, dict]:
    """Run faster-whisper and return a Transcript plus raw info dict.

    When ``progress`` is supplied, each segment emitted by the model
    updates the bar at ``seg.end`` (audio seconds processed). Callers
    that build the ``StageProgress`` outside — typically ``pipeline.run``
    — do so with ``total=<audio duration>`` so the bar carries a
    meaningful ETA. Without ``progress`` the iteration runs silently
    (same behaviour as before this flag existed).
    """
    from faster_whisper import WhisperModel

    # CPU on Apple Silicon. GPU would be "cuda" on Linux boxes.
    model = WhisperModel(model_name, device="cpu", compute_type=compute_type)

    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=word_timestamps,
        vad_filter=vad_filter,
    )

    # faster-whisper returns a generator; consume it.
    segments: list[contracts.Segment] = []
    for seg in segments_iter:
        words = None
        if word_timestamps and seg.words:
            words = [
                contracts.Word(
                    start=w.start,
                    end=w.end,
                    text=w.word,
                    probability=getattr(w, "probability", None),
                )
                for w in seg.words
            ]
        segments.append(
            contracts.Segment(
                start=float(seg.start),
                end=float(seg.end),
                text=seg.text.strip(),
                words=words,
                avg_logprob=getattr(seg, "avg_logprob", None),
                no_speech_prob=getattr(seg, "no_speech_prob", None),
            )
        )
        if progress is not None:
            progress.update(float(seg.end))

    recording_id = audio_path.stem
    tr = contracts.Transcript(
        recording_id=recording_id,
        backend="faster_whisper",
        model=model_name,
        language=info.language,
        duration_sec=float(info.duration),
        segments=segments,
    )
    return tr, {
        "language_probability": float(info.language_probability),
        "compute_type": compute_type,
        "vad_filter": vad_filter,
        "word_timestamps": word_timestamps,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio", required=True, type=Path, help="input audio file")
    ap.add_argument(
        "--model",
        default="large-v3",
        help="faster-whisper model name (tiny/base/small/medium/large-v3)",
    )
    ap.add_argument(
        "--language",
        default="zh",
        help="language hint, or 'auto' to let the model detect",
    )
    ap.add_argument(
        "--compute-type",
        default="int8",
        help="int8 / int8_float16 / float16 / float32",
    )
    ap.add_argument(
        "--output",
        default="runs",
        type=Path,
        help="output root directory (will create <output>/<recording_id>/)",
    )
    ap.add_argument(
        "--no-word-timestamps",
        action="store_true",
        help="skip word-level timestamps (faster, less data)",
    )
    ap.add_argument(
        "--no-vad",
        action="store_true",
        help="disable VAD filter (default: enabled)",
    )
    args = ap.parse_args()

    if not args.audio.exists():
        print(f"error: audio not found: {args.audio}", file=sys.stderr)
        return 2

    recording_id = args.audio.stem
    out_dir = args.output / recording_id
    out_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = out_dir / "transcript_faster_whisper.json"
    meta_path = out_dir / "transcript_faster_whisper.meta.json"

    lang = None if args.language.lower() == "auto" else args.language
    audio_duration = probe_audio_duration(args.audio)

    print(f"[01_transcribe] audio:   {args.audio}")
    print(f"[01_transcribe] recording_id: {recording_id}")
    print(f"[01_transcribe] model:   {args.model} ({args.compute_type})")
    print(f"[01_transcribe] language: {lang or 'auto'}")
    if audio_duration:
        print(f"[01_transcribe] duration: {human_duration(audio_duration)}")
    print(f"[01_transcribe] output:  {out_dir}")
    print(f"[01_transcribe] starting... (first run downloads model weights)")

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()

    try:
        transcript, raw_info = transcribe(
            audio_path=args.audio,
            model_name=args.model,
            language=lang,
            compute_type=args.compute_type,
            word_timestamps=not args.no_word_timestamps,
            vad_filter=not args.no_vad,
        )
    except Exception as e:
        print(f"[01_transcribe] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        raise

    elapsed = time.perf_counter() - t0
    peak = peak_rss_bytes()

    transcript.to_json(transcript_path)

    meta = contracts.StageMeta(
        stage_id="01_transcribe_faster_whisper",
        backend="faster_whisper",
        model=args.model,
        started_at=started_at,
        elapsed_sec=round(elapsed, 2),
        peak_rss_bytes=peak,
        config={
            "language": lang,
            "compute_type": args.compute_type,
            "word_timestamps": not args.no_word_timestamps,
            "vad_filter": not args.no_vad,
            **raw_info,
        },
    )
    meta.to_json(meta_path)

    print()
    print(f"[01_transcribe] done in {elapsed:.1f}s, peak RSS {peak / 1e9:.2f} GB")
    print(f"[01_transcribe] detected language: {transcript.language}")
    print(f"[01_transcribe] segments: {len(transcript.segments)}")
    print(f"[01_transcribe] transcript → {transcript_path}")
    print(f"[01_transcribe] meta       → {meta_path}")

    # Quick preview
    if transcript.segments:
        print()
        print("[01_transcribe] first 3 segments:")
        for seg in transcript.segments[:3]:
            print(f"  [{seg.start:7.2f} → {seg.end:7.2f}] {seg.text}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
