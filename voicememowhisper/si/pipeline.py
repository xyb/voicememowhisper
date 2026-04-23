"""Speaker-ID pipeline orchestrator.

Stages (run in this order; each step's intermediate is cached on disk):
    1. transcribe → runs/<id>/transcript_faster_whisper.json
    2. diarize    → runs/<id>/diarization_pyannote.json + .embeddings.npz
    3. identify   → runs/<id>/identification.json
    4. merge      → runs/<id>/merged.json
    5. render     → outputs/<id>/transcript.md (+ transcript.txt)

`run()` accepts `from_step` / `to_step` / `force`:
    - from_step: skip stages < N. Their cached outputs must already exist.
    - to_step:   stop after stage N (don't run later ones).
    - force:     delete the cached output of each in-window stage before
                 running, so the stage actually re-executes.

Stages are deliberately serial — every step is CPU + multi-GB RAM and
parallel execution OOMs on Apple Silicon (see memory rule
feedback_speaker_id_no_parallel).
"""

from __future__ import annotations

import argparse
import resource
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import contracts  # stdlib only — safe to import at module top.

_SCRIPT_DIR = Path(__file__).resolve().parent

# Heavy ML modules (numpy / torch / pyannote / faster-whisper) are imported
# lazily inside `run()` so that `voicememo-whisper si --help` / `si steps` /
# `si library list` work even without the [speaker-id] extra installed.

# Single source of truth for stage order, names, and CLI help text.
STEPS: list[tuple[int, str, str]] = [
    (1, "transcribe", "Whisper turns audio into a JSON transcript with per-segment timestamps."),
    (2, "diarize",    "pyannote splits audio into per-speaker segments + embedding vectors."),
    (3, "identify",   "Match each anonymous speaker against speaker-library centroids."),
    (4, "merge",      "Align transcript segments with speaker labels/names into one structure."),
    (5, "render",     "Emit transcript.md (speaker turns + ⚠️) and transcript.txt."),
]
STEP_NAMES = [name for _, name, _ in STEPS]
STEP_BY_NAME = {name: num for num, name, _ in STEPS}
STEP_BY_NUM = {num: name for num, name, _ in STEPS}


def resolve_step(value: str | int) -> int:
    """Accept '1', 1, or 'transcribe' and return the canonical step number."""
    if isinstance(value, int):
        if value not in STEP_BY_NUM:
            raise ValueError(f"step out of range: {value} (valid: 1-{len(STEPS)})")
        return value
    s = str(value).strip().lower()
    if s.isdigit():
        return resolve_step(int(s))
    if s in STEP_BY_NAME:
        return STEP_BY_NAME[s]
    raise ValueError(
        f"unknown step: {value!r} (valid names: {', '.join(STEP_NAMES)}; or 1-{len(STEPS)})"
    )


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def _human_duration(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m}m{s:02d}s"


def _stage_header(num: int, name: str) -> str:
    # Match the `[i/N name]` prefix the StageProgress helper emits, so a
    # `grep -E '\[[0-9]+/' run.log` catches every stage-level line
    # regardless of whether it came from pipeline.py or progress.py.
    return f"[{num}/{len(STEPS)} {name}]"


def run(
    audio_path: Path,
    *,
    model: str = "medium",
    language: str = "zh",
    compute_type: str = "int8",
    library_dir: Path | None = None,
    threshold: float = 0.5,
    runs_dir: Path | None = None,
    output_dir: Path | None = None,
    output_transcript_dir: Path | None = None,
    skip_identify: bool = False,
    from_step: int | str = 1,
    to_step: int | str = 5,
    force: bool = False,
    recording_id: str | None = None,
    asr_backend: str = "faster_whisper",
    asr_backend_config: dict | None = None,
    diarize_backend: str = "local_pyannote",
    diarize_backend_config: dict | None = None,
    archive_dir: Path | None = None,
    archive: bool = True,
) -> Path | None:
    """Run the speaker-id pipeline.

    Returns the path to transcript.md if stage 5 ran, else None.

    ``recording_id`` overrides the default (``audio_path.stem``) used to
    locate ``runs/<id>/`` cache. Useful when the audio has been archived
    and its filename no longer matches the cache directory — pass the
    original stem so late stages (identify / merge / render) can still
    consume the cached transcript / diarization / embeddings.

    ``asr_backend`` picks the transcription engine for stage 1:

    - ``"faster_whisper"`` (default): in-process Python, uses ``model`` /
      ``compute_type`` args. Cache file: ``transcript_faster_whisper.json``.
    - ``"openai-audio"``: HTTP call to any OpenAI Audio API compatible
      server (OpenAI Whisper, a self-hosted FunASR slim, groq, ...).
      Requires ``asr_backend_config`` with at least ``url`` and ``model``;
      optional ``api_key`` / ``host_header`` / ``language`` /
      ``response_format`` / ``timeout_sec``. Cache file:
      ``transcript_openai_audio.json``. The ``model`` / ``compute_type``
      args above are ignored for this backend.

    ``diarize_backend`` picks the diarization engine for stage 2:

    - ``"local_pyannote"`` (default): in-process pyannote.audio on
      CPU/MPS. 20+ minutes on a 23-min recording on Mac CPU.
    - ``"http"``: HTTP call to the self-hosted pyannote service.
      Requires ``diarize_backend_config`` with at least ``url``;
      optional ``host_header`` / ``api_key`` / ``include_embeddings``
      / ``num_speakers`` / ``min_speakers`` / ``max_speakers`` /
      ``timeout_sec``. Cache file: ``diarization_pyannote.json`` (same
      as local — the output schema is identical).
    """
    from_n = resolve_step(from_step)
    to_n = resolve_step(to_step)
    if from_n > to_n:
        raise ValueError(f"from_step ({from_n}) must be <= to_step ({to_n})")

    # Lazy ML imports — keep `si --help` working without the extra installed.
    from . import speaker_embed
    from . import diarize as mod_diarize
    from . import identify as mod_identify
    from . import merge as mod_merge
    from . import render as mod_render
    from . import transcribe as mod_transcribe

    if library_dir is None:
        library_dir = _SCRIPT_DIR / "speaker-library"
    if runs_dir is None:
        runs_dir = _SCRIPT_DIR / "runs"
    if output_dir is None:
        output_dir = _SCRIPT_DIR / "outputs"

    audio_path = Path(audio_path).resolve()
    if recording_id is None:
        recording_id = audio_path.stem
    run_dir = runs_dir / recording_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Cache file name is keyed by backend so different engines' outputs
    # can coexist under runs/<id>/ without clobbering each other. Hyphens
    # in backend ids ("openai-audio") become underscores so the filename
    # stays shell-friendly.
    _backend_fname = asr_backend.replace("-", "_")
    transcript_path = run_dir / f"transcript_{_backend_fname}.json"
    diarization_path = run_dir / "diarization_pyannote.json"
    embeddings_path = run_dir / "diarization_pyannote.embeddings.npz"
    identification_path = run_dir / "identification.json"
    merged_path = run_dir / "merged.json"

    total_start = time.monotonic()
    print("=" * 60)
    print(f"[pipeline] recording: {recording_id}")
    print(f"[pipeline] audio:     {audio_path}")
    if asr_backend == "faster_whisper":
        print(f"[pipeline] asr:       faster_whisper / {model} (compute_type={compute_type})")
    else:
        cfg = asr_backend_config or {}
        cfg_model = cfg.get("model", "?")
        cfg_url = cfg.get("url", "?")
        print(f"[pipeline] asr:       {asr_backend} / {cfg_model} @ {cfg_url}")
    if diarize_backend == "local_pyannote":
        print(f"[pipeline] diarize:   local_pyannote / community-1")
    else:
        dcfg = diarize_backend_config or {}
        print(f"[pipeline] diarize:   {diarize_backend} @ {dcfg.get('url', '?')}")
    print(f"[pipeline] library:   {library_dir}")
    print(f"[pipeline] steps:     {from_n} ({STEP_BY_NUM[from_n]}) → "
          f"{to_n} ({STEP_BY_NUM[to_n]}){'  [force]' if force else ''}")
    print("=" * 60)
    print()

    transcript_obj: Optional[contracts.Transcript] = None
    diar_obj: Optional[contracts.Diarization] = None
    identification_obj: Optional[contracts.Identification] = None
    merged_obj: Optional[contracts.MergedTranscript] = None
    md_path: Optional[Path] = None

    # ── Stage 1: Transcribe ──────────────────────────────────────────
    if from_n <= 1 <= to_n:
        from .progress import StageProgress  # lazy so `si --help` stays light

        if force and transcript_path.exists():
            transcript_path.unlink()
        if transcript_path.exists():
            print(f"{_stage_header(1,'transcribe')} cached → {transcript_path}")
            transcript_obj = contracts.Transcript.from_json(transcript_path)
        elif asr_backend == "faster_whisper":
            # ffprobe duration lets StageProgress render a real ETA bar.
            # If ffprobe is missing or fails, total=None degrades to an
            # indeterminate spinner; the model's own info.duration is
            # still written into the Transcript JSON after the run.
            audio_duration = mod_transcribe.probe_audio_duration(audio_path)
            with StageProgress(
                "transcribe", stage_num=1, total_stages=len(STEPS),
                total=audio_duration,
            ) as prog:
                prog.note(f"model={model}")
                transcript_obj, _info = mod_transcribe.transcribe(
                    audio_path,
                    model_name=model,
                    language=language,
                    compute_type=compute_type,
                    word_timestamps=False,
                    vad_filter=True,
                    progress=prog,
                )
                transcript_obj.to_json(transcript_path)
                prog.note(
                    f"{len(transcript_obj.segments)} segments, "
                    f"{_peak_rss_mb():.1f} MB RSS"
                )
        elif asr_backend in ("openai-audio", "openai_audio"):
            # HTTP backend. Progress bar stays indeterminate because the
            # server doesn't stream partial segments back — we just send
            # the audio, wait, get the full result.
            from .asr_backends import openai_audio as mod_openai_audio

            cfg_dict = dict(asr_backend_config or {})
            if "url" not in cfg_dict or "model" not in cfg_dict:
                raise ValueError(
                    "asr_backend='openai-audio' requires asr_backend_config "
                    "with at least 'url' and 'model'"
                )
            # Per-backend language default: config override > pipeline-level language.
            cfg_dict.setdefault("language", language)
            cfg = mod_openai_audio.OpenAIAudioConfig(**cfg_dict)
            with StageProgress(
                "transcribe", stage_num=1, total_stages=len(STEPS),
            ) as prog:
                prog.note(f"backend=openai-audio model={cfg.model} url={cfg.url}")
                transcript_obj, raw_info = mod_openai_audio.transcribe(
                    audio_path, cfg
                )
                transcript_obj.to_json(transcript_path)
                prog.note(
                    f"{len(transcript_obj.segments)} segments in "
                    f"{raw_info['wall_clock_sec']}s, "
                    f"{_peak_rss_mb():.1f} MB RSS"
                )
        else:
            raise ValueError(f"unknown asr_backend: {asr_backend!r}")
        print()
    elif 2 <= to_n:
        # Need transcript downstream — load cached.
        _require_cache(transcript_path, 1)
        transcript_obj = contracts.Transcript.from_json(transcript_path)

    # ── Stage 2: Diarize ─────────────────────────────────────────────
    if from_n <= 2 <= to_n:
        from .progress import StageProgress  # lazy so `si --help` stays light

        if force:
            for p in (diarization_path, embeddings_path):
                if p.exists():
                    p.unlink()
        if diarization_path.exists():
            print(f"{_stage_header(2,'diarize')} cached → {diarization_path}")
            diar_obj = contracts.Diarization.from_json(diarization_path)
        elif diarize_backend == "local_pyannote":
            # pyannote has no linear time-axis the bar can track against,
            # so we leave total=None — the DiarizeProgressHook surfaces
            # sub-step boundaries + per-substep counters instead.
            with StageProgress(
                "diarize", stage_num=2, total_stages=len(STEPS),
            ) as prog:
                diar_obj, duration_sec, _labels = mod_diarize.run_diarization(
                    audio_path,
                    model_name="pyannote/speaker-diarization-community-1",
                    hf_token=None,
                    num_speakers=None,
                    min_speakers=None,
                    max_speakers=None,
                    embeddings_out_path=embeddings_path,
                    progress=prog,
                )
                diar_obj.to_json(diarization_path)
                if duration_sec and transcript_obj is not None and transcript_obj.duration_sec == 0:
                    transcript_obj.duration_sec = duration_sec
                    transcript_obj.to_json(transcript_path)
                prog.note(
                    f"{diar_obj.num_speakers} speakers, "
                    f"{len(diar_obj.segments)} segments, "
                    f"{_peak_rss_mb():.1f} MB RSS"
                )
        elif diarize_backend == "http":
            from .diarize_backends import http as mod_diarize_http

            cfg_dict = dict(diarize_backend_config or {})
            if "url" not in cfg_dict:
                raise ValueError(
                    "diarize_backend='http' requires diarize_backend_config "
                    "with at least 'url'"
                )
            cfg = mod_diarize_http.DiarizeHTTPConfig(**cfg_dict)
            with StageProgress(
                "diarize", stage_num=2, total_stages=len(STEPS),
            ) as prog:
                prog.note(f"backend=http url={cfg.url}")
                diar_obj, duration_sec, _labels = mod_diarize_http.run_diarization(
                    audio_path, cfg, embeddings_out_path=embeddings_path,
                )
                diar_obj.to_json(diarization_path)
                if duration_sec and transcript_obj is not None and transcript_obj.duration_sec == 0:
                    transcript_obj.duration_sec = duration_sec
                    transcript_obj.to_json(transcript_path)
                wall = getattr(diar_obj, "_wall_clock_sec", None)
                infer = getattr(diar_obj, "_infer_sec", None)
                prog.note(
                    f"{diar_obj.num_speakers} speakers, "
                    f"{len(diar_obj.segments)} segments, "
                    f"{_peak_rss_mb():.1f} MB RSS"
                    + (f", wall {wall}s" if wall else "")
                    + (f", infer {infer}s" if infer else "")
                )
        else:
            raise ValueError(f"unknown diarize_backend: {diarize_backend!r}")
        print()
    elif 3 <= to_n:
        _require_cache(diarization_path, 2)
        diar_obj = contracts.Diarization.from_json(diarization_path)

    # ── Stage 3: Identify ────────────────────────────────────────────
    # Pre-load identification from cache if downstream stages need it but
    # this stage isn't being re-run.
    if from_n <= 3 <= to_n:
        if force and identification_path.exists():
            identification_path.unlink()
        if identification_path.exists():
            print(f"{_stage_header(3,'identify')} cached → {identification_path}")
            identification_obj = contracts.Identification.from_json(identification_path)
        elif skip_identify:
            print(f"{_stage_header(3,'identify')} skipped: skip_identify=True")
        elif not library_dir.exists():
            print(f"{_stage_header(3,'identify')} skipped: library not found ({library_dir})")
        else:
            assert diar_obj is not None
            library = mod_identify.load_library(library_dir)
            if not library:
                print(f"{_stage_header(3,'identify')} skipped: speaker library is empty")
            else:
                print(f"{_stage_header(3,'identify')} starting "
                      f"(library: {len(library)} speakers, threshold={threshold})...")
                t0 = time.monotonic()
                label_centroids = mod_identify.get_label_centroids_from_npz(diar_obj, embeddings_path)
                if not label_centroids:
                    print(f"{_stage_header(3,'identify')} no cached embeddings, computing from audio...")
                    embedder = speaker_embed.CommunityOneEmbedder()
                    label_centroids = mod_identify.compute_label_centroids_from_audio(
                        diar_obj, audio_path, embedder
                    )
                assignments = mod_identify.match_labels(label_centroids, library, threshold)
                identification_obj = contracts.Identification(
                    recording_id=recording_id,
                    backend="pyannote_embedding",
                    model="pyannote/speaker-diarization-community-1",
                    threshold=threshold,
                    assignments=assignments,
                )
                identification_obj.to_json(identification_path)
                resolved = sum(1 for a in assignments if a.name is not None)
                elapsed = time.monotonic() - t0
                print(f"{_stage_header(3,'identify')} done: {resolved}/{len(assignments)} identified, "
                      f"{_human_duration(elapsed)}")
                for a in assignments:
                    tag = f"→ {a.name}" if a.name else "→ (unknown)"
                    print(f"  {a.label}: {tag} (cosine={a.confidence})")
        print()
    elif 4 <= to_n and identification_path.exists():
        identification_obj = contracts.Identification.from_json(identification_path)

    # ── Stage 4: Merge ───────────────────────────────────────────────
    if from_n <= 4 <= to_n:
        if force and merged_path.exists():
            merged_path.unlink()
        assert transcript_obj is not None and diar_obj is not None
        print(f"{_stage_header(4,'merge')} starting...")
        t0 = time.monotonic()
        pipeline_name = f"faster-whisper({model}) + pyannote-community-1"
        if identification_obj:
            pipeline_name += " + speaker-library"
        merged_obj = mod_merge.merge(
            transcript_obj, diar_obj, identification_obj, pipeline_name=pipeline_name,
        )
        merged_obj.to_json(merged_path)
        elapsed = time.monotonic() - t0
        groups = mod_render.group_consecutive(merged_obj.segments)
        print(f"{_stage_header(4,'merge')} done: {len(merged_obj.segments)} segments → "
              f"{len(groups)} speaker blocks, {_human_duration(elapsed)}")
        if merged_obj.unresolved_labels:
            print(f"{_stage_header(4,'merge')} unresolved: {merged_obj.unresolved_labels}")
        print()
    elif 5 <= to_n:
        _require_cache(merged_path, 4)
        merged_obj = contracts.MergedTranscript.from_json(merged_path)

    # ── Stage 5: Render ──────────────────────────────────────────────
    if from_n <= 5 <= to_n:
        assert merged_obj is not None
        out_dir = output_dir / recording_id
        out_dir.mkdir(parents=True, exist_ok=True)

        generated_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        md_content = mod_render.render(merged_obj, generated_at)
        md_path = out_dir / "transcript.md"
        md_path.write_text(md_content, encoding="utf-8")
        shutil.copy2(merged_path, out_dir / "merged.json")

        groups = mod_render.group_consecutive(merged_obj.segments)
        print(f"{_stage_header(5,'render')} → {md_path}")
        print(f"{_stage_header(5,'render')} {len(groups)} speaker blocks, "
              f"{md_path.stat().st_size / 1024:.1f} KB")

        txt_content = _render_plain_text(merged_obj)
        txt_path = out_dir / "transcript.txt"
        txt_path.write_text(txt_content, encoding="utf-8")

        if output_transcript_dir:
            # Decouple the user-facing copy name from the cache key. For
            # fresh runs they match (audio.stem == recording_id). For
            # re-runs against an archived file via --recording-id, the
            # cache key stays put but the copy uses the audio's current
            # stem — which is the canonical vault filename in practice.
            # This avoids the manual `mv <cache-key>.md <canonical>.md`
            # step after every --recording-id re-render.
            output_transcript_dir = Path(output_transcript_dir)
            output_transcript_dir.mkdir(parents=True, exist_ok=True)
            copy_stem = audio_path.stem
            target_md = output_transcript_dir / f"{copy_stem}.md"
            target_txt = output_transcript_dir / f"{copy_stem}.txt"
            shutil.copy2(md_path, target_md)
            shutil.copy2(txt_path, target_txt)
            print(f"[pipeline] copied → {target_md}")
            print(f"[pipeline] copied → {target_txt}")

    # ── Auto-archive source audio ────────────────────────────────────
    # Run only when stage 5 succeeded (so we know the recording was processed
    # end-to-end) and archive=True. For workflows that don't run the watcher
    # daemon, `si run` is the only entry that can keep Inbox/ from growing
    # forever. Skip if source already lives under the archive dir, or if dest
    # exists (idempotent).
    if archive and to_n >= 5 and md_path is not None:
        if archive_dir is None:
            archive_dir = Path.home() / "Documents/VoiceMemoWhisper/Audio"
        archive_dir = Path(archive_dir)
        try:
            already_archived = audio_path.resolve().is_relative_to(archive_dir.resolve())
        except (AttributeError, ValueError):
            # is_relative_to is 3.9+; on older Pythons or non-overlapping paths
            already_archived = str(audio_path.resolve()).startswith(str(archive_dir.resolve()))
        if not already_archived:
            archive_dir.mkdir(parents=True, exist_ok=True)
            target = archive_dir / audio_path.name
            if target.exists():
                print(f"[archive] target exists, skipping → {target}")
            else:
                shutil.move(str(audio_path), str(target))
                print(f"[archive] moved → {target}")

    total_elapsed = time.monotonic() - total_start
    print()
    print("=" * 60)
    print(f"[pipeline] DONE in {_human_duration(total_elapsed)}  "
          f"(steps {from_n}–{to_n})")
    print(f"[pipeline] peak RSS: {_peak_rss_mb():.1f} MB")
    if md_path is not None:
        print(f"[pipeline] outputs:  {md_path.parent}/")
    print("=" * 60)
    return md_path


def _require_cache(path: Path, step_num: int) -> None:
    if not path.exists():
        name = STEP_BY_NUM[step_num]
        raise FileNotFoundError(
            f"Stage {step_num} ({name}) output not found: {path}\n"
            f"Run that step first (e.g. `voicememo-whisper si {name} <audio>`),\n"
            f"or use `voicememo-whisper si run --from {name} ...` to start earlier."
        )


def _render_plain_text(merged: contracts.MergedTranscript) -> str:
    """Render a plain-text version (speaker labels + text, no timestamps)."""
    from . import render as mod_render
    groups = mod_render.group_consecutive(merged.segments)
    lines: list[str] = []
    for group in groups:
        label = group[0].speaker_label
        name = next((s.speaker_name for s in group if s.speaker_name), None) or label
        texts = [s.text.strip() for s in group if s.text.strip()]
        body = "".join(texts) if any("。" in t or "," in t for t in texts) else " ".join(texts)
        lines.append(f"[{name}] {body}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    """Standalone entry: `python -m voicememowhisper.si.pipeline`.

    For day-to-day use, prefer `voicememo-whisper si run` which has the
    same flags and is the documented entry point.
    """
    ap = argparse.ArgumentParser(
        description="Run the speaker-id pipeline (transcribe → diarize → identify → merge → render).",
    )
    ap.add_argument("audio", type=Path, help="Path to audio file (.m4a, .wav, .mp3)")
    ap.add_argument("--model", default="medium", help="Whisper model (default: medium)")
    ap.add_argument("--language", default="zh", help="Language code (default: zh)")
    ap.add_argument("--compute-type", default="int8", help="Compute type (default: int8)")
    ap.add_argument("--library", type=Path, default=None, help="Speaker library directory")
    ap.add_argument("--threshold", type=float, default=0.5, help="Cosine threshold (default: 0.5)")
    ap.add_argument("--runs", type=Path, default=None, help="Runs directory for intermediates")
    ap.add_argument("--output", type=Path, default=None, help="Output directory for final artifacts")
    ap.add_argument("--output-transcript", type=Path, default=None,
                    help="Copy final transcript to this directory")
    ap.add_argument("--skip-identify", action="store_true", help="Skip speaker identification")
    ap.add_argument("--from-step", default="1",
                    help=f"Start at this step (name or 1-{len(STEPS)}). Default: 1.")
    ap.add_argument("--to-step", default=str(len(STEPS)),
                    help=f"Stop after this step (name or 1-{len(STEPS)}). Default: {len(STEPS)}.")
    ap.add_argument("--force", action="store_true",
                    help="Re-run in-window steps even if cached output exists.")
    ap.add_argument("--recording-id", default=None,
                    help="Override the cache-dir key (default: audio.stem). "
                         "Use when the audio has been archived/renamed but you "
                         "want to reuse the original runs/<id>/ cache.")
    args = ap.parse_args()

    if not args.audio.exists():
        print(f"error: audio file not found: {args.audio}", file=sys.stderr)
        return 2

    try:
        run(
            args.audio,
            model=args.model,
            language=args.language,
            compute_type=args.compute_type,
            library_dir=args.library,
            threshold=args.threshold,
            runs_dir=args.runs,
            output_dir=args.output,
            output_transcript_dir=args.output_transcript,
            skip_identify=args.skip_identify,
            from_step=args.from_step,
            to_step=args.to_step,
            force=args.force,
            recording_id=args.recording_id,
        )
    except KeyboardInterrupt:
        print("\n[pipeline] interrupted", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\n[pipeline] error: {e}", file=sys.stderr)
        raise

    return 0


if __name__ == "__main__":
    sys.exit(main())
