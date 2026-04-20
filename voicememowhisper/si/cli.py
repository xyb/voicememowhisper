"""`voicememo-whisper si` subcommand group.

User-facing entry into the speaker-ID pipeline. Each pipeline stage
gets its own subcommand; the description in `--help` shows the step
number, so the processing order is visible at the command line without
baking position numbers into the subcommand name itself.

Subcommands:
    transcribe   Step 1/5
    diarize      Step 2/5
    identify     Step 3/5
    merge        Step 4/5
    render       Step 5/5
    run          Run a contiguous range of steps (--from / --to / --force)
    library      Manage the speaker-library (list / add / rebuild)
    inspect      Show cached intermediate state for a recording
    steps        Print the ordered list of pipeline steps and exit

The heavy ML imports (faster-whisper, pyannote.audio, torch) are
deferred to the actual step invocation, so `si --help`, `si steps`,
and `si library list` work even before the optional `[speaker-id]`
extra is installed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Lightweight imports only at module top.
from .pipeline import STEPS, STEP_BY_NAME, STEP_BY_NUM, STEP_NAMES, resolve_step

DEFAULT_RUNS = Path.home() / ".local" / "share" / "voicememowhisper" / "speaker-id" / "runs"
DEFAULT_OUTPUT = Path.home() / ".local" / "share" / "voicememowhisper" / "speaker-id" / "outputs"
DEFAULT_LIBRARY = Path.home() / ".local" / "share" / "voicememowhisper" / "speaker-library"


def _step_help(num: int) -> str:
    name = STEP_BY_NUM[num]
    desc = next(d for n, _, d in STEPS if n == num)
    return f"Step {num}/{len(STEPS)} — {desc}"


# ───────── Subcommand handlers ────────────────────────────────────────


def _cmd_single_step(step_num: int, args: argparse.Namespace) -> int:
    """Run exactly one stage. Always forces re-run (single-step is the
    explicit way to redo a stage; use `run` to leverage caching)."""
    audio = _resolve_audio(args)
    from .pipeline import run
    run(
        audio,
        model=getattr(args, "model", "medium") or "medium",
        language=getattr(args, "language", "zh") or "zh",
        compute_type=getattr(args, "compute_type", "int8") or "int8",
        library_dir=_path_or_default(getattr(args, "library", None), DEFAULT_LIBRARY),
        threshold=getattr(args, "threshold", 0.5),
        runs_dir=_path_or_default(getattr(args, "runs", None), DEFAULT_RUNS),
        output_dir=_path_or_default(getattr(args, "output", None), DEFAULT_OUTPUT),
        output_transcript_dir=_path_or_none(getattr(args, "output_transcript", None)),
        skip_identify=getattr(args, "skip_identify", False),
        from_step=step_num,
        to_step=step_num,
        force=True,
    )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    audio = _resolve_audio(args)
    from .pipeline import run
    run(
        audio,
        model=args.model or "medium",
        language=args.language or "zh",
        compute_type=args.compute_type or "int8",
        library_dir=_path_or_default(args.library, DEFAULT_LIBRARY),
        threshold=args.threshold,
        runs_dir=_path_or_default(args.runs, DEFAULT_RUNS),
        output_dir=_path_or_default(args.output, DEFAULT_OUTPUT),
        output_transcript_dir=_path_or_none(args.output_transcript),
        skip_identify=args.skip_identify,
        from_step=args.from_step,
        to_step=args.to_step,
        force=args.force,
    )
    return 0


def _cmd_steps(_args: argparse.Namespace) -> int:
    print("Speaker-ID pipeline stages:\n")
    for num, name, desc in STEPS:
        print(f"  {num}. {name:<10s}  {desc}")
    print(
        "\nRun a single stage:   voicememo-whisper si <name> <audio>"
        "\nRun a range:          voicememo-whisper si run <audio> --from <name> --to <name>"
        "\nForce re-run:         add --force"
    )
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    audio = _resolve_audio(args)
    runs = _path_or_default(args.runs, DEFAULT_RUNS)
    output = _path_or_default(args.output, DEFAULT_OUTPUT)
    rid = audio.stem
    run_dir = runs / rid
    out_dir = output / rid

    artifacts = [
        (1, "transcribe", run_dir / "transcript_faster_whisper.json"),
        (2, "diarize",    run_dir / "diarization_pyannote.json"),
        (2, "diarize",    run_dir / "diarization_pyannote.embeddings.npz"),
        (3, "identify",   run_dir / "identification.json"),
        (4, "merge",      run_dir / "merged.json"),
        (5, "render",     out_dir / "transcript.md"),
        (5, "render",     out_dir / "transcript.txt"),
    ]
    print(f"recording: {rid}")
    print(f"audio:     {audio}")
    print(f"runs dir:  {run_dir}")
    print(f"output:    {out_dir}")
    print()
    print(f"{'step':<6}{'stage':<12}{'state':<10}{'size':>10}  path")
    print("-" * 78)
    for num, name, p in artifacts:
        if p.exists():
            size = f"{p.stat().st_size / 1024:>9.1f}K"
            state = "cached"
        else:
            size = "       —"
            state = "MISSING"
        print(f"{num:<6}{name:<12}{state:<10}{size}  {p}")
    return 0


def _cmd_library(args: argparse.Namespace) -> int:
    library = _path_or_default(args.library, DEFAULT_LIBRARY)

    if args.lib_action == "list":
        if not library.exists():
            print(f"speaker library does not exist: {library}", file=sys.stderr)
            return 1
        print(f"library: {library}\n")
        rows = []
        for d in sorted(p for p in library.iterdir() if p.is_dir()):
            clips = sorted(d.glob("*.wav")) + sorted(d.glob("*.m4a"))
            embedding = d / "embedding.npy"
            rows.append((d.name, len(clips), "yes" if embedding.exists() else "NO"))
        if not rows:
            print("(empty)")
            return 0
        name_w = max(len(r[0]) for r in rows)
        print(f"{'speaker':<{name_w}}  {'clips':>5}  embedding")
        for name, n_clips, has_emb in rows:
            print(f"{name:<{name_w}}  {n_clips:>5}  {has_emb}")
        return 0

    if args.lib_action == "add":
        speaker = args.speaker
        clip = Path(args.clip).expanduser().resolve()
        if not clip.exists():
            print(f"clip not found: {clip}", file=sys.stderr)
            return 2
        target_dir = library / speaker
        target_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(target_dir.glob("[0-9][0-9][0-9].*"))
        next_num = (int(existing[-1].stem) + 1) if existing else 1
        target = target_dir / f"{next_num:03d}{clip.suffix.lower()}"
        shutil.copy2(clip, target)
        print(f"copied {clip} → {target}")
        if not args.no_rebuild:
            return _rebuild_speaker(library, speaker)
        print("(skipping embedding rebuild; rerun with `library rebuild` to update)")
        return 0

    if args.lib_action == "rebuild":
        return _rebuild_speaker(library, args.speaker)

    return 0


def _rebuild_speaker(library: Path, speaker: str | None) -> int:
    """Recompute embedding.npy for one speaker (or all if speaker is None)."""
    from . import library as mod_library  # lazy: needs numpy + pyannote
    code = mod_library.run_build(library_dir=library, speaker_filter=speaker)
    return code


# ───────── Helpers ────────────────────────────────────────────────────


def _resolve_audio(args: argparse.Namespace) -> Path:
    audio = Path(args.audio).expanduser().resolve()
    if not audio.exists():
        raise SystemExit(f"error: audio file not found: {audio}")
    return audio


def _path_or_default(value, default: Path) -> Path:
    if value is None:
        default.mkdir(parents=True, exist_ok=True)
        return default
    return Path(value).expanduser().resolve()


def _path_or_none(value):
    return Path(value).expanduser().resolve() if value else None


# ───────── Argparse plumbing ──────────────────────────────────────────


def _add_common_audio_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("audio", help="Path to audio file (.m4a, .wav, .mp3)")
    p.add_argument("--runs", help=f"Runs directory for intermediates (default: {DEFAULT_RUNS})")
    p.add_argument("--output", help=f"Final outputs directory (default: {DEFAULT_OUTPUT})")
    p.add_argument("--library", help=f"Speaker library directory (default: {DEFAULT_LIBRARY})")
    p.add_argument("--output-transcript",
                   help="Copy final transcript.md/.txt to this directory (e.g. ~/Documents/VoiceMemoWhisper/Transcripts)")


def _add_stage_specific_args(name: str, p: argparse.ArgumentParser) -> None:
    if name == "transcribe":
        p.add_argument("--model", default="medium",
                       help="Whisper model: tiny / base / small / medium / large-v3 (default: medium)")
        p.add_argument("--language", default="zh",
                       help="Language hint, or 'auto' (default: zh)")
        p.add_argument("--compute-type", default="int8",
                       help="int8 / int8_float16 / float16 / float32 (default: int8)")
    if name == "identify":
        p.add_argument("--threshold", type=float, default=0.5,
                       help="Cosine match threshold (default: 0.5)")
    if name == "merge":
        # merge needs identify-related model name only for the pipeline tag string
        p.add_argument("--model", default="medium", help="(only used in pipeline label)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="voicememo-whisper si",
        description=(
            "Speaker-ID pipeline. Runs in 5 ordered stages "
            "(transcribe → diarize → identify → merge → render). "
            "Each stage caches its output; subcommands let you replay one stage, "
            "or `run` a contiguous range. Use `si steps` for the full ordered list."
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    # Per-stage subcommands.
    for num, name, _desc in STEPS:
        sp = sub.add_parser(name, help=_step_help(num),
                            description=_step_help(num))
        _add_common_audio_args(sp)
        _add_stage_specific_args(name, sp)
        if name == "identify":
            sp.add_argument("--skip-identify", action="store_true",
                            help="Run as no-op (skip identification)")
        sp.set_defaults(_handler=lambda a, n=num: _cmd_single_step(n, a))

    # `run` — contiguous range.
    sr = sub.add_parser("run", help="Run a range of steps (with stage caching)",
                        description="Run a contiguous range of steps. Cached intermediates "
                                    "are reused unless --force is given.")
    _add_common_audio_args(sr)
    sr.add_argument("--model", default="medium",
                    help="Whisper model (default: medium)")
    sr.add_argument("--language", default="zh", help="Language hint (default: zh)")
    sr.add_argument("--compute-type", default="int8",
                    help="Whisper compute type (default: int8)")
    sr.add_argument("--threshold", type=float, default=0.5,
                    help="Identify cosine threshold (default: 0.5)")
    sr.add_argument("--from", dest="from_step", default="1",
                    help=f"Start step. Name (one of {', '.join(STEP_NAMES)}) or number 1-{len(STEPS)}. Default: 1.")
    sr.add_argument("--to", dest="to_step", default=str(len(STEPS)),
                    help=f"End step (inclusive). Same format as --from. Default: {len(STEPS)}.")
    sr.add_argument("--force", action="store_true",
                    help="Re-run in-window steps even if cached output exists.")
    sr.add_argument("--skip-identify", action="store_true",
                    help="Skip the identify stage (treats it as no-op)")
    sr.set_defaults(_handler=_cmd_run)

    # `library`.
    sl = sub.add_parser("library", help="Manage the speaker-library",
                        description="List, add to, or rebuild the speaker-library used for identification.")
    sl.add_argument("--library", help=f"Speaker library directory (default: {DEFAULT_LIBRARY})")
    lsub = sl.add_subparsers(dest="lib_action", required=True, metavar="ACTION")
    lsub.add_parser("list", help="List speakers and their clip / embedding state")
    sla = lsub.add_parser("add", help="Add a clip to a speaker, then rebuild that speaker's embedding")
    sla.add_argument("speaker", help="Speaker id (directory name under the library)")
    sla.add_argument("clip", help="Path to a .wav (or .m4a) clip to enroll")
    sla.add_argument("--no-rebuild", action="store_true",
                     help="Don't rebuild the embedding after copying")
    slr = lsub.add_parser("rebuild", help="Recompute centroid embeddings")
    slr.add_argument("--speaker", help="Restrict to a single speaker id (default: all speakers)")
    sl.set_defaults(_handler=_cmd_library)

    # `inspect`.
    si = sub.add_parser("inspect",
                        help="Show cached intermediate state for a recording",
                        description="List which pipeline artifacts exist for a given audio file.")
    _add_common_audio_args(si)
    si.set_defaults(_handler=_cmd_inspect)

    # `steps`.
    sst = sub.add_parser("steps", help="Print the ordered list of pipeline steps",
                         description="Print the ordered list of pipeline steps and exit.")
    sst.set_defaults(_handler=_cmd_steps)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args._handler(args)
    except KeyboardInterrupt:
        print("\nsi: interrupted", file=sys.stderr)
        return 130
    except SystemExit:
        raise
    except Exception as e:
        print(f"\nsi: error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
