from __future__ import annotations

import argparse
import logging
import sys
import time
import shutil
from datetime import datetime
from dataclasses import replace
from pathlib import Path

from .config import Settings, load_settings, DEFAULT_ARCHIVE_PATH, DEFAULT_TRANSCRIPT_PATH, DEFAULT_STATE_DB_PATH
from .listing import collect_recordings, format_list_output
from . import listing as listing_mod
from .state import StateStore

LOGGER = logging.getLogger("cli")


class _VerbosityFilter(logging.Filter):
    """
    Allow INFO/DEBUG logs based on -v/-vv.

    - WARNING/ERROR/CRITICAL always shown
    - INFO shown when verbosity >= record.verbosity (default 1)
    - DEBUG shown when verbosity >= record.verbosity (default 2)
    """

    def __init__(self, verbosity: int) -> None:
        super().__init__()
        self._verbosity = verbosity

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        required = getattr(record, "verbosity", None)
        if required is None:
            required = 2 if record.levelno <= logging.DEBUG else 1
        return self._verbosity >= int(required)


def _configure_logging(level: str, verbosity: int) -> None:
    # Force line-buffered stdout/stderr so log lines flush immediately even
    # when redirected to a file (`> run.log 2>&1`). Without this, Python uses
    # 4 KB block buffering on non-tty streams and progress is invisible for
    # minutes during long-running stages.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass

    logging.basicConfig(
        # Keep root level permissive; filter controls what is printed.
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    root = logging.getLogger()
    for handler in root.handlers:
        handler.addFilter(_VerbosityFilter(verbosity))
        # Wrap emit() to flush after every record, in case the underlying
        # stream falls back to block buffering.
        if isinstance(handler, logging.StreamHandler):
            _orig_emit = handler.emit
            def _flushing_emit(record, _orig=_orig_emit, _h=handler):
                _orig(record)
                try:
                    _h.flush()
                except Exception:
                    pass
            handler.emit = _flushing_emit  # type: ignore[assignment]


def build_settings(args: argparse.Namespace) -> Settings:
    settings = load_settings()
    overrides = {}
    if args.model:
        overrides["whisperkit_model"] = args.model
    if args.language:
        overrides["language"] = args.language
    if args.newest_first is not None:
        overrides["processing_order"] = "newest-first" if args.newest_first else "oldest-first"
    
    if args.transcript_dir:
        overrides["transcript_dir"] = Path(args.transcript_dir).expanduser()

    # Archiving configuration (default enabled; use --no-archive to disable)
    overrides["archive_enabled"] = bool(getattr(args, "archive", True))
    if args.archive_dir:
        overrides["archive_dir"] = Path(args.archive_dir).expanduser()
        overrides["archive_enabled"] = True

    if overrides:
        settings = replace(settings, **overrides)
    return settings


def _format_duration(seconds: float | None) -> str:
    # Backward-compatible wrapper; formatting lives in listing module.
    if seconds is None:
        return "-"
    minutes, rem = divmod(int(seconds), 60)
    if minutes:
        return f"{minutes}m{rem:02d}s"
    return f"{rem}s"


def _parse_filename(path: Path) -> tuple[str | None, str | None]:
    # Backward-compatible wrapper kept for tests; actual parsing lives in listing module.
    from .listing import _parse_filename as _parse  # local import to avoid cycles

    return _parse(path)


def _list_recordings(settings: Settings, *, limit: int = 10) -> int:
    try:
        items = collect_recordings(settings)
    except Exception:
        return 1

    visible_all = [i for i in items if (i.has_source or i.has_transcript or i.has_archive)]
    total = len(visible_all)
    visible = visible_all
    if limit > 0:
        visible = visible[:limit]

    LOGGER.debug(
        "List summary: total=%d shown=%d limit=%d",
        total,
        len(visible),
        limit,
        extra={"verbosity": 2},
    )

    # Fill in duration for archive-only / inbox-imported files (best-effort).
    # Probe only for the items we are actually going to show to keep --list fast.
    probed = 0
    updated = 0
    store: StateStore | None = None
    for item in visible:
        if item.duration is not None:
            continue
        if not item.has_archive or item.archive_path is None:
            continue
        try:
            seconds = listing_mod.probe_audio_duration_seconds(item.archive_path)
        except Exception as err:
            LOGGER.debug(
                "Duration probe failed for %s: %s",
                item.archive_path.name,
                err,
                extra={"verbosity": 2},
            )
            seconds = None
        if seconds is not None:
            item.duration = seconds
            probed += 1
            # Persist best-effort duration back to our state DB to avoid re-probing next time.
            try:
                if store is None:
                    store = StateStore(settings.state_db)
                updated += store.set_duration_for_archived_path(item.archive_path, seconds)
            except Exception as err:
                LOGGER.debug(
                    "Failed to persist duration for %s: %s",
                    item.archive_path.name,
                    err,
                    extra={"verbosity": 2},
                )

    if probed:
        LOGGER.debug(
            "Filled duration from audio probe for %d item(s) (persisted=%d)",
            probed,
            updated,
            extra={"verbosity": 2},
        )
    if store is not None:
        store.close()

    shown = len(visible)
    output = format_list_output(visible, shown=shown, total=total)
    print(output, end="")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Route the `si` subcommand group to the speaker-id CLI before argparse
    # touches the rest of the args. Keeping it as a pre-dispatch (rather than
    # a subparser) means existing flat flags (`--watch`, `-l`, ...) keep
    # working unchanged — this is the user-facing entry, no migration needed.
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    if raw_argv and raw_argv[0] == "si":
        from .si.cli import main as si_main
        return si_main(raw_argv[1:])

    parser = argparse.ArgumentParser(
        description="Transcribe Apple Voice Memos with WhisperKit (default) or the speaker-id pipeline.",
        epilog=(
            "Subcommands:\n"
            "  si <subcommand>   Speaker-ID pipeline (transcribe → diarize → identify → merge → render).\n"
            "                    Run `voicememo-whisper si --help` or `voicememo-whisper si steps` for details."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v for info, -vv for debug).",
    )
    parser.add_argument("--watch", action="store_true", help="Keep running and watch for new recordings.")
    parser.add_argument(
        "--model", help="WhisperKit model identifier (default from env or 'large-v3-v20240930_turbo')."
    )
    parser.add_argument("--language", help="Language hint for Whisper (e.g. 'en', 'zh').")
    parser.add_argument("-l", "--list", action="store_true", help="List available recordings and exit.")
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=10,
        help="For --list: number of items to show (default: 10; 0 for all).",
    )
    parser.add_argument(
        "--archive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable archiving of processed recordings (default: true). Use --no-archive to disable.",
    )
    parser.add_argument(
        "--archive-dir", 
        help=f"Directory to archive audio files (implies --archive). Defaults to '{DEFAULT_ARCHIVE_PATH}' or VOICE_MEMO_ARCHIVE_DIR env var."
    )
    parser.add_argument(
        "--transcript-dir",
        help=f"Directory to save transcripts. Defaults to '{DEFAULT_TRANSCRIPT_PATH}' or VOICE_MEMO_TRANSCRIPT_DIR env var."
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING...). Default: INFO.",
    )
    parser.add_argument(
        "--newest-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Process backlog newest first (disable for oldest first). Default: true.",
    )

    args = parser.parse_args(argv)
    _configure_logging(args.log_level, args.verbose)

    try:
        settings = build_settings(args)
    except Exception as err:
        LOGGER.error("%s", err)
        return 1

    if args.list:
        return _list_recordings(settings, limit=args.limit)

    # Long-running processing: guard against a second instance racing on
    # the same Voice Memos source / state DB / archive dir.
    from ._lock import single_instance_lock
    with single_instance_lock():
        try:
            from .service import VoiceMemoService
            service = VoiceMemoService(settings)
        except Exception as err:
            LOGGER.error("%s", err)
            return 1

        try:
            service.start(watch=args.watch)
            if args.watch:
                logging.info(
                    "Backlog synced. Watching for new recordings. Press Ctrl+C to exit.",
                    extra={"verbosity": 1},
                )
                while True:
                    time.sleep(1)
            else:
                service.join()
        except KeyboardInterrupt:
            logging.info("Interrupted by user.", extra={"verbosity": 1})
        finally:
            service.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
