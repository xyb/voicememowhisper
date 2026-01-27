from __future__ import annotations

import argparse
import logging
import time
import shutil
from datetime import datetime
from dataclasses import replace
from pathlib import Path

from .config import Settings, load_settings, DEFAULT_ARCHIVE_PATH, DEFAULT_TRANSCRIPT_PATH, DEFAULT_STATE_DB_PATH
from .listing import collect_recordings, format_list_output

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
    logging.basicConfig(
        # Keep root level permissive; filter controls what is printed.
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    root = logging.getLogger()
    for handler in root.handlers:
        handler.addFilter(_VerbosityFilter(verbosity))


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

    # Archiving configuration
    if args.archive_dir:
        overrides["archive_dir"] = Path(args.archive_dir).expanduser()
        overrides["archive_enabled"] = True
    elif args.archive:
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


def _list_recordings(settings: Settings) -> int:
    try:
        items = collect_recordings(settings)
    except Exception:
        return 1

    output = format_list_output(items)
    print(output, end="")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Transcribe Apple Voice Memos with WhisperKit.")
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
    parser.add_argument("--list", action="store_true", help="List available recordings and exit.")
    parser.add_argument("--archive", action="store_true", help="Enable archiving of processed recordings.")
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
        return _list_recordings(settings)

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
