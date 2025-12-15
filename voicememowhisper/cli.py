from __future__ import annotations

import argparse
import logging
import time
from dataclasses import replace

from .config import Settings, load_settings, DEFAULT_ARCHIVE_PATH, DEFAULT_TRANSCRIPT_PATH
from .metadata import list_voice_memos, resolve_created_at
from .service import VoiceMemoService
from .state import StateStore

LOGGER = logging.getLogger("cli")


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


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
    if seconds is None:
        return "-"
    minutes, rem = divmod(int(seconds), 60)
    if minutes:
        return f"{minutes}m{rem:02d}s"
    return f"{rem}s"


def _list_recordings(settings: Settings) -> int:
    store: StateStore | None = None
    state_map: dict[str, tuple[Optional[Path], Optional[Path]]] = {}
    try:
        store = StateStore(settings.state_db)
        for guid in store.known_guids():
            state_map[guid] = store.get_state(guid)
    except Exception as err:
        LOGGER.warning("Unable to read state database: %s", err)
    finally:
        if store is not None:
            store.close()

    try:
        memos = list_voice_memos(settings)
    except Exception as err:
        LOGGER.error("Failed to list recordings: %s", err)
        return 1

    if not memos:
        logging.info("No recordings found in %s", settings.recordings_dir)
        return 0

    print("/-- Transcribed")
    print("|/- Archived")
    print(f"{'T':<1}{'A':<1}  {'When':19}  {'Duration':8}  Title")
    for memo in memos:
        created = resolve_created_at(memo)
        when = created.strftime("%Y-%m-%d %H:%M:%S") if created else "unknown"
        title = (memo.title or "").strip() or memo.guid
        duration = _format_duration(memo.duration_seconds)

        transcript_path, archived_path = state_map.get(memo.guid, (None, None))
        transcribed_status = "✓" if transcript_path else "."
        archived_status = "✓" if archived_path else "."

        print(f"{transcribed_status:<1}{archived_status:<1}  {when:19}  {duration:<8}  {title}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Transcribe Apple Voice Memos with WhisperKit.")
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
    _configure_logging(args.log_level)

    try:
        settings = build_settings(args)
    except Exception as err:
        LOGGER.error("%s", err)
        return 1

    if args.list:
        return _list_recordings(settings)

    try:
        service = VoiceMemoService(settings)
    except Exception as err:
        LOGGER.error("%s", err)
        return 1

    try:
        service.start(watch=args.watch)
        if args.watch:
            logging.info("Backlog synced. Watching for new recordings. Press Ctrl+C to exit.")
            while True:
                time.sleep(1)
        else:
            service.join()
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    finally:
        service.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
