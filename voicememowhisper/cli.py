from __future__ import annotations

import argparse
import logging
import time
import shutil
import re
from datetime import datetime
from dataclasses import replace
from pathlib import Path

from .config import Settings, load_settings, DEFAULT_ARCHIVE_PATH, DEFAULT_TRANSCRIPT_PATH, DEFAULT_STATE_DB_PATH
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


def _parse_filename(path: Path) -> tuple[str | None, str | None]:
    """Parse timestamp and title from filename."""
    stem = path.stem
    # Format: YYYY-MM-DD_HH-MM-SS_Title
    match = re.match(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(.*)$", stem)
    if match:
        timestamp_str, title = match.groups()
        try:
            dt = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
            return dt.isoformat(), title
        except ValueError:
            pass
    
    if stem.startswith("undated_"):
        return None, stem[8:]
    
    return None, stem


def _list_recordings(settings: Settings) -> int:
    store: StateStore | None = None
    db_rows: list[dict] = []
    
    try:
        store = StateStore(settings.state_db)
        db_rows = [dict(r) for r in store.get_all_processed()]
    except Exception as err:
        LOGGER.warning("Unable to read state database: %s", err)
    finally:
        if store is not None:
            store.close()

    try:
        source_memos = list_voice_memos(settings)
    except Exception as err:
        LOGGER.error("Failed to list recordings: %s", err)
        return 1

    # Data structure to hold merged items
    # Key: GUID (if known) or Filename Stem (for orphans)
    all_items: dict[str, dict] = {}

    def get_item(key: str) -> dict:
        if key not in all_items:
            all_items[key] = {
                'key': key,
                'created_at': None,
                'duration': None,
                'title': None,
                't': False, 'a': False, 's': False,
            }
        return all_items[key]

    # Map filenames to GUIDs based on DB to link files back to known records
    filename_to_guid: dict[str, str] = {}
    stem_to_guid: dict[str, str] = {}
    db_records_map: dict[str, dict] = {}
    
    for record in db_rows:
        guid = record['guid']
        db_records_map[guid] = record
        if record['transcript_path']:
            p = Path(record['transcript_path'])
            filename_to_guid[p.name] = guid
            stem_to_guid[p.stem] = guid
        if record['archived_path']:
            p = Path(record['archived_path'])
            filename_to_guid[p.name] = guid
            stem_to_guid[p.stem] = guid

    # --- Phase 1: Process Source Memos (App) ---
    for memo in source_memos:
        item = get_item(memo.guid)
        item['created_at'] = resolve_created_at(memo)
        item['duration'] = memo.duration_seconds
        item['title'] = (memo.title or "").strip() or memo.guid
        item['s'] = True

    # --- Phase 2: Scan Directories for Files ---
    def process_file(path: Path, type_key: str):
        filename = path.name
        stem = path.stem
        
        # Try exact filename match first, then stem match
        guid = filename_to_guid.get(filename) or stem_to_guid.get(stem)
        
        if guid:
            # Matches a known DB record
            item = get_item(guid)
            item[type_key] = True
        else:
            # Orphan file
            item = get_item(stem)
            item[type_key] = True
            
        # Extract metadata from filename if missing
        parsed_dt_str, parsed_title = _parse_filename(path)
        if not item['created_at']:
            if parsed_dt_str:
                try:
                    item['created_at'] = datetime.fromisoformat(parsed_dt_str)
                except ValueError:
                    pass
        
        # Fallback to file mtime if still missing
        if not item['created_at']:
            item['created_at'] = datetime.fromtimestamp(path.stat().st_mtime)
        
        if not item['title']:
            item['title'] = parsed_title or stem

    if settings.transcript_dir.exists():
        for f in settings.transcript_dir.glob("*.txt"):
            process_file(f, 't')

    if settings.archive_dir and settings.archive_dir.exists():
        for f in settings.archive_dir.glob("*.m4a"):
            process_file(f, 'a')

    # --- Phase 3: Enrich with DB Metadata ---
    # Only for items that already exist (from App or Files)
    for guid, record in db_records_map.items():
        if guid in all_items:
            item = all_items[guid]
            # Backfill if missing
            if not item['created_at'] and record['created_at']:
                try: item['created_at'] = datetime.fromisoformat(record['created_at'])
                except ValueError: pass
            
            if not item['duration'] and record['duration']:
                item['duration'] = record['duration']
                
            if not item['title'] and record['title']:
                item['title'] = record['title']

    # --- Phase 4: Finalize and Sort ---
    display_list = list(all_items.values())
    
    if not display_list:
        logging.info("No recordings found.")
        return 0

    def to_naive(dt: datetime | None) -> datetime:
        if dt is None:
            return datetime.min
        return dt.replace(tzinfo=None)

    # Sort items by string representation of naive datetime for stability
    def sort_key(x):
        dt = x['created_at']
        if dt is None:
            return ""
        return str(to_naive(dt))

    display_list.sort(key=sort_key, reverse=True)
    
    # Print header
    print("/-- Transcribed")
    print("|/-- Archived")
    print("||/-- Source Exists")
    print(f"{'T':<1}{'A':<1}{'S':<1}  {'When':19}  {'Duration':8}  Title")

    for item in display_list:
        if not (item['s'] or item['t'] or item['a']):
            continue

        when = item['created_at'].strftime("%Y-%m-%d %H:%M:%S") if item['created_at'] else "unknown"
        duration_str = _format_duration(item['duration'])
        
        t_char = "✓" if item['t'] else "."
        a_char = "✓" if item['a'] else "."
        s_char = "✓" if item['s'] else "x"
        
        print(f"{t_char:<1}{a_char:<1}{s_char:<1}  {when:19}  {duration_str:<8}  {item['title'] or item['key']}")

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
