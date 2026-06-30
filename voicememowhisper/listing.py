from __future__ import annotations

import logging
import re
import subprocess
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from .config import Settings
from .list_model import RecordingItem
from .metadata import list_voice_memos, resolve_created_at
from .naming import dedup_key, normalize_title, to_naive as naming_to_naive
from .state import StateStore

LOGGER = logging.getLogger("listing")

# Examples:
# - "estimated duration: 7113.492000 sec"
# - "duration: 42.000000 sec"
_AFINFO_DURATION_RE = re.compile(
    r"(?:estimated\s+)?duration:\s*([0-9]+(?:\.[0-9]+)?)\s*sec",
    re.IGNORECASE,
)


def probe_audio_duration_seconds(path: Path) -> float | None:
    """
    Best-effort duration probe for local audio files.

    On macOS, use the built-in `afinfo` tool to avoid extra dependencies.
    """
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            ["/usr/bin/afinfo", str(path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=3.0,
        )
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        LOGGER.debug(
            "afinfo timed out for %s in %.1fms",
            path.name,
            (time.perf_counter() - start) * 1000.0,
            extra={"verbosity": 2},
        )
        return None
    except Exception:
        return None

    if proc.returncode != 0:
        LOGGER.debug(
            "afinfo failed (rc=%s) for %s in %.1fms",
            proc.returncode,
            path.name,
            (time.perf_counter() - start) * 1000.0,
            extra={"verbosity": 2},
        )
        return None

    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    match = _AFINFO_DURATION_RE.search(text)
    if not match:
        LOGGER.debug(
            "afinfo output missing duration for %s in %.1fms",
            path.name,
            (time.perf_counter() - start) * 1000.0,
            extra={"verbosity": 2},
        )
        return None
    try:
        seconds = float(match.group(1))
        LOGGER.debug(
            "afinfo duration=%.3fs for %s in %.1fms",
            seconds,
            path.name,
            (time.perf_counter() - start) * 1000.0,
            extra={"verbosity": 2},
        )
        return seconds
    except ValueError:
        return None


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    minutes, rem = divmod(int(seconds), 60)
    if minutes:
        return f"{minutes}m{rem:02d}s"
    return f"{rem}s"


def _parse_filename(path: Path) -> tuple[str | None, str | None]:
    """
    Parse timestamp and title from filename.

    Returns (iso_timestamp_str, title) when timestamped, otherwise (None, title).
    """
    stem = path.stem
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


def collect_recordings(settings: Settings) -> list[RecordingItem]:
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
        raise

    all_items: dict[str, RecordingItem] = {}

    def get_item(key: str) -> RecordingItem:
        if key not in all_items:
            all_items[key] = RecordingItem(key=key)
        return all_items[key]

    filename_to_guid: dict[str, str] = {}
    stem_to_guid: dict[str, str] = {}
    db_records_map: dict[str, dict] = {}

    for record in db_rows:
        guid = record["guid"]
        db_records_map[guid] = record
        if record.get("transcript_path"):
            p = Path(record["transcript_path"])
            filename_to_guid[p.name] = guid
            stem_to_guid[p.stem] = guid
        if record.get("archived_path"):
            p = Path(record["archived_path"])
            filename_to_guid[p.name] = guid
            stem_to_guid[p.stem] = guid

    # Phase 1: Source memos from app
    for memo in source_memos:
        item = get_item(memo.guid)
        item.created_at = resolve_created_at(memo)
        item.duration = memo.duration_seconds
        item.title = (memo.title or "").strip() or memo.guid
        item.has_source = True

    # Phase 2: Scan transcript/archive directories
    def process_file(path: Path, *, kind: str) -> None:
        filename = path.name
        stem = path.stem

        guid = filename_to_guid.get(filename) or stem_to_guid.get(stem)
        if guid:
            item = get_item(guid)
        else:
            item = get_item(stem)

        if kind == "t":
            item.has_transcript = True
            item.transcript_path = path
        elif kind == "a":
            item.has_archive = True
            item.archive_path = path
        elif kind == "inbox":
            # Inbox files are sources awaiting processing — same status bit as
            # Voice Memos sources, so they show in -l / --dry-run as pending.
            item.has_source = True

        dt_str, parsed_title = _parse_filename(path)
        if item.created_at is None:
            if dt_str:
                try:
                    item.created_at = datetime.fromisoformat(dt_str)
                except ValueError:
                    pass
        if item.created_at is None:
            item.created_at = datetime.fromtimestamp(path.stat().st_mtime)
        if not item.title:
            item.title = parsed_title or stem

    if settings.transcript_dir.exists():
        for f in settings.transcript_dir.glob("*.txt"):
            process_file(f, kind="t")

    if settings.archive_dir and settings.archive_dir.exists():
        for f in settings.archive_dir.glob("*.m4a"):
            process_file(f, kind="a")

    # Phase 2.5: Inbox files awaiting processing (imported, not yet
    # transcribed/archived). Without this they were invisible to -l /
    # --dry-run and got missed every time. Mirrors InboxProcessor.scan's
    # extension set; dedup paths since macOS globbing is case-insensitive.
    if settings.inbox_dir:
        try:
            inbox_exists = settings.inbox_dir.exists()
        except Exception:
            inbox_exists = False
        if inbox_exists:
            from .inbox import DEFAULT_INBOX_EXTENSIONS
            seen_inbox: set[Path] = set()
            for ext in DEFAULT_INBOX_EXTENSIONS:
                for f in list(settings.inbox_dir.glob(f"*{ext}")) + list(
                    settings.inbox_dir.glob(f"*{ext.upper()}")
                ):
                    if f in seen_inbox:
                        continue
                    seen_inbox.add(f)
                    process_file(f, kind="inbox")

    # Phase 3: Enrich from DB metadata when item exists
    for guid, record in db_records_map.items():
        if guid not in all_items:
            continue
        item = all_items[guid]
        if item.created_at is None and record.get("created_at"):
            try:
                item.created_at = datetime.fromisoformat(record["created_at"])
            except ValueError:
                pass
        if item.duration is None and record.get("duration"):
            item.duration = record["duration"]
        if not item.title and record.get("title"):
            item.title = record["title"]

    items = list(all_items.values())

    # Phase 4: Deduplicate by (timestamp to second, normalized title)
    deduped: dict[tuple[str, str], RecordingItem] = {}
    leftovers: list[RecordingItem] = []
    for item in items:
        if not item.created_at or not item.title:
            leftovers.append(item)
            continue
        when_key, title_key = dedup_key(item.created_at, item.title or item.key)
        if not when_key and not title_key:
            leftovers.append(item)
            continue
        key = (when_key, title_key)
        if key not in deduped:
            deduped[key] = item
        else:
            existing = deduped[key]
            existing.merge_flags_from(item)
            # Prefer title with source info or longer descriptive text
            if (existing.has_source is False and item.has_source is True) or (
                len(item.title or "") > len(existing.title or "")
            ):
                existing.title = item.title
            # Keep earliest created_at (should be same to second)
            if existing.created_at and item.created_at:
                a = naming_to_naive(existing.created_at)
                b = naming_to_naive(item.created_at)
                if a and b:
                    existing.created_at = a if a <= b else b

    out = list(deduped.values()) + leftovers

    def sort_key(x: RecordingItem) -> str:
        if x.created_at is None:
            return ""
        naive = x.created_at.replace(tzinfo=None)
        return str(naive)

    out.sort(key=sort_key, reverse=True)
    return out


def format_list_output(
    items: list[RecordingItem],
    *,
    shown: int | None = None,
    total: int | None = None,
) -> str:
    if not items:
        return "No recordings found.\n"

    lines: list[str] = []
    lines.append("/-- Transcribed")
    lines.append("|/-- Archived")
    lines.append("||/-- Source Exists")
    title_header = "Title"
    if shown is not None and total is not None:
        title_header = f"Title ({shown}/{total})"
    lines.append(f"{'T':<1}{'A':<1}{'S':<1}  {'When':19}  {'Duration':8}  {title_header}")

    for item in items:
        if not (item.has_source or item.has_transcript or item.has_archive):
            continue
        when = item.created_at.strftime("%Y-%m-%d %H:%M:%S") if item.created_at else "unknown"
        duration_str = _format_duration(item.duration)
        t_char = "✓" if item.has_transcript else "."
        a_char = "✓" if item.has_archive else "."
        s_char = "✓" if item.has_source else "x"
        title = item.title or item.key
        lines.append(f"{t_char:<1}{a_char:<1}{s_char:<1}  {when:19}  {duration_str:<8}  {title}")

    return "\n".join(lines) + "\n"

