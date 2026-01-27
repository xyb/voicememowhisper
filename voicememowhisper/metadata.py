from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import logging
import sqlite3
from pathlib import Path
from typing import List, Optional

from .config import Settings, load_settings
from .db_introspection import clear_caches, find_record_table
from .metadata_constants import (
    DATE_COLUMNS,
    DURATION_COLUMNS,
    GUID_COLUMNS,
    MAC_EPOCH,
    TITLE_COLUMNS,
)
from .metadata_parser import (
    is_trashed as _is_trashed,
    pick as _pick,
    resolve_path as _resolve_path,
    resolve_related_title as _resolve_related_title,
    to_datetime as _to_datetime,
)

LOGGER = logging.getLogger("metadata")
@dataclass(frozen=True)
class VoiceMemo:
    guid: str
    path: Path
    title: str | None = None
    created_at: datetime | None = None
    duration_seconds: float | None = None
    is_trashed: bool = False

def load_voice_memos(settings: Settings | None = None) -> dict[str, VoiceMemo]:
    """Load Voice Memo metadata keyed by GUID."""
    settings = settings or load_settings()
    db_path = settings.metadata_db
    fallback = settings.legacy_metadata_db

    clear_caches()

    if not db_path.exists():
        if fallback and fallback.exists():
            LOGGER.info("Primary metadata database missing; using legacy database at %s", fallback)
            db_path = fallback
        else:
            LOGGER.debug("Metadata database not found at %s", db_path)
            return {}

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.OperationalError as err:  # pragma: no cover - sqlite permissions
        message = str(err).lower()
        if "permission" in message or "authorized" in message or "authorised" in message:
            raise PermissionError(
                f"Insufficient permissions to read Voice Memo metadata at {db_path}. "
                "Grant Full Disk Access (System Settings → Privacy & Security → Full Disk Access) and re-run."
            ) from err
        LOGGER.error("Unable to open metadata database %s: %s", db_path, err)
        return {}
    except sqlite3.Error as err:  # pragma: no cover
        LOGGER.error("Unable to open metadata database %s: %s", db_path, err)
        return {}

    with conn:
        table = find_record_table(conn)
        if not table:
            LOGGER.warning("No suitable table found in metadata database %s", db_path)
            return {}

        try:
            rows = conn.execute(f"SELECT * FROM {table}")
        except sqlite3.Error as err:
            LOGGER.error("Failed to query metadata table %s: %s", table, err)
            return {}

        memos: dict[str, VoiceMemo] = {}
        for row in rows:
            guid_raw = _pick(row, GUID_COLUMNS)
            if not guid_raw:
                continue
            guid = str(guid_raw)

            path = _resolve_path(row, settings, guid)

            trashed = _is_trashed(row)

            title_value = _pick(row, TITLE_COLUMNS)
            created_value = _pick(row, DATE_COLUMNS)
            duration_value = _pick(row, DURATION_COLUMNS)

            if not title_value:
                title_value = _resolve_related_title(conn, row)

            memo = VoiceMemo(
                guid=guid,
                path=path,
                title=str(title_value) if title_value is not None else None,
                created_at=_to_datetime(created_value),
                duration_seconds=float(duration_value) if duration_value is not None else None,
                is_trashed=trashed,
            )
            memos[memo.path.stem] = memo
        return memos


def resolve_created_at(memo: VoiceMemo) -> datetime | None:
    """Return the most accurate creation time available for a memo."""
    if memo.created_at:
        return memo.created_at.astimezone(datetime.now().astimezone().tzinfo)

    try:
        stats = memo.path.stat()
    except FileNotFoundError:
        return None

    tz = datetime.now().astimezone().tzinfo
    if hasattr(stats, "st_birthtime"):
        return datetime.fromtimestamp(stats.st_birthtime, tz=tz)
    return datetime.fromtimestamp(stats.st_mtime, tz=tz)


def list_voice_memos(settings: Settings | None = None) -> List[VoiceMemo]:
    """Return Voice Memo entries for every recording on disk."""
    settings = settings or load_settings()
    memos = load_voice_memos(settings)

    results: List[VoiceMemo] = []
    seen_paths: set[str] = set()
    try:
        paths = sorted(settings.recordings_dir.glob("*.m4a"))
    except PermissionError as err:
        raise PermissionError(
            f"Unable to access {settings.recordings_dir}. Grant the terminal Full Disk Access."
        ) from err

    for path in paths:
        guid = path.stem
        memo = memos.get(guid)
        if memo:
            if memo.path != path:
                memo = replace(memo, path=path)
                memos[guid] = memo
        else:
            memo = VoiceMemo(guid=guid, path=path)
        if not memo.is_trashed and guid not in seen_paths:
            results.append(memo)
            seen_paths.add(guid)

    # Include metadata-only entries (for recently deleted files that are still present in app listing).
    for memo in memos.values():
        if memo.is_trashed:
            continue
        stem = memo.path.stem
        if stem not in seen_paths:
            results.append(memo)
            seen_paths.add(stem)

    results.sort(key=lambda m: resolve_created_at(m) or datetime.fromtimestamp(0), reverse=True)
    return results
