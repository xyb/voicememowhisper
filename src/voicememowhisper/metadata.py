from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import logging
import sqlite3
from pathlib import Path
from typing import Iterable, List

from .config import Settings, load_settings

LOGGER = logging.getLogger(__name__)
MAC_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class VoiceMemo:
    guid: str
    path: Path
    title: str | None = None
    created_at: datetime | None = None
    duration_seconds: float | None = None


def _to_datetime(value: float | int | None) -> datetime | None:
    if value is None:
        return None
    try:
        return MAC_EPOCH + timedelta(seconds=float(value))
    except Exception:  # pragma: no cover
        LOGGER.debug("Failed to convert %s to datetime", value, exc_info=True)
        return None


def _pick(row: sqlite3.Row, candidates: Iterable[str]) -> str | None:
    for name in candidates:
        if name in row.keys():
            value = row[name]
            if value not in (None, ""):
                return value
    return None


def load_voice_memos(settings: Settings | None = None) -> dict[str, VoiceMemo]:
    """Load Voice Memo metadata keyed by GUID."""
    settings = settings or load_settings()
    db_path = settings.metadata_db

    if not db_path.exists():
        LOGGER.warning("Metadata database not found at %s", db_path)
        return {}

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
    except sqlite3.Error as err:  # pragma: no cover - sqlite permissions
        LOGGER.error("Unable to open metadata database %s: %s", db_path, err)
        return {}

    with conn:
        try:
            rows = conn.execute("SELECT * FROM ZVOICE")
        except sqlite3.Error as err:
            LOGGER.error("Failed to query metadata table: %s", err)
            return {}

        memos: dict[str, VoiceMemo] = {}
        for row in rows:
            guid = _pick(row, ("ZUUID", "ZIDENTIFIER", "Z_PK"))
            if not guid:
                continue
            guid = str(guid)
            memo = VoiceMemo(
                guid=guid,
                path=settings.recordings_dir / f"{guid}.m4a",
                title=_pick(row, ("ZTITLE", "ZNAME", "ZGENERICNAME")),
                created_at=_to_datetime(_pick(row, ("ZCREATIONDATE", "ZDATE"))),
                duration_seconds=float(_pick(row, ("ZDURATION", "ZLENGTH"))) if _pick(row, ("ZDURATION", "ZLENGTH")) else None,
            )
            memos[guid] = memo
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
        else:
            memo = VoiceMemo(guid=guid, path=path)
        results.append(memo)

    return results
