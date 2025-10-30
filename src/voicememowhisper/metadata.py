from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import logging
import sqlite3
from pathlib import Path
from typing import Iterable, List, Sequence

from .config import Settings, load_settings

LOGGER = logging.getLogger(__name__)
MAC_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)
GUID_COLUMNS: Sequence[str] = (
    "ZUUID",
    "ZCLOUDUUID",
    "ZIDENTIFIER",
    "ZGUID",
    "Z_PK",
)
TITLE_COLUMNS: Sequence[str] = (
    "ZTITLE",
    "ZDISPLAYTITLE",
    "ZCUSTOMLABEL",
    "ZNAME",
    "ZGENERICNAME",
)
DATE_COLUMNS: Sequence[str] = (
    "ZCREATIONDATE",
    "ZCACHEDRECORDINGDATE",
    "ZDATE",
    "ZRECORDINGDATE",
    "ZMODIFICATIONDATE",
)
DURATION_COLUMNS: Sequence[str] = (
    "ZDURATION",
    "ZCACHEDDURATION",
    "ZTRIMMEDDURATION",
    "ZCACHEDTRIMMEDDURATION",
    "ZLENGTH",
)
PATH_COLUMNS: Sequence[str] = (
    "ZRELATIVEPATH",
    "ZPATHRELATIVE",
    "ZPATH",
    "ZLOCALPATH",
    "ZLOCALIZEDPATH",
    "ZCACHEDPATH",
    "ZCACHEDFILEPATH",
    "ZFILEPATH",
    "ZCLOUDLOCALPATHRELATIVE",
    "ZCACHEDPATHRELATIVE",
    "ZPATHRELATIVESTRING",
)
TRASH_COLUMNS: Sequence[str] = (
    "ZTRASHEDDATE",
    "ZTRASHED",
    "ZMARKEDFORDELETION",
    "ZMARKEDTRASHED",
    "ZISDELETED",
    "ZISPENDINGDELETE",
    "ZINTRASH",
    "ZNEEDSDELETE",
)


@dataclass(frozen=True)
class VoiceMemo:
    guid: str
    path: Path
    title: str | None = None
    created_at: datetime | None = None
    duration_seconds: float | None = None
    is_trashed: bool = False


def _to_datetime(value: float | int | None) -> datetime | None:
    if value is None:
        return None
    try:
        return MAC_EPOCH + timedelta(seconds=float(value))
    except Exception:  # pragma: no cover
        LOGGER.debug("Failed to convert %s to datetime", value, exc_info=True)
        return None


def _truthy(value) -> bool:
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "no")
    return bool(value)


def _pick(row: sqlite3.Row, candidates: Iterable[str]):
    keys = row.keys()
    for name in candidates:
        if name in keys:
            value = row[name]
            if value not in (None, ""):
                return value
    return None


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info('{table}')")
    except sqlite3.Error:  # pragma: no cover - corrupt db
        return set()
    return {row[1] for row in rows}


def _find_record_table(conn: sqlite3.Connection) -> str | None:
    priority = [
        "ZCLOUDRECORDING",
        "ZVOICE",
        "ZRECORDING",
        "ZCLOUDRECORDINGS",
    ]
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for name in priority:
        if name in tables:
            return name
    for name in tables:
        cols = _table_columns(conn, name)
        if not cols:
            continue
        if any(col in cols for col in GUID_COLUMNS) and any(col in cols for col in TITLE_COLUMNS):
            if any(col in cols for col in DATE_COLUMNS) or any(col in cols for col in DURATION_COLUMNS):
                return name
    return None


def _resolve_path(row: sqlite3.Row, settings: Settings, guid: str) -> Path:
    keys = row.keys()
    for name in PATH_COLUMNS:
        if name in keys:
            value = row[name]
            if isinstance(value, str) and value.strip():
                candidate = value.strip()
                if candidate.startswith("file://"):
                    candidate = candidate[7:]
                if candidate.startswith("~/"):
                    return Path(candidate).expanduser()
                path = Path(candidate)
                if path.is_absolute():
                    return path
                return settings.container_root / candidate.lstrip("/")
    return settings.recordings_dir / f"{guid}.m4a"


def load_voice_memos(settings: Settings | None = None) -> dict[str, VoiceMemo]:
    """Load Voice Memo metadata keyed by GUID."""
    settings = settings or load_settings()
    db_path = settings.metadata_db
    fallback = settings.legacy_metadata_db

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
        table = _find_record_table(conn)
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

            trashed = any(
                _truthy(row[name])
                for name in TRASH_COLUMNS
                if name in row.keys()
            )

            title_value = _pick(row, TITLE_COLUMNS)
            created_value = _pick(row, DATE_COLUMNS)
            duration_value = _pick(row, DURATION_COLUMNS)

            memo = VoiceMemo(
                guid=guid,
                path=path,
                title=str(title_value) if title_value is not None else None,
                created_at=_to_datetime(created_value),
                duration_seconds=float(duration_value) if duration_value is not None else None,
                is_trashed=trashed,
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
    seen: set[str] = set()
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
        if not memo.is_trashed and guid not in seen:
            results.append(memo)
            seen.add(guid)

    # Include metadata-only entries (for recently deleted files that are still present in app listing).
    for memo in memos.values():
        if memo.is_trashed:
            continue
        if memo.guid not in seen:
            results.append(memo)
            seen.add(memo.guid)

    results.sort(key=lambda m: resolve_created_at(m) or datetime.fromtimestamp(0), reverse=True)
    return results
