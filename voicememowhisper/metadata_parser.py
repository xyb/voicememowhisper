from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

from .config import Settings
from .db_introspection import tables_with_titles
from .metadata_constants import MAC_EPOCH, PATH_COLUMNS, REFERENCE_COLUMNS, TITLE_COLUMNS, TRASH_COLUMNS

LOGGER = logging.getLogger("metadata_parser")


def to_datetime(value: float | int | None) -> datetime | None:
    if value is None:
        return None
    try:
        return MAC_EPOCH + timedelta(seconds=float(value))
    except Exception:  # pragma: no cover
        LOGGER.debug("Failed to convert %s to datetime", value, exc_info=True)
        return None


def truthy(value) -> bool:
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "no")
    return bool(value)


def normalize_value(value):
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytes):
        for encoding in ("utf-8", "utf-16-le", "utf-16-be"):
            try:
                decoded = value.decode(encoding)
                return decoded.replace("\x00", "").strip()
            except UnicodeDecodeError:
                continue
        return value.decode("utf-8", errors="ignore").replace("\x00", "").strip()
    if isinstance(value, str):
        return value.strip()
    return value


def pick(row: sqlite3.Row, candidates: Iterable[str]):
    keys = row.keys()
    for name in candidates:
        if name in keys:
            value = row[name]
            if value not in (None, ""):
                normalized = normalize_value(value)
                if normalized in (None, ""):
                    continue
                return normalized
    return None


def resolve_path(row: sqlite3.Row, settings: Settings, guid: str) -> Path:
    keys = row.keys()
    for name in PATH_COLUMNS:
        if name in keys:
            value = normalize_value(row[name])
            if isinstance(value, str) and value.strip():
                candidate = value.strip()
                if candidate.startswith("file://"):
                    candidate = candidate[7:]
                if candidate.startswith("~/"):
                    return Path(candidate).expanduser()
                path = Path(candidate)
                if path.is_absolute():
                    return path
                parts = path.parts
                if parts and parts[0].lower() == "recordings":
                    return settings.container_root / Path(*parts)
                return settings.recordings_dir / path
    return settings.recordings_dir / f"{guid}.m4a"


def is_trashed(row: sqlite3.Row) -> bool:
    return any(truthy(row[name]) for name in TRASH_COLUMNS if name in row.keys())


def resolve_related_title(conn: sqlite3.Connection, row: sqlite3.Row) -> str | None:
    keys = row.keys()
    tables = tables_with_titles(conn)
    if not tables:
        return None

    for ref in REFERENCE_COLUMNS:
        if ref not in keys:
            continue
        ref_value = row[ref]
        if ref_value in (None, 0):
            continue
        try:
            ref_id = int(ref_value)
        except (TypeError, ValueError):
            continue
        for table in tables:
            try:
                candidate = conn.execute(
                    f"SELECT * FROM {table} WHERE Z_PK = ? LIMIT 1", (ref_id,)
                ).fetchone()
            except sqlite3.Error:
                continue
            if not candidate:
                continue
            title = pick(candidate, TITLE_COLUMNS)
            if title:
                return title
    return None

