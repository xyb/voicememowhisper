from __future__ import annotations

import sqlite3
from typing import Dict, List

from .metadata_constants import DATE_COLUMNS, DURATION_COLUMNS, GUID_COLUMNS, TITLE_COLUMNS

_TABLE_COLUMN_CACHE: Dict[tuple[int, str], set[str]] = {}
_TABLES_WITH_TITLES_CACHE: Dict[int, List[str]] = {}


def clear_caches() -> None:
    _TABLE_COLUMN_CACHE.clear()
    _TABLES_WITH_TITLES_CACHE.clear()


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    key = (id(conn), table)
    cached = _TABLE_COLUMN_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        rows = conn.execute(f"PRAGMA table_info('{table}')")
    except sqlite3.Error:  # pragma: no cover - corrupt db
        columns = set()
    else:
        columns = {row[1] for row in rows}
    _TABLE_COLUMN_CACHE[key] = columns
    return columns


def tables_with_titles(conn: sqlite3.Connection) -> List[str]:
    cache = _TABLES_WITH_TITLES_CACHE.get(id(conn))
    if cache is not None:
        return cache
    tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    result: List[str] = []
    for name in tables:
        cols = table_columns(conn, name)
        if "Z_PK" in cols and any(col in cols for col in TITLE_COLUMNS):
            result.append(name)
    _TABLES_WITH_TITLES_CACHE[id(conn)] = result
    return result


def find_record_table(conn: sqlite3.Connection) -> str | None:
    priority = [
        "ZCLOUDRECORDING",
        "ZVOICE",
        "ZRECORDING",
        "ZCLOUDRECORDINGS",
    ]
    tables = set(tables_with_titles(conn)) | {
        row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    for name in priority:
        if name in tables:
            return name
    for name in tables:
        cols = table_columns(conn, name)
        if not cols:
            continue
        if any(col in cols for col in GUID_COLUMNS) and any(col in cols for col in TITLE_COLUMNS):
            if any(col in cols for col in DATE_COLUMNS) or any(col in cols for col in DURATION_COLUMNS):
                return name
    return None

