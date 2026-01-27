from __future__ import annotations

import sqlite3


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS processed (
            guid TEXT PRIMARY KEY,
            transcript_path TEXT NOT NULL,
            archived_path TEXT,
            title TEXT,
            duration REAL,
            created_at TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    columns = existing_columns(conn, "processed")
    if "archived_path" not in columns:
        conn.execute("ALTER TABLE processed ADD COLUMN archived_path TEXT")
    if "title" not in columns:
        conn.execute("ALTER TABLE processed ADD COLUMN title TEXT")
    if "duration" not in columns:
        conn.execute("ALTER TABLE processed ADD COLUMN duration REAL")
    if "created_at" not in columns:
        conn.execute("ALTER TABLE processed ADD COLUMN created_at TEXT")


def existing_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cursor = conn.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cursor.fetchall()}

