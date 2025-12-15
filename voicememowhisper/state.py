from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Iterable, Set, Optional


class StateStore:
    """Persist processed voice memo GUIDs in a sqlite database."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS processed (
                guid TEXT PRIMARY KEY,
                transcript_path TEXT NOT NULL,
                archived_path TEXT, -- New column for archived file path
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        # Add archived_path column if it doesn't exist (for backward compatibility with older SQLite)
        # SQLite < 3.35.0 does not support ADD COLUMN IF NOT EXISTS
        cursor = self._conn.execute("PRAGMA table_info(processed)")
        columns = [row[1] for row in cursor.fetchall()]
        if "archived_path" not in columns:
            self._conn.execute("ALTER TABLE processed ADD COLUMN archived_path TEXT")
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def is_processed(self, guid: str) -> bool:
        with self._lock:
            cursor = self._conn.execute("SELECT 1 FROM processed WHERE guid = ? LIMIT 1;", (guid,))
            return cursor.fetchone() is not None

    def known_guids(self) -> Set[str]:
        with self._lock:
            cursor = self._conn.execute("SELECT guid FROM processed;")
            return {row[0] for row in cursor.fetchall()}

    def mark_processed(self, guid: str, transcript_path: Path, archived_path: Optional[Path] = None) -> None:
        with self._lock:
            self._conn.execute(
                """
            INSERT INTO processed (guid, transcript_path, archived_path)
            VALUES (?, ?, ?)
            ON CONFLICT(guid) DO UPDATE SET
                transcript_path = excluded.transcript_path,
                archived_path = excluded.archived_path,
                updated_at = CURRENT_TIMESTAMP;
            """,
                (guid, str(transcript_path), str(archived_path) if archived_path else None),
            )
            self._conn.commit()

    def get_state(self, guid: str) -> tuple[Optional[Path], Optional[Path]]:
        """Retrieve transcript_path and archived_path for a given GUID."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT transcript_path, archived_path FROM processed WHERE guid = ? LIMIT 1;", (guid,)
            )
            row = cursor.fetchone()
            if row:
                transcript_path = Path(row[0]) if row[0] else None
                archived_path = Path(row[1]) if row[1] else None
                return transcript_path, archived_path
            return None, None

