from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Iterable, Set


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
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
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

    def mark_processed(self, guid: str, transcript_path: Path) -> None:
        with self._lock:
            self._conn.execute(
                """
            INSERT INTO processed (guid, transcript_path)
            VALUES (?, ?)
            ON CONFLICT(guid) DO UPDATE SET
                transcript_path = excluded.transcript_path,
                updated_at = CURRENT_TIMESTAMP;
            """,
                (guid, str(transcript_path)),
            )
            self._conn.commit()
