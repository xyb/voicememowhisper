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
                title TEXT,
                duration REAL,
                created_at TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        # Add columns if they don't exist (for backward compatibility)
        cursor = self._conn.execute("PRAGMA table_info(processed)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if "archived_path" not in columns:
            self._conn.execute("ALTER TABLE processed ADD COLUMN archived_path TEXT")
        if "title" not in columns:
            self._conn.execute("ALTER TABLE processed ADD COLUMN title TEXT")
        if "duration" not in columns:
            self._conn.execute("ALTER TABLE processed ADD COLUMN duration REAL")
        if "created_at" not in columns:
            self._conn.execute("ALTER TABLE processed ADD COLUMN created_at TEXT")
            
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

    def mark_processed(
        self, 
        guid: str, 
        transcript_path: Path, 
        archived_path: Optional[Path] = None,
        title: Optional[str] = None,
        duration: Optional[float] = None,
        created_at: Optional[str] = None
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
            INSERT INTO processed (guid, transcript_path, archived_path, title, duration, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(guid) DO UPDATE SET
                transcript_path = excluded.transcript_path,
                archived_path = excluded.archived_path,
                title = excluded.title,
                duration = excluded.duration,
                created_at = excluded.created_at,
                updated_at = CURRENT_TIMESTAMP;
            """,
                (
                    guid, 
                    str(transcript_path), 
                    str(archived_path) if archived_path else None,
                    title,
                    duration,
                    created_at
                ),
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

    def get_all_processed(self) -> list[dict]:
        """Retrieve all processed records with metadata."""
        with self._lock:
            self._conn.row_factory = sqlite3.Row
            cursor = self._conn.execute("SELECT * FROM processed")
            rows = [dict(row) for row in cursor.fetchall()]
            self._conn.row_factory = None # Reset row factory
            return rows

