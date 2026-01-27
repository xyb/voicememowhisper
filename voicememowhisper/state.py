from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Iterable, Set, Optional

from .state_schema import ensure_schema


class StateStore:
    """Persist processed voice memo GUIDs in a sqlite database."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._lock = threading.Lock()
        ensure_schema(self._conn)
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

    def has_archived_path(self, archived_path: Path) -> bool:
        """Return True if any processed row references this archived_path."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT 1 FROM processed WHERE archived_path = ? LIMIT 1;",
                (str(archived_path),),
            )
            return cursor.fetchone() is not None

    def set_duration_for_archived_path(self, archived_path: Path, duration: float) -> int:
        """
        Best-effort backfill for duration when we can probe it from the audio file.

        Returns the number of rows updated (0 or 1 in normal usage).
        """
        with self._lock:
            cursor = self._conn.execute(
                """
                UPDATE processed
                SET duration = ?, updated_at = CURRENT_TIMESTAMP
                WHERE archived_path = ? AND duration IS NULL
                """,
                (float(duration), str(archived_path)),
            )
            self._conn.commit()
            return int(cursor.rowcount or 0)

    def get_all_processed(self) -> list[dict]:
        """Retrieve all processed records with metadata."""
        with self._lock:
            self._conn.row_factory = sqlite3.Row
            cursor = self._conn.execute("SELECT * FROM processed")
            rows = [dict(row) for row in cursor.fetchall()]
            self._conn.row_factory = None # Reset row factory
            return rows

