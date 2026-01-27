from __future__ import annotations

import sqlite3
from pathlib import Path

from voicememowhisper.state import StateStore


def test_state_store_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "state.sqlite"
    store = StateStore(db)
    try:
        transcript = tmp_path / "t.txt"
        archived = tmp_path / "a.m4a"
        store.mark_processed(
            guid="g1",
            transcript_path=transcript,
            archived_path=archived,
            title="Title",
            duration=1.2,
            created_at="2026-01-01T00:00:00",
        )

        assert store.is_processed("g1") is True
        assert store.is_processed("missing") is False
        assert store.known_guids() == {"g1"}

        t2, a2 = store.get_state("g1")
        assert t2 == transcript
        assert a2 == archived

        assert store.has_archived_path(archived) is True
        assert store.has_archived_path(tmp_path / "nope.m4a") is False
    finally:
        store.close()


def test_state_store_updates_existing_row(tmp_path: Path) -> None:
    db = tmp_path / "state.sqlite"
    store = StateStore(db)
    try:
        transcript1 = tmp_path / "t1.txt"
        transcript2 = tmp_path / "t2.txt"
        archived1 = tmp_path / "a1.m4a"
        archived2 = tmp_path / "a2.m4a"

        store.mark_processed("g1", transcript1, archived1)
        store.mark_processed("g1", transcript2, archived2)

        t, a = store.get_state("g1")
        assert t == transcript2
        assert a == archived2
    finally:
        store.close()


def test_state_store_migrates_missing_columns(tmp_path: Path) -> None:
    """
    Create an "old" schema (no archived_path/title/duration/created_at), then ensure
    StateStore adds columns on init.
    """
    db = tmp_path / "state.sqlite"
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("CREATE TABLE processed (guid TEXT PRIMARY KEY, transcript_path TEXT NOT NULL);")
        conn.commit()
    finally:
        conn.close()

    store = StateStore(db)
    try:
        cols = {row[1] for row in store._conn.execute("PRAGMA table_info(processed)").fetchall()}  # type: ignore[attr-defined]
        assert "archived_path" in cols
        assert "title" in cols
        assert "duration" in cols
        assert "created_at" in cols
    finally:
        store.close()

