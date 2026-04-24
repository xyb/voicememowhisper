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


def test_find_by_archived_basename_single_match(tmp_path: Path) -> None:
    db = tmp_path / "state.sqlite"
    store = StateStore(db)
    try:
        old_archive = Path("/old/VoiceMemoArchives/2024-08-15_10-42-46_sample.m4a")
        store.mark_processed("g1", tmp_path / "t.txt", old_archive)
        matches = store.find_by_archived_basename("2024-08-15_10-42-46_sample.m4a")
        assert matches == [("g1", old_archive)]
        assert store.find_by_archived_basename("nope.m4a") == []
    finally:
        store.close()


def test_find_by_archived_basename_multiple_matches(tmp_path: Path) -> None:
    """When two rows happen to share a basename, both are returned — caller
    decides what to do (the service layer refuses to self-heal ambiguous cases)."""
    db = tmp_path / "state.sqlite"
    store = StateStore(db)
    try:
        p1 = Path("/a/dup.m4a")
        p2 = Path("/b/dup.m4a")
        store.mark_processed("g1", tmp_path / "t1.txt", p1)
        store.mark_processed("g2", tmp_path / "t2.txt", p2)
        matches = store.find_by_archived_basename("dup.m4a")
        assert sorted(matches) == sorted([("g1", p1), ("g2", p2)])
    finally:
        store.close()


def test_find_by_archived_basename_skips_null_archive(tmp_path: Path) -> None:
    """Rows with NULL archived_path (transcribed-only) must not appear."""
    db = tmp_path / "state.sqlite"
    store = StateStore(db)
    try:
        store.mark_processed("g1", tmp_path / "t.txt", archived_path=None)
        assert store.find_by_archived_basename("whatever.m4a") == []
    finally:
        store.close()


def test_update_archived_path_rewrites_row(tmp_path: Path) -> None:
    db = tmp_path / "state.sqlite"
    store = StateStore(db)
    try:
        old = Path("/old/VoiceMemoArchives/x.m4a")
        new = Path("/new/VoiceMemoWhisper/Audio/x.m4a")
        store.mark_processed("g1", tmp_path / "t.txt", old)
        assert store.update_archived_path("g1", new) == 1

        _, archived = store.get_state("g1")
        assert archived == new
        # The old path must no longer be findable.
        assert store.has_archived_path(old) is False
        assert store.has_archived_path(new) is True
    finally:
        store.close()


def test_update_archived_path_missing_guid_is_noop(tmp_path: Path) -> None:
    db = tmp_path / "state.sqlite"
    store = StateStore(db)
    try:
        assert store.update_archived_path("ghost", tmp_path / "x.m4a") == 0
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

