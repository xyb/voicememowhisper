from __future__ import annotations

import sqlite3
from pathlib import Path

import voicememowhisper.metadata_parser as mp
from voicememowhisper.config import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        container_root=tmp_path,
        recordings_dir=tmp_path / "recordings",
        metadata_db=tmp_path / "metadata.db",
        legacy_metadata_db=None,
        transcript_dir=tmp_path / "transcripts",
        archive_dir=None,
        archive_enabled=False,
        inbox_dir=None,
        state_db=tmp_path / "state.sqlite",
        whisperkit_cli="whisperkit-cli",
        whisperkit_model="dummy-model",
        whisperkit_extra_args=(),
        language=None,
        processing_order="newest-first",
    )


def test_truthy_non_scalar_types() -> None:
    assert mp.truthy([]) is False
    assert mp.truthy([1]) is True


def test_normalize_value_memoryview_and_decode_fallback() -> None:
    assert mp.normalize_value(memoryview(b"abc")) == "abc"
    # This will fail utf-8 and utf-16 decodes, then fall back to errors=ignore.
    assert mp.normalize_value(b"\xff").strip() == ""


def test_pick_skips_normalized_empty(tmp_path: Path) -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE t (a BLOB)")
    conn.execute("INSERT INTO t (a) VALUES (?)", (b"\x00",))
    row = conn.execute("SELECT * FROM t").fetchone()
    assert row is not None
    assert mp.pick(row, ["a"]) is None
    conn.close()


def test_resolve_path_tilde_and_relative(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE p (ZPATH TEXT)")
    conn.execute("INSERT INTO p (ZPATH) VALUES (?)", ("~/example.m4a",))
    row = conn.execute("SELECT * FROM p").fetchone()
    assert row is not None
    resolved = mp.resolve_path(row, settings, guid="abc")
    assert str(resolved).endswith("example.m4a")

    conn.execute("DELETE FROM p")
    conn.execute("INSERT INTO p (ZPATH) VALUES (?)", ("foo/bar.m4a",))
    row2 = conn.execute("SELECT * FROM p").fetchone()
    assert row2 is not None
    resolved2 = mp.resolve_path(row2, settings, guid="abc")
    assert resolved2 == settings.recordings_dir / "foo" / "bar.m4a"
    conn.close()


def test_resolve_related_title_branches(monkeypatch) -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE R (Z_PK INTEGER PRIMARY KEY, ZMETADATA TEXT)")
    conn.execute("INSERT INTO R (Z_PK, ZMETADATA) VALUES (1, 'not-int')")
    row = conn.execute("SELECT * FROM R").fetchone()
    assert row is not None

    # No tables with titles -> early return
    monkeypatch.setattr(mp, "tables_with_titles", lambda _c: [])
    assert mp.resolve_related_title(conn, row) is None

    # Bad ref value -> int conversion fails
    monkeypatch.setattr(mp, "tables_with_titles", lambda _c: ["ZMETA"])
    assert mp.resolve_related_title(conn, row) is None

    # Force sqlite error during lookup
    conn.execute("UPDATE R SET ZMETADATA = 1 WHERE Z_PK = 1")
    row2 = conn.execute("SELECT * FROM R").fetchone()
    assert row2 is not None
    monkeypatch.setattr(mp, "tables_with_titles", lambda _c: ["NO_SUCH_TABLE"])
    assert mp.resolve_related_title(conn, row2) is None

    conn.close()
