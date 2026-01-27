from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import voicememowhisper.metadata as md


def _write_metadata_db(path: Path) -> None:
    """
    Create a minimal sqlite schema that md.load_voice_memos() can discover.
    """
    conn = sqlite3.connect(str(path))
    try:
        # Primary table with GUID/title/date/duration.
        conn.execute(
            """
            CREATE TABLE ZVOICE (
                Z_PK INTEGER PRIMARY KEY,
                ZUNIQUEID TEXT,
                ZDISPLAYTITLE TEXT,
                ZCREATIONDATE REAL,
                ZDURATION REAL,
                ZTRASHED INTEGER
            );
            """
        )
        # A "titles" table referenced by ZTITLEMETADATA.
        conn.execute(
            """
            CREATE TABLE ZTITLEMETA (
                Z_PK INTEGER PRIMARY KEY,
                ZDISPLAYTITLE TEXT
            );
            """
        )

        # Record with direct title.
        conn.execute(
            "INSERT INTO ZVOICE (Z_PK, ZUNIQUEID, ZDISPLAYTITLE, ZCREATIONDATE, ZDURATION, ZTRASHED) VALUES (?, ?, ?, ?, ?, ?)",
            (1, "guid-direct", "DirectTitle", 0.0, 12.5, 0),
        )

        # Record with missing title that should resolve via related table reference.
        conn.execute("INSERT INTO ZTITLEMETA (Z_PK, ZDISPLAYTITLE) VALUES (?, ?)", (10, "RelatedTitle"))
        conn.execute(
            "INSERT INTO ZVOICE (Z_PK, ZUNIQUEID, ZDISPLAYTITLE, ZCREATIONDATE, ZDURATION, ZTRASHED) VALUES (?, ?, ?, ?, ?, ?)",
            (2, "guid-related", None, 60.0, 3.0, 0),
        )
        # Add the reference column after insert to keep schema simple.
        conn.execute("ALTER TABLE ZVOICE ADD COLUMN ZTITLEMETADATA INTEGER")
        conn.execute("UPDATE ZVOICE SET ZTITLEMETADATA = ? WHERE Z_PK = ?", (10, 2))

        # Trashed record should be present but marked trashed.
        conn.execute(
            "INSERT INTO ZVOICE (Z_PK, ZUNIQUEID, ZDISPLAYTITLE, ZCREATIONDATE, ZDURATION, ZTRASHED) VALUES (?, ?, ?, ?, ?, ?)",
            (3, "guid-trash", "Trash", 120.0, 1.0, 1),
        )
        conn.commit()
    finally:
        conn.close()


def test_load_voice_memos_reads_rows_and_resolves_related_title(settings_factory, tmp_path) -> None:
    recordings = tmp_path / "recordings"
    recordings.mkdir(exist_ok=True)
    db = tmp_path / "CloudRecordings.db"
    _write_metadata_db(db)

    settings = settings_factory(
        recordings_dir=recordings,
        metadata_db=db,
        legacy_metadata_db=None,
        archive_enabled=False,
        inbox_dir=None,
    )

    memos = md.load_voice_memos(settings)
    assert "guid-direct" in memos
    assert memos["guid-direct"].title == "DirectTitle"
    assert memos["guid-direct"].duration_seconds == 12.5
    assert memos["guid-direct"].is_trashed is False

    assert "guid-related" in memos
    assert memos["guid-related"].title == "RelatedTitle"

    assert "guid-trash" in memos
    assert memos["guid-trash"].is_trashed is True


def test_load_voice_memos_uses_legacy_db_when_primary_missing(settings_factory, tmp_path) -> None:
    recordings = tmp_path / "recordings"
    recordings.mkdir(exist_ok=True)
    primary = tmp_path / "missing.db"
    legacy = tmp_path / "legacy.db"
    _write_metadata_db(legacy)

    settings = settings_factory(
        recordings_dir=recordings,
        metadata_db=primary,
        legacy_metadata_db=legacy,
        archive_enabled=False,
        inbox_dir=None,
    )

    memos = md.load_voice_memos(settings)
    assert "guid-direct" in memos


def test_to_datetime_handles_mac_epoch() -> None:
    # 0 seconds since 2001-01-01 UTC.
    dt = md._to_datetime(0)  # type: ignore[attr-defined]
    assert dt is not None
    assert dt.tzinfo is timezone.utc
    assert dt == md.MAC_EPOCH


def test_resolve_path_handles_relative_recordings_prefix(settings_factory, tmp_path) -> None:
    # Exercise _resolve_path "Recordings/..." special-case.
    settings = settings_factory(container_root=tmp_path, recordings_dir=tmp_path / "rec")
    row = {"ZRELATIVEPATH": "Recordings/foo.m4a"}
    # emulate sqlite Row by providing keys() and __getitem__
    class _Row(dict):
        def keys(self):
            return super().keys()

    p = md._resolve_path(_Row(row), settings, "guid")  # type: ignore[attr-defined]
    assert str(p).endswith("Recordings/foo.m4a")

