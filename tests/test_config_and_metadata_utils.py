from __future__ import annotations

import sqlite3
from pathlib import Path

import voicememowhisper.config as config
from voicememowhisper.config import Settings
from voicememowhisper.db_introspection import clear_caches, find_record_table, tables_with_titles
from voicememowhisper.metadata import VoiceMemo
from voicememowhisper.metadata_cache import MetadataCache
from voicememowhisper.metadata_parser import normalize_value, pick, resolve_path, truthy
from voicememowhisper.metadata_parser import is_trashed, resolve_related_title, to_datetime


def test_detect_default_paths_cloud_and_legacy(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config.Path, "home", lambda: tmp_path)

    def _safe_exists(p: Path) -> bool:
        s = str(p)
        return s.endswith("/Recordings") or s.endswith("/CloudRecordings.db") or s.endswith("/Recents.sqlite")

    monkeypatch.setattr(config, "_safe_exists", _safe_exists)
    root, recordings, metadata, legacy = config._detect_default_paths()
    assert recordings.name == "Recordings"
    assert metadata.name in ("CloudRecordings.db", "Recents.sqlite")
    assert legacy is not None and legacy.name == "Recents.sqlite"


def test_detect_default_paths_recents_only(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config.Path, "home", lambda: tmp_path)

    def _safe_exists(p: Path) -> bool:
        s = str(p)
        return s.endswith("/Recordings") or s.endswith("/Recents.sqlite")

    monkeypatch.setattr(config, "_safe_exists", _safe_exists)
    root, recordings, metadata, legacy = config._detect_default_paths()
    assert recordings.name == "Recordings"
    assert metadata.name == "Recents.sqlite"
    assert legacy is None


def test_detect_default_paths_fallback_when_no_recordings(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(config, "_safe_exists", lambda _p: False)
    root, recordings, metadata, legacy = config._detect_default_paths()
    assert "Recordings" in str(recordings)
    assert metadata.name in ("CloudRecordings.db", "Recents.sqlite")


def test_metadata_cache_refresh_permission_error(monkeypatch, tmp_path: Path) -> None:
    settings = Settings(
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

    cache = MetadataCache(settings, loader=lambda _s: (_ for _ in ()).throw(PermissionError("nope")))
    cache.refresh()
    memo = cache.get_memo(tmp_path / "recordings" / "abc.m4a")
    assert memo.guid == "abc"


def test_metadata_cache_get_memo_updates_path_when_title_present(tmp_path: Path) -> None:
    settings = Settings(
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

    memo = VoiceMemo(guid="abc", path=tmp_path / "recordings" / "abc.m4a", title="example")
    cache = MetadataCache(settings, loader=lambda _s: {"abc": memo})
    cache.refresh()
    new_path = tmp_path / "other" / "abc.m4a"
    got = cache.get_memo(new_path)
    assert got.path == new_path


def test_metadata_cache_lazy_refresh_updates_path(tmp_path: Path) -> None:
    settings = Settings(
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
    # Loader returns a memo only after refresh; should update memo.path to requested path.
    memo = VoiceMemo(guid="abc", path=tmp_path / "recordings" / "abc.m4a", title="example")
    cache = MetadataCache(settings, loader=lambda _s: {"abc": memo})
    requested = tmp_path / "other" / "abc.m4a"
    got = cache.get_memo(requested)
    assert got.path == requested


def test_metadata_cache_display_name_prefers_title(tmp_path: Path) -> None:
    memo = VoiceMemo(guid="abc", path=tmp_path / "x.m4a", title="  example  ")
    assert MetadataCache.display_name(memo) == "example"


def test_truthy_and_normalize_helpers() -> None:
    assert truthy(None) is False
    assert truthy("false") is False
    assert truthy("0") is False
    assert truthy("yes") is True
    assert normalize_value(b"t\x00e\x00s\x00t\x00") == "test"
    assert to_datetime(None) is None


def test_pick_and_resolve_path(monkeypatch, tmp_path: Path) -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE t (a TEXT, b TEXT)")
    conn.execute("INSERT INTO t (a, b) VALUES (?, ?)", ("", " chosen "))
    row = conn.execute("SELECT * FROM t").fetchone()
    assert row is not None
    assert pick(row, ["a", "b"]) == "chosen"

    settings = Settings(
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

    conn.execute("CREATE TABLE p (ZPATH TEXT)")
    conn.execute("INSERT INTO p (ZPATH) VALUES (?)", ("Recordings/foo.m4a",))
    prow = conn.execute("SELECT * FROM p").fetchone()
    assert prow is not None
    resolved = resolve_path(prow, settings, guid="abc")
    assert str(resolved).endswith("Recordings/foo.m4a")
    conn.close()


def test_resolve_path_variants_and_trash(monkeypatch, tmp_path: Path) -> None:
    settings = Settings(
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
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE p (ZPATH TEXT, ZTRASHED INTEGER)")
    abs_path = tmp_path / "abs.m4a"
    conn.execute("INSERT INTO p (ZPATH, ZTRASHED) VALUES (?, ?)", (f"file://{abs_path}", 1))
    row = conn.execute("SELECT * FROM p").fetchone()
    assert row is not None
    assert resolve_path(row, settings, guid="abc") == abs_path
    assert is_trashed(row) is True
    conn.close()


def test_resolve_related_title(monkeypatch) -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE ZMETA (Z_PK INTEGER PRIMARY KEY, ZTITLE TEXT)")
    conn.execute("INSERT INTO ZMETA (Z_PK, ZTITLE) VALUES (1, 'example')")  # synthetic
    conn.execute("CREATE TABLE ZREC (Z_PK INTEGER PRIMARY KEY, ZMETADATA INTEGER)")
    conn.execute("INSERT INTO ZREC (Z_PK, ZMETADATA) VALUES (2, 1)")
    row = conn.execute("SELECT * FROM ZREC").fetchone()
    assert row is not None
    assert resolve_related_title(conn, row) == "example"
    conn.close()


def test_env_helpers(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VOICE_MEMO_CONTAINER", str(tmp_path))
    assert config._default_recordings_dir().name == "Recordings"
    _ = config._default_metadata_db()


def test_config_optional_env_path_and_legacy_db_branches(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("X_TEST_PATH", "~/example")
    p = config._optional_env_path("X_TEST_PATH", None)
    assert p is not None and p.name == "example"

    # Cover _detect_default_paths branch where only recordings exist.
    monkeypatch.setattr(config.Path, "home", lambda: tmp_path)

    def _safe_exists(pth: Path) -> bool:
        return str(pth).endswith("/Recordings")

    monkeypatch.setattr(config, "_safe_exists", _safe_exists)
    _root, _recordings, _metadata, _legacy = config._detect_default_paths()

    # Cover _default_legacy_metadata_db container branches.
    container = tmp_path / "container"
    monkeypatch.setenv("VOICE_MEMO_CONTAINER", str(container))

    def _safe_exists2(pth: Path) -> bool:
        s = str(pth)
        if s.endswith("CloudRecordings.db"):
            return True
        if s.endswith("Recents.sqlite"):
            return True
        return False

    monkeypatch.setattr(config, "_safe_exists", _safe_exists2)
    assert config._default_legacy_metadata_db() is not None

    def _safe_exists3(pth: Path) -> bool:
        s = str(pth)
        if s.endswith("CloudRecordings.db"):
            return False
        if s.endswith("Recents.sqlite"):
            return True
        return False

    monkeypatch.setattr(config, "_safe_exists", _safe_exists3)
    assert config._default_legacy_metadata_db() is None

    _ = config.load_settings()


def test_db_introspection_tables_with_titles_and_find_record_table() -> None:
    clear_caches()
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE ZVOICE (Z_PK INTEGER PRIMARY KEY, ZTITLE TEXT, ZDURATION REAL, ZGUID TEXT)")
    conn.execute("CREATE TABLE OTHER (X INTEGER)")

    tables = tables_with_titles(conn)
    assert "ZVOICE" in tables
    assert find_record_table(conn) == "ZVOICE"
    conn.close()


def test_db_introspection_finds_non_priority_table() -> None:
    clear_caches()
    conn = sqlite3.connect(":memory:")
    # Non-priority table name with required columns should be found.
    conn.execute(
        "CREATE TABLE ZSOMETHING (Z_PK INTEGER PRIMARY KEY, ZTITLE TEXT, ZGUID TEXT, ZCREATIONDATE REAL)"
    )
    assert find_record_table(conn) == "ZSOMETHING"
    conn.close()
