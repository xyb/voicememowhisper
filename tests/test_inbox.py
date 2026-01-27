from __future__ import annotations

# pyright: reportMissingImports=false

from datetime import datetime

import pytest

import voicememowhisper.service as svc


def test_process_inbox_uses_filename_date_and_syncs_mtime(service_factory, settings, monkeypatch) -> None:
    fixed_dt = datetime(2024, 1, 2, 3, 4, 5)
    monkeypatch.setattr(svc, "_date_from_filename", lambda _p: fixed_dt)
    service = service_factory(settings)

    inbox_file = settings.inbox_dir / "2024-01-02 foo.m4a"  # type: ignore[operator]
    inbox_file.write_text("hello")

    dest = service._process_inbox_file(inbox_file)

    assert dest is not None
    assert not inbox_file.exists()
    assert dest.exists()
    assert dest.name.startswith("2024-01-02_03-04-05_2024-01-02 foo")
    assert dest.stat().st_mtime == pytest.approx(fixed_dt.timestamp(), abs=2)


def test_process_inbox_discards_duplicate_by_hash(service_factory, settings, monkeypatch) -> None:
    fixed_dt = datetime(2024, 1, 2, 0, 0, 0)
    monkeypatch.setattr(svc, "_date_from_filename", lambda _p: fixed_dt)
    service = service_factory(settings)

    archive_path = settings.archive_dir / "2024-01-02_00-00-00_2024-01-02 foo.m4a"  # type: ignore[operator]
    archive_path.write_text("same-content")

    inbox_file = settings.inbox_dir / "2024-01-02 foo.m4a"  # type: ignore[operator]
    inbox_file.write_text("same-content")

    dest = service._process_inbox_file(inbox_file)

    assert dest == archive_path
    assert not inbox_file.exists()
    assert archive_path.read_text() == "same-content"


def test_process_inbox_conflict_different_hash_gets_suffix(service_factory, settings, monkeypatch) -> None:
    fixed_dt = datetime(2024, 1, 2, 0, 0, 0)
    monkeypatch.setattr(svc, "_date_from_filename", lambda _p: fixed_dt)
    service = service_factory(settings)

    archive_path = settings.archive_dir / "2024-01-02_00-00-00_2024-01-02 foo.m4a"  # type: ignore[operator]
    archive_path.write_text("old")

    inbox_file = settings.inbox_dir / "2024-01-02 foo.m4a"  # type: ignore[operator]
    inbox_file.write_text("new")

    dest = service._process_inbox_file(inbox_file)

    assert dest is not None
    assert dest != archive_path
    assert dest.name.endswith("_1.m4a")
    assert not inbox_file.exists()
    assert archive_path.read_text() == "old"
    assert dest.read_text() == "new"


def test_scan_archive_backfills_untranscribed(service, settings) -> None:
    archive_file = settings.archive_dir / "foo.m4a"  # type: ignore[operator]
    archive_file.write_text("audio")

    service._scan_archive_for_untranscribed()

    queued = service._queue.get(timeout=1)
    assert queued == archive_file
    assert "foo" in service._inflight
