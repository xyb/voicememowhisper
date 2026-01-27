from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import voicememowhisper.inbox as inbox_mod
from voicememowhisper.archive import ArchiveManager
from voicememowhisper.inbox import InboxProcessor


def test_start_watcher_returns_none_when_inbox_missing(settings_factory) -> None:
    settings = settings_factory(inbox_dir=Path("/does/not/exist"))
    archive = ArchiveManager(settings)
    processor = InboxProcessor(settings, archive, enqueue_callback=lambda _p: None)
    assert processor.start_watcher() is None


def test_start_watcher_calls_watcher_when_inbox_exists(settings_factory, monkeypatch) -> None:
    settings = settings_factory()
    archive = ArchiveManager(settings)
    called: dict[str, object] = {}

    def _fake_start_watcher(path: Path, callback, *, extensions):
        called["path"] = path
        called["extensions"] = extensions
        return object()

    monkeypatch.setattr(inbox_mod, "start_watcher", _fake_start_watcher)
    processor = InboxProcessor(settings, archive, enqueue_callback=lambda _p: None)
    assert processor.start_watcher() is not None
    assert called["path"] == settings.inbox_dir
    assert ".m4a" in called["extensions"]


def test_scan_noops_when_inbox_missing(settings_factory) -> None:
    settings = settings_factory(inbox_dir=Path("/does/not/exist"))
    archive = ArchiveManager(settings)
    processor = InboxProcessor(settings, archive, enqueue_callback=lambda _p: None)
    processor.scan()  # should not raise


def test_handle_new_file_enqueues_when_processed(settings_factory, monkeypatch) -> None:
    settings = settings_factory()
    archive = ArchiveManager(settings)

    moved = settings.archive_dir / "2026-01-01_00-00-00_example.m4a"  # type: ignore[operator]
    enqueued: list[Path] = []

    processor = InboxProcessor(settings, archive, enqueue_callback=lambda p: enqueued.append(p))
    monkeypatch.setattr(processor, "process_file", lambda _p: moved)
    processor.handle_new_file(settings.inbox_dir / "example.m4a")  # type: ignore[operator]
    assert enqueued == [moved]


def test_process_file_skips_non_audio(settings_factory) -> None:
    settings = settings_factory()
    archive = ArchiveManager(settings)
    processor = InboxProcessor(settings, archive, enqueue_callback=lambda _p: None)

    p = settings.inbox_dir / "not-audio.txt"  # type: ignore[operator]
    p.write_text("x")
    assert processor.process_file(p) is None
    assert p.exists()


def test_process_file_returns_none_without_archive_dir(settings_factory) -> None:
    settings = settings_factory()
    settings = replace(settings, archive_dir=None)
    archive = ArchiveManager(settings)
    processor = InboxProcessor(settings, archive, enqueue_callback=lambda _p: None)

    p = settings.inbox_dir / "example.m4a"  # type: ignore[operator]
    p.write_text("x")
    assert processor.process_file(p) is None


def test_process_file_uses_ctime_when_no_date_prefix(settings_factory, monkeypatch) -> None:
    settings = settings_factory()
    archive = ArchiveManager(settings)

    processor = InboxProcessor(settings, archive, enqueue_callback=lambda _p: None)
    monkeypatch.setattr(processor, "date_from_filename_func", lambda _p: None)

    p = settings.inbox_dir / "example.m4a"  # type: ignore[operator]
    p.write_text("x")

    # Ensure deterministic timestamp.
    fixed = datetime(2026, 1, 2, 3, 4, 5)
    monkeypatch.setattr(inbox_mod, "datetime", SimpleNamespace(fromtimestamp=lambda _t: fixed))

    dest = processor.process_file(p)
    assert dest is not None
    assert dest.name.startswith("2026-01-02_03-04-05_")
