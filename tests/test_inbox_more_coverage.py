from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import voicememowhisper.inbox as inbox_mod
from voicememowhisper.archive import ArchiveManager
from voicememowhisper.inbox import InboxProcessor, date_from_filename


def test_date_from_filename_returns_none_when_no_prefix(tmp_path: Path) -> None:
    p = tmp_path / "no-date.m4a"
    p.write_text("x")
    assert date_from_filename(p) is None


def test_date_from_filename_returns_none_on_invalid_date(tmp_path: Path) -> None:
    p = tmp_path / "2026-99-99_bad.m4a"
    p.write_text("x")
    assert date_from_filename(p) is None


def test_date_from_filename_tolerates_stat_failure(tmp_path: Path, monkeypatch) -> None:
    p = tmp_path / "2026-01-02_ok.m4a"
    p.write_text("x")

    # Force stat() to fail so we hit the OSError branch.
    monkeypatch.setattr(inbox_mod.Path, "stat", lambda *_a, **_k: (_ for _ in ()).throw(OSError("nope")))
    dt = date_from_filename(p)
    assert dt is not None
    assert dt.strftime("%Y-%m-%d") == "2026-01-02"


class _FakeDir:
    def __init__(self, *, exists: bool = True, glob_error: Exception | None = None) -> None:
        self._exists = exists
        self._glob_error = glob_error

    def exists(self) -> bool:
        return self._exists

    def glob(self, _pattern: str):
        if self._glob_error:
            raise self._glob_error
        return []


def test_scan_handles_permission_error(settings_factory) -> None:
    settings = settings_factory()
    # Use a fake inbox_dir that raises PermissionError when globbing.
    fake_settings = replace(settings, inbox_dir=_FakeDir(glob_error=PermissionError("denied")))  # type: ignore[arg-type]
    archive = ArchiveManager(settings)
    processor = InboxProcessor(fake_settings, archive, enqueue_callback=lambda _p: None)  # type: ignore[arg-type]
    processor.scan()  # should not raise


def test_scan_handles_generic_error(settings_factory) -> None:
    settings = settings_factory()
    fake_settings = replace(settings, inbox_dir=_FakeDir(glob_error=RuntimeError("boom")))  # type: ignore[arg-type]
    archive = ArchiveManager(settings)
    processor = InboxProcessor(fake_settings, archive, enqueue_callback=lambda _p: None)  # type: ignore[arg-type]
    processor.scan()  # should not raise


def test_process_file_duplicate_unlink_failure_is_nonfatal(settings_factory, monkeypatch) -> None:
    settings = settings_factory()
    archive = ArchiveManager(settings)
    processor = InboxProcessor(settings, archive, enqueue_callback=lambda _p: None)

    inbox_file = settings.inbox_dir / "2026-01-02 example.m4a"  # type: ignore[operator]
    inbox_file.write_text("same")

    # Force the target archive file to "exist" and hashes to match.
    target = settings.archive_dir / "undated_2026-01-02 example.m4a"  # type: ignore[operator]
    target.write_text("same")

    monkeypatch.setattr(processor, "date_from_filename_func", lambda _p: None)
    monkeypatch.setattr(inbox_mod, "hash_file", lambda _p: "h")
    monkeypatch.setattr(inbox_mod.Path, "unlink", lambda *_a, **_k: (_ for _ in ()).throw(OSError("nope")))

    dest = processor.process_file(inbox_file)
    assert dest is not None


def test_process_file_move_error_returns_none(settings_factory, monkeypatch) -> None:
    settings = settings_factory()
    archive = ArchiveManager(settings)
    processor = InboxProcessor(settings, archive, enqueue_callback=lambda _p: None)

    inbox_file = settings.inbox_dir / "2026-01-02 example.m4a"  # type: ignore[operator]
    inbox_file.write_text("x")

    monkeypatch.setattr(processor, "date_from_filename_func", lambda _p: None)
    monkeypatch.setattr(inbox_mod, "resolve_conflict_path", lambda p: p)
    monkeypatch.setattr(inbox_mod.shutil, "move", lambda *_a, **_k: (_ for _ in ()).throw(OSError("fail")))
    assert processor.process_file(inbox_file) is None
