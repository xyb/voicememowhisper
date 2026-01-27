from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import voicememowhisper.service as svc


def test_hash_file_returns_none_on_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "nope.m4a"
    assert svc._hash_file(missing) is None  # type: ignore[attr-defined]


def test_date_from_filename_parses_prefix_and_preserves_time(monkeypatch, tmp_path: Path) -> None:
    p = tmp_path / "2026-01-27 foo.m4a"
    p.write_text("a")
    # Force mtime to provide time component 03:04:05
    fixed = datetime(2026, 1, 1, 3, 4, 5)
    monkeypatch.setattr(Path, "stat", lambda self: type("S", (), {"st_mtime": fixed.timestamp()})())

    dt = svc._date_from_filename(p)  # type: ignore[attr-defined]
    assert dt is not None
    assert dt.year == 2026 and dt.month == 1 and dt.day == 27
    assert (dt.hour, dt.minute, dt.second) == (3, 4, 5)


def test_scan_inbox_skips_when_dir_missing(service, settings) -> None:
    # ensure settings.inbox_dir points to a missing directory
    missing = settings.container_root / "missing-inbox"
    service.settings = replace(service.settings, inbox_dir=missing)
    service._scan_inbox()
    assert service._queue.empty()

