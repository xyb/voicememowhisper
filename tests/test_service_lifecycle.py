# pyright: reportMissingImports=false

from __future__ import annotations

import logging
from pathlib import Path

import pytest


def test_service_start_join_stop_processes_backlog(service_factory, settings) -> None:
    audio = settings.recordings_dir / "foo.m4a"
    audio.write_bytes(b"audio" + b"\0" * 1024)  # > _MIN_VALID_M4A_SIZE

    service = service_factory(settings)
    service.start(watch=False)
    try:
        service.join()
    finally:
        service.stop()

    # Transcript + archive should exist and state should be recorded.
    transcripts = list(settings.transcript_dir.glob("*_foo.txt"))
    assert len(transcripts) == 1
    content = transcripts[0].read_text().strip()
    assert content.startswith("TRANSCRIPT:")
    assert "foo" in content
    assert content.endswith(".m4a")

    archives = list(settings.archive_dir.glob("*_foo*.m4a"))  # type: ignore[union-attr]
    assert len(archives) == 1

    t, a = service.state.data.get("foo")  # type: ignore[attr-defined]
    assert t is not None
    assert a is not None


def test_log_sources_uses_overrides(monkeypatch, service_factory, settings, caplog) -> None:
    # Trigger both branches: recordings override + transcript override.
    monkeypatch.setenv("VOICE_MEMO_RECORDINGS_DIR", str(settings.recordings_dir))
    monkeypatch.setenv("VOICE_MEMO_TRANSCRIPT_DIR", str(settings.transcript_dir))

    service = service_factory(settings)
    with caplog.at_level(logging.INFO):
        service._log_sources()

    msgs = "\n".join(r.message for r in caplog.records)
    assert "VOICE_MEMO_RECORDINGS_DIR" in msgs
    assert "VOICE_MEMO_TRANSCRIPT_DIR" in msgs


def test_archive_memo_returns_none_on_copy_error(service, settings, monkeypatch) -> None:
    audio = settings.recordings_dir / "bad.m4a"
    audio.write_text("audio")
    memo = service._memo_for_path(audio)

    def boom(_src, _dst):
        raise OSError("nope")

    monkeypatch.setattr("voicememowhisper.service.shutil.copy2", boom)
    out = service._archive_memo(memo, "x.m4a")
    assert out is None


def test_process_inbox_file_skips_non_audio(service, settings) -> None:
    inbox = settings.inbox_dir
    assert inbox is not None
    f = inbox / "note.txt"
    f.write_text("x")
    assert service._process_inbox_file(f) is None


def test_process_memo_gives_up_on_empty_file(service, settings, monkeypatch) -> None:
    # Avoid sleeping in retry loop.
    monkeypatch.setattr("voicememowhisper.service.time.sleep", lambda _s: None)

    audio = settings.recordings_dir / "empty.m4a"
    audio.write_bytes(b"")  # size 0 => retry loop => give up
    memo = service._memo_for_path(audio)
    service._process_memo(memo)

    # No transcript should be written.
    assert list(settings.transcript_dir.glob("*_empty.txt")) == []

