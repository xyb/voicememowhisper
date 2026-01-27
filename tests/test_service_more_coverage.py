from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import voicememowhisper.service as svc
from voicememowhisper.config import Settings


def _base_settings(tmp_path: Path) -> Settings:
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    archive = tmp_path / "archive"
    archive.mkdir()
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    return Settings(
        container_root=tmp_path,
        recordings_dir=recordings,
        metadata_db=tmp_path / "metadata.db",
        legacy_metadata_db=None,
        transcript_dir=transcripts,
        archive_dir=archive,
        archive_enabled=False,
        inbox_dir=inbox,
        state_db=tmp_path / "state.sqlite",
        whisperkit_cli="echo",
        whisperkit_model="dummy-model",
        whisperkit_extra_args=(),
        language=None,
        processing_order="newest-first",
    )


def test_service_init_raises_when_recordings_missing(tmp_path: Path) -> None:
    settings = _base_settings(tmp_path)
    missing = tmp_path / "missing-recordings"
    settings = replace(settings, recordings_dir=missing)
    with pytest.raises(FileNotFoundError):
        svc.VoiceMemoService(settings)


def test_service_init_auto_enables_archive_when_inbox_exists(tmp_path: Path, monkeypatch) -> None:
    settings = _base_settings(tmp_path)
    settings = replace(settings, archive_enabled=False)
    monkeypatch.setattr(svc, "WhisperTranscriber", lambda _s: object())
    monkeypatch.setattr(svc, "StateStore", lambda _p: object())
    monkeypatch.setattr(svc, "load_voice_memos", lambda _settings: {})
    service = svc.VoiceMemoService(settings)
    assert service.settings.archive_enabled is True


def test_service_start_watch_handles_inbox_watcher_error(tmp_path: Path, monkeypatch) -> None:
    settings = _base_settings(tmp_path)
    settings = replace(settings, archive_enabled=True)

    # Keep start() lightweight.
    monkeypatch.setattr(svc, "WhisperTranscriber", lambda _s: object())
    class _State:
        def __init__(self, _p: Path) -> None:
            return

        def get_state(self, _guid: str):
            return (None, None)

        def close(self):
            return

    monkeypatch.setattr(svc, "StateStore", _State)
    monkeypatch.setattr(svc, "load_voice_memos", lambda _settings: {})

    service = svc.VoiceMemoService(settings)
    monkeypatch.setattr(service, "enqueue_existing", lambda: None)

    class _DummyThread:
        def __init__(self, *a, **k):
            return

        def start(self):
            return

        def join(self):
            return

    monkeypatch.setattr(svc.threading, "Thread", _DummyThread)

    class _DummyObs:
        def stop(self):
            return

        def join(self):
            return

    monkeypatch.setattr(svc, "start_watcher", lambda *_a, **_k: _DummyObs())
    monkeypatch.setattr(service.inbox, "start_watcher", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    service.start(watch=True)
    service.stop()


def test_enqueue_existing_handles_permission_error(tmp_path: Path, monkeypatch) -> None:
    settings = _base_settings(tmp_path)
    monkeypatch.setattr(svc, "WhisperTranscriber", lambda _s: object())
    monkeypatch.setattr(svc, "StateStore", lambda _p: object())
    monkeypatch.setattr(svc, "load_voice_memos", lambda _settings: {})
    service = svc.VoiceMemoService(settings)

    original_glob = svc.Path.glob

    def _glob(self: Path, pattern: str):
        if self == settings.recordings_dir and pattern == "*.m4a":
            raise PermissionError("denied")
        return original_glob(self, pattern)

    monkeypatch.setattr(svc.Path, "glob", _glob)
    service.enqueue_existing()  # should not raise


def test_scan_archive_attribute_error_branch(tmp_path: Path, monkeypatch) -> None:
    settings = _base_settings(tmp_path)
    monkeypatch.setattr(svc, "WhisperTranscriber", lambda _s: object())
    monkeypatch.setattr(svc, "load_voice_memos", lambda _settings: {})

    # State object without has_archived_path triggers AttributeError branch.
    class _State:
        def __init__(self, _p: Path) -> None:
            return

        def get_state(self, _guid: str):
            return (None, None)

        def close(self):
            return

    monkeypatch.setattr(svc, "StateStore", _State)
    service = svc.VoiceMemoService(settings)

    audio = settings.archive_dir / "2026-01-01_00-00-00_example.m4a"  # type: ignore[operator]
    audio.write_text("x")
    service._scan_archive_for_untranscribed()
