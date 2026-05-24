from __future__ import annotations

import sys
import types
import importlib.util

import pytest

from voicememowhisper.config import Settings
from dataclasses import replace
from pathlib import Path


def _install_watchdog_stubs() -> None:
    """
    Provide minimal watchdog stubs so unit tests don't require the real dependency.

    Some modules import watchdog at import time (e.g., service/cli), so this must
    run during test collection.
    """
    if importlib.util.find_spec("watchdog") is not None:
        return

    class DummyObserver:
        def schedule(self, *args, **kwargs):
            return None

        def start(self):
            return None

        def stop(self):
            return None

        def join(self):
            return None

    class DummyEvent:
        def __init__(self, src_path: str = "", is_directory: bool = False, dest_path: str | None = None):
            self.src_path = src_path
            self.is_directory = is_directory
            self.dest_path = dest_path

    class DummyHandler:
        def __init__(self, *args, **kwargs):
            return None

    observers = types.ModuleType("watchdog.observers")
    observers.Observer = DummyObserver
    events = types.ModuleType("watchdog.events")
    events.FileSystemEvent = DummyEvent
    events.FileSystemEventHandler = DummyHandler
    watchdog = types.ModuleType("watchdog")
    watchdog.observers = observers
    watchdog.events = events

    sys.modules.setdefault("watchdog", watchdog)
    sys.modules.setdefault("watchdog.observers", observers)
    sys.modules.setdefault("watchdog.events", events)


_install_watchdog_stubs()


class FakeState:
    def __init__(self, _path: Path) -> None:
        self.data: dict[str, tuple[Path | None, Path | None]] = {}

    def get_state(self, guid: str) -> tuple[Path | None, Path | None]:
        return self.data.get(guid, (None, None))

    def has_archived_path(self, archived_path: Path) -> bool:
        for _guid, (_transcript, archived) in self.data.items():
            if archived == archived_path:
                return True
        return False

    def find_by_archived_basename(self, basename: str) -> list[tuple[str, Path]]:
        return [
            (guid, archived)
            for guid, (_transcript, archived) in self.data.items()
            if archived is not None and archived.name == basename
        ]

    def update_archived_path(self, guid: str, new_archived_path: Path) -> int:
        if guid not in self.data:
            return 0
        transcript, _ = self.data[guid]
        self.data[guid] = (transcript, new_archived_path)
        return 1

    def mark_processed(
        self,
        guid: str,
        transcript_path: Path | None,
        archived_path: Path | None,
        title: str | None = None,
        duration: float | None = None,
        created_at: str | None = None,
    ) -> None:
        self.data[guid] = (transcript_path, archived_path)

    def close(self) -> None:
        return


class FakeTranscriber:
    def __init__(self, _settings: Settings) -> None:
        return

    def transcribe(self, audio_path: Path, *, label: str | None = None) -> str:
        return f"TRANSCRIPT:{audio_path.name}"


@pytest.fixture()
def settings_factory(tmp_path: Path) -> callable:
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    archive = tmp_path / "archive"
    archive.mkdir()
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()

    base = Settings(
        container_root=tmp_path,
        recordings_dir=recordings,
        metadata_db=tmp_path / "metadata.db",
        legacy_metadata_db=None,
        transcript_dir=transcripts,
        archive_dir=archive,
        archive_enabled=True,
        inbox_dir=inbox,
        state_db=tmp_path / "state.sqlite",
        whisperkit_cli="echo",
        whisperkit_model="dummy-model",
        whisperkit_extra_args=(),
        language=None,
        processing_order="newest-first",
        # Service-level tests exercise the WhisperKit path with a fake
        # transcriber. The speaker pipeline would otherwise try to call
        # a live funasr server on fake bytes — fine when failures used
        # to silently fall back to WhisperKit, but those silent
        # fallbacks were removed (see test_processor_no_silent_fallback).
        # Default off here; opt in per-test by passing
        # ``speaker_pipeline_enabled=True`` to ``settings_factory``.
        speaker_pipeline_enabled=False,
    )

    def _make(**overrides):
        return replace(base, **overrides)

    return _make


@pytest.fixture()
def settings(settings_factory) -> Settings:
    return settings_factory()


@pytest.fixture()
def service_factory(settings_factory, monkeypatch) -> callable:
    import voicememowhisper.service as svc

    def _make(settings: Settings | None = None) -> svc.VoiceMemoService:
        s = settings or settings_factory()
        monkeypatch.setattr(svc, "WhisperTranscriber", FakeTranscriber)
        monkeypatch.setattr(svc, "StateStore", FakeState)
        monkeypatch.setattr(svc, "load_voice_memos", lambda _settings: {})
        return svc.VoiceMemoService(s)

    return _make


@pytest.fixture()
def service(service_factory, settings):
    return service_factory(settings)

