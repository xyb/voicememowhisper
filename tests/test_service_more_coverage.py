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


class _HealingState:
    """In-memory state with the 3 methods needed for scan self-heal."""

    def __init__(self, seed: dict[str, tuple[Path | None, Path | None]] | None = None) -> None:
        self.data: dict[str, tuple[Path | None, Path | None]] = dict(seed or {})
        self.enqueued: list[Path] = []

    def __call__(self, _p: Path) -> "_HealingState":
        # Used as StateStore factory (monkeypatched class).
        return self

    def get_state(self, guid: str):
        return self.data.get(guid, (None, None))

    def has_archived_path(self, archived_path: Path) -> bool:
        return any(a == archived_path for (_t, a) in self.data.values())

    def find_by_archived_basename(self, basename: str) -> list[tuple[str, Path]]:
        return [
            (guid, a)
            for guid, (_t, a) in self.data.items()
            if a is not None and a.name == basename
        ]

    def update_archived_path(self, guid: str, new_path: Path) -> int:
        if guid not in self.data:
            return 0
        t, _ = self.data[guid]
        self.data[guid] = (t, new_path)
        return 1

    def close(self):
        return


def _make_service_with_state(settings, state, monkeypatch):
    monkeypatch.setattr(svc, "WhisperTranscriber", lambda _s: object())
    monkeypatch.setattr(svc, "load_voice_memos", lambda _settings: {})
    monkeypatch.setattr(svc, "StateStore", state)
    service = svc.VoiceMemoService(settings)
    monkeypatch.setattr(service, "enqueue_path", lambda p: state.enqueued.append(p))
    return service


def test_scan_archive_heals_stale_path_same_basename(tmp_path: Path, monkeypatch) -> None:
    """Archive dir was moved: state DB still points to old path, but the file
    with the same basename exists in the new archive_dir. Scan should update
    the state row to the new path and NOT enqueue."""
    settings = _base_settings(tmp_path)
    settings = replace(settings, archive_enabled=True)

    new_path = settings.archive_dir / "2024-08-15_10-42-46_sample_meeting.m4a"
    new_path.write_text("x")
    stale_path = Path("/old/VoiceMemoArchives/2024-08-15_10-42-46_sample_meeting.m4a")
    state = _HealingState({
        "apple-uuid-1": (tmp_path / "old-t.txt", stale_path),
    })

    service = _make_service_with_state(settings, state, monkeypatch)
    service._scan_archive_for_untranscribed()

    # Healed: archived_path rewritten to new path
    _, archived = state.get_state("apple-uuid-1")
    assert archived == new_path
    # Not enqueued (treated as already processed)
    assert state.enqueued == []


def test_scan_archive_does_not_heal_on_basename_collision(tmp_path: Path, monkeypatch) -> None:
    """Two stale rows happen to share a basename — refuse to self-heal,
    fall through to normal enqueue path (safer to reprocess than to rewrite
    the wrong row)."""
    settings = _base_settings(tmp_path)
    settings = replace(settings, archive_enabled=True)

    new_path = settings.archive_dir / "dup.m4a"
    new_path.write_text("x")
    state = _HealingState({
        "g1": (tmp_path / "t1.txt", Path("/a/dup.m4a")),
        "g2": (tmp_path / "t2.txt", Path("/b/dup.m4a")),
    })

    service = _make_service_with_state(settings, state, monkeypatch)
    service._scan_archive_for_untranscribed()

    # Neither row was rewritten
    assert state.get_state("g1")[1] == Path("/a/dup.m4a")
    assert state.get_state("g2")[1] == Path("/b/dup.m4a")
    # File falls through; stem "dup" isn't a known guid so enqueue is called
    assert state.enqueued == [new_path]


def test_scan_archive_new_file_still_enqueues(tmp_path: Path, monkeypatch) -> None:
    """Truly new archive file (no state row matches either by path or basename)
    must still be enqueued."""
    settings = _base_settings(tmp_path)
    settings = replace(settings, archive_enabled=True)

    new_path = settings.archive_dir / "2026-05-01_10-00-00_brand_new.m4a"
    new_path.write_text("x")
    state = _HealingState({})

    service = _make_service_with_state(settings, state, monkeypatch)
    service._scan_archive_for_untranscribed()

    assert state.enqueued == [new_path]


def test_scan_archive_exact_path_match_does_not_trigger_heal(tmp_path: Path, monkeypatch) -> None:
    """When the state row already points to the correct current path, do not
    call update_archived_path / do not log a heal."""
    settings = _base_settings(tmp_path)
    settings = replace(settings, archive_enabled=True)

    new_path = settings.archive_dir / "already-tracked.m4a"
    new_path.write_text("x")
    state = _HealingState({"g1": (tmp_path / "t.txt", new_path)})
    # Sentinel: if update_archived_path is called, fail the test.
    called = []
    orig = state.update_archived_path
    def _spy(guid, path):
        called.append((guid, path))
        return orig(guid, path)
    state.update_archived_path = _spy  # type: ignore[assignment]

    service = _make_service_with_state(settings, state, monkeypatch)
    service._scan_archive_for_untranscribed()

    assert called == []
    assert state.enqueued == []
