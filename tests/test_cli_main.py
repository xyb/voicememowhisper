from __future__ import annotations

import logging
from dataclasses import dataclass

import pytest

import voicememowhisper.cli as cli
import voicememowhisper.service as service_mod
from voicememowhisper.config import Settings
from voicememowhisper.list_model import RecordingItem


def test_verbosity_filter_gates_info_and_debug() -> None:
    info = logging.LogRecord("x", logging.INFO, "p", 1, "m", args=(), exc_info=None)
    debug = logging.LogRecord("x", logging.DEBUG, "p", 1, "m", args=(), exc_info=None)

    assert cli._VerbosityFilter(0).filter(info) is False
    assert cli._VerbosityFilter(1).filter(info) is True

    assert cli._VerbosityFilter(1).filter(debug) is False
    assert cli._VerbosityFilter(2).filter(debug) is True


def test_configure_logging_adds_filter() -> None:
    cli._configure_logging("INFO", verbosity=1)
    root = logging.getLogger()
    assert any(
        any(isinstance(f, cli._VerbosityFilter) for f in h.filters) for h in root.handlers
    )


def test_build_settings_archive_flag_enables_archive(tmp_path, monkeypatch) -> None:
    base = Settings(
        container_root=tmp_path,
        recordings_dir=tmp_path / "recordings",
        metadata_db=tmp_path / "metadata.db",
        legacy_metadata_db=None,
        transcript_dir=tmp_path / "transcripts",
        archive_dir=tmp_path / "archive",
        archive_enabled=False,
        inbox_dir=None,
        state_db=tmp_path / "state.sqlite",
        whisperkit_cli="whisperkit-cli",
        whisperkit_model="base-model",
        whisperkit_extra_args=(),
        language=None,
        processing_order="newest-first",
    )
    monkeypatch.setattr(cli, "load_settings", lambda: base)
    args = type(
        "Args",
        (),
        dict(
            model=None,
            language=None,
            newest_first=True,
            transcript_dir=None,
            archive_dir=None,
            archive=True,
        ),
    )()
    s = cli.build_settings(args)
    assert s.archive_enabled is True


def test_main_list_calls_list_recordings_with_limit(monkeypatch, tmp_path) -> None:
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

    monkeypatch.setattr(cli, "_configure_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "build_settings", lambda _args: settings)

    called: dict[str, int] = {}

    def _fake_list(_settings: Settings, *, limit: int = 10) -> int:
        called["limit"] = limit
        return 0

    monkeypatch.setattr(cli, "_list_recordings", _fake_list)
    assert cli.main(["--list", "-n", "5"]) == 0
    assert called["limit"] == 5


def test_main_returns_1_when_build_settings_fails(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_configure_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "build_settings", lambda _args: (_ for _ in ()).throw(ValueError("boom")))
    assert cli.main(["--list"]) == 1


def test_main_returns_1_when_service_init_fails(monkeypatch, tmp_path) -> None:
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

    monkeypatch.setattr(cli, "_configure_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "build_settings", lambda _args: settings)
    monkeypatch.setattr(service_mod, "VoiceMemoService", lambda _s: (_ for _ in ()).throw(RuntimeError("boom")))
    assert cli.main([]) == 1


@dataclass
class _DummyService:
    started: bool = False
    stopped: bool = False
    joined: bool = False
    watch: bool = False

    def __init__(self, _settings: Settings) -> None:
        return

    def start(self, *, watch: bool = False) -> None:
        self.started = True
        self.watch = watch

    def join(self) -> None:
        self.joined = True

    def stop(self) -> None:
        self.stopped = True


def test_main_runs_service_non_watch(monkeypatch, tmp_path) -> None:
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

    monkeypatch.setattr(cli, "_configure_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "build_settings", lambda _args: settings)

    dummy = _DummyService(settings)
    monkeypatch.setattr(service_mod, "VoiceMemoService", lambda _s: dummy)

    assert cli.main([]) == 0
    assert dummy.started is True
    assert dummy.watch is False
    assert dummy.joined is True
    assert dummy.stopped is True


def test_main_watch_exits_on_keyboard_interrupt(monkeypatch, tmp_path) -> None:
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

    monkeypatch.setattr(cli, "_configure_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "build_settings", lambda _args: settings)

    dummy = _DummyService(settings)
    monkeypatch.setattr(service_mod, "VoiceMemoService", lambda _s: dummy)
    monkeypatch.setattr(cli.time, "sleep", lambda _s: (_ for _ in ()).throw(KeyboardInterrupt()))

    assert cli.main(["--watch"]) == 0
    assert dummy.started is True
    assert dummy.watch is True
    assert dummy.stopped is True


def test_list_recordings_returns_1_on_collect_error(monkeypatch, tmp_path, capsys) -> None:
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
    monkeypatch.setattr(cli, "collect_recordings", lambda _s: (_ for _ in ()).throw(RuntimeError("boom")))
    assert cli._list_recordings(settings, limit=0) == 1
    _ = capsys.readouterr()


def test_list_recordings_covers_probe_and_persist_exceptions(monkeypatch, tmp_path, capsys) -> None:
    settings = Settings(
        container_root=tmp_path,
        recordings_dir=tmp_path / "recordings",
        metadata_db=tmp_path / "metadata.db",
        legacy_metadata_db=None,
        transcript_dir=tmp_path / "transcripts",
        archive_dir=tmp_path / "archive",
        archive_enabled=False,
        inbox_dir=None,
        state_db=tmp_path / "state.sqlite",
        whisperkit_cli="whisperkit-cli",
        whisperkit_model="dummy-model",
        whisperkit_extra_args=(),
        language=None,
        processing_order="newest-first",
    )

    item = RecordingItem(
        key="k",
        has_archive=True,
        archive_path=tmp_path / "archive" / "example.m4a",
        duration=None,
    )
    monkeypatch.setattr(cli, "collect_recordings", lambda _s: [item])

    # Probe raises -> should be swallowed and duration remains None.
    monkeypatch.setattr(cli.listing_mod, "probe_audio_duration_seconds", lambda _p: (_ for _ in ()).throw(RuntimeError("boom")))
    assert cli._list_recordings(settings, limit=0) == 0
    _ = capsys.readouterr()

    # Probe succeeds but persistence fails.
    monkeypatch.setattr(cli.listing_mod, "probe_audio_duration_seconds", lambda _p: 1.0)

    class _BadStore:
        def __init__(self, _p: Path) -> None:
            raise RuntimeError("no db")

    monkeypatch.setattr(cli, "StateStore", _BadStore)
    item.duration = None
    assert cli._list_recordings(settings, limit=0) == 0
    _ = capsys.readouterr()
