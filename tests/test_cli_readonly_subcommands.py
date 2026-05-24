"""Read-only entry points must not be blocked by the single-instance lock.

The user complained: `python -m voicememowhisper ls` got rejected with
"Another instance already running" while a long-running transcribe held
the lock. Two failure modes:

1. `-l` / `--list` is read-only and already bypasses the lock — but `ls`
   and `list` (the natural typo) fall through to the positional `audio`
   path, which is locked.
2. Any non-existent `audio` argument waits on the lock first, then dies
   later — `voicememo-whisper foo` should fail-fast outside the lock
   with a hint pointing at `-l`.

These tests guard both behaviours so they don't regress.
"""

from __future__ import annotations

# pyright: reportMissingImports=false

from pathlib import Path

import pytest

import voicememowhisper.cli as cli
from voicememowhisper.config import Settings


def _stub_settings(tmp_path: Path) -> Settings:
    return Settings(
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


@pytest.mark.parametrize("alias", ["ls", "list"])
def test_main_ls_and_list_aliases_route_to_list_recordings(
    alias: str, monkeypatch, tmp_path
) -> None:
    """`ls` / `list` should behave like `-l`: list and exit, no lock."""
    settings = _stub_settings(tmp_path)
    monkeypatch.setattr(cli, "_configure_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "build_settings", lambda _args: settings)

    called: dict[str, object] = {}

    def _fake_list(_settings: Settings, *, limit: int = 10) -> int:
        called["limit"] = limit
        return 0

    def _explode_lock(*_a, **_k):
        raise AssertionError("read-only alias must not acquire single_instance_lock")

    monkeypatch.setattr(cli, "_list_recordings", _fake_list)
    # Patch the lock symbol in its source module — main() imports it lazily.
    import voicememowhisper._lock as lock_mod
    monkeypatch.setattr(lock_mod, "single_instance_lock", _explode_lock)

    assert cli.main([alias]) == 0
    assert called["limit"] == 10


@pytest.mark.parametrize("alias", ["ls", "list"])
def test_main_ls_and_list_aliases_pass_through_limit(
    alias: str, monkeypatch, tmp_path
) -> None:
    """`ls -n 3` / `list -n 3` should still honour `-n`."""
    settings = _stub_settings(tmp_path)
    monkeypatch.setattr(cli, "_configure_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "build_settings", lambda _args: settings)

    called: dict[str, object] = {}

    def _fake_list(_settings: Settings, *, limit: int = 10) -> int:
        called["limit"] = limit
        return 0

    monkeypatch.setattr(cli, "_list_recordings", _fake_list)

    assert cli.main([alias, "-n", "3"]) == 0
    assert called["limit"] == 3


def test_main_nonexistent_audio_fails_fast_before_lock(
    monkeypatch, tmp_path, caplog
) -> None:
    """`voicememo-whisper /no/such/file.m4a` must error out without ever
    touching `single_instance_lock` — otherwise a running transcribe in
    another shell would make a plain typo look like a "stuck" pipeline."""
    settings = _stub_settings(tmp_path)
    monkeypatch.setattr(cli, "_configure_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "build_settings", lambda _args: settings)

    def _explode_lock(*_a, **_k):
        raise AssertionError("typo path must not reach single_instance_lock")

    import voicememowhisper._lock as lock_mod
    monkeypatch.setattr(lock_mod, "single_instance_lock", _explode_lock)

    missing = tmp_path / "definitely-not-here.m4a"
    assert cli.main([str(missing)]) == 1
    # The hint must mention `-l` so the user can self-recover.
    combined = " ".join(rec.getMessage() for rec in caplog.records).lower()
    assert "not found" in combined
    assert "-l" in combined
