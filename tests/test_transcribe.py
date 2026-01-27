from __future__ import annotations

import types
from unittest.mock import patch

import pytest

from voicememowhisper.transcribe import WhisperTranscriber


def test_transcribe_builds_expected_command(settings_factory, tmp_path, monkeypatch) -> None:
    settings = settings_factory(
        archive_enabled=False,
        inbox_dir=None,
        whisperkit_cli="whisperkit-cli",
        whisperkit_model="dummy-model",
        whisperkit_extra_args=("--extra", "1"),
        language="en",
    )
    audio = tmp_path / "audio.m4a"
    audio.write_text("audio")

    calls: list[list[str]] = []

    def fake_run(cmd, capture_output, text):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=0, stdout="hello", stderr="")

    monkeypatch.setattr("voicememowhisper.transcribe.shutil.which", lambda _b: "/bin/whisperkit-cli")
    monkeypatch.setattr("voicememowhisper.transcribe.subprocess.run", fake_run)

    t = WhisperTranscriber(settings)
    out = t.transcribe(audio, label="x")

    assert out == "hello"
    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[0] == "/bin/whisperkit-cli"
    assert "--model" in cmd and "dummy-model" in cmd
    assert "--audio-path" in cmd and str(audio) in cmd
    assert "--language" in cmd and "en" in cmd
    assert "--extra" in cmd


def test_transcribe_raises_on_failure(settings_factory, tmp_path, monkeypatch) -> None:
    settings = settings_factory(
        archive_enabled=False,
        inbox_dir=None,
        whisperkit_cli="whisperkit-cli",
        whisperkit_model="dummy-model",
        whisperkit_extra_args=(),
        language=None,
    )
    audio = tmp_path / "audio.m4a"
    audio.write_text("audio")

    def fake_run(_cmd, capture_output, text):
        return types.SimpleNamespace(returncode=2, stdout="", stderr="boom")

    monkeypatch.setattr("voicememowhisper.transcribe.shutil.which", lambda _b: "/bin/whisperkit-cli")
    monkeypatch.setattr("voicememowhisper.transcribe.subprocess.run", fake_run)

    t = WhisperTranscriber(settings)
    with pytest.raises(RuntimeError):
        t.transcribe(audio)


def test_resolve_cli_binary_uses_existing_path(settings_factory, tmp_path, monkeypatch) -> None:
    cli_path = tmp_path / "wk"
    cli_path.write_text("#!/bin/sh\n")
    settings = settings_factory(
        archive_enabled=False,
        inbox_dir=None,
        whisperkit_cli=str(cli_path),
    )

    monkeypatch.setattr("voicememowhisper.transcribe.shutil.which", lambda _b: None)
    t = WhisperTranscriber(settings)
    assert t._cli == str(cli_path)

