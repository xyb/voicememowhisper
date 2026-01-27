from __future__ import annotations

import subprocess
from types import SimpleNamespace
from pathlib import Path

import voicememowhisper.listing as listing


def test_probe_audio_duration_parses_estimated_duration(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "example.m4a"
    audio.write_bytes(b"x")

    stdout = "estimated duration: 65.000000 sec\n"
    monkeypatch.setattr(
        listing.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(returncode=0, stdout=stdout, stderr=""),
    )
    assert listing.probe_audio_duration_seconds(audio) == 65.0


def test_probe_audio_duration_returns_none_on_nonzero_rc(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "example.m4a"
    audio.write_bytes(b"x")

    monkeypatch.setattr(
        listing.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(returncode=1, stdout="", stderr=""),
    )
    assert listing.probe_audio_duration_seconds(audio) is None


def test_probe_audio_duration_returns_none_on_missing_duration(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "example.m4a"
    audio.write_bytes(b"x")

    monkeypatch.setattr(
        listing.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(returncode=0, stdout="no duration here\n", stderr=""),
    )
    assert listing.probe_audio_duration_seconds(audio) is None


def test_probe_audio_duration_returns_none_on_timeout(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "example.m4a"
    audio.write_bytes(b"x")

    def _raise(*_a, **_k):
        raise subprocess.TimeoutExpired(cmd=["afinfo"], timeout=1)

    monkeypatch.setattr(listing.subprocess, "run", _raise)
    assert listing.probe_audio_duration_seconds(audio) is None
