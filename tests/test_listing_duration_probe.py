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


def test_collect_recordings_includes_inbox_files(monkeypatch, tmp_path: Path) -> None:
    """Inbox files awaiting processing must show up (pending) in the listing —
    otherwise -l / --dry-run miss them every time."""
    inbox = tmp_path / "inbox"; inbox.mkdir()
    (inbox / "meeting-from-phone.m4a").write_bytes(b"x")
    transcripts = tmp_path / "t"; transcripts.mkdir()
    archive = tmp_path / "a"; archive.mkdir()

    settings = SimpleNamespace(
        state_db=tmp_path / "state.sqlite",
        transcript_dir=transcripts,
        archive_dir=archive,
        inbox_dir=inbox,
    )

    # No DB, no Voice Memos source — only the Inbox file exists.
    class _EmptyStore:
        def __init__(self, *_a, **_k): pass
        def get_all_processed(self): return []
        def close(self): pass
    monkeypatch.setattr(listing, "StateStore", _EmptyStore)
    monkeypatch.setattr(listing, "list_voice_memos", lambda _s: [])

    items = listing.collect_recordings(settings)
    inbox_items = [i for i in items if "meeting-from-phone" in (i.title or i.key)]
    assert len(inbox_items) == 1
    it = inbox_items[0]
    assert it.has_source is True          # shows as a source...
    assert it.has_transcript is False     # ...that is still pending
