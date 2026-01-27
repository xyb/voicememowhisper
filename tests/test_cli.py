from __future__ import annotations

# pyright: reportMissingImports=false

import argparse
from datetime import datetime
from pathlib import Path

import pytest

import voicememowhisper.cli as cli
import voicememowhisper.listing as listing
from voicememowhisper.cli import _format_duration, _list_recordings, _parse_filename, build_settings
from voicememowhisper.config import Settings
from voicememowhisper.metadata import VoiceMemo


class FakeStateStore:
    def __init__(self, _path: Path) -> None:
        return

    def get_all_processed(self) -> list[dict]:
        return []

    def close(self) -> None:
        return


def test_format_duration() -> None:
    assert _format_duration(None) == "-"
    assert _format_duration(5) == "5s"
    assert _format_duration(65) == "1m05s"


def test_parse_filename_timestamped() -> None:
    dt_str, title = _parse_filename(Path("2026-01-27_09-30-03_meeting.m4a"))
    assert dt_str == "2026-01-27T09:30:03"
    assert title == "meeting"


def test_parse_filename_undated_prefix() -> None:
    dt_str, title = _parse_filename(Path("undated_foo.txt"))
    assert dt_str is None
    assert title == "foo"


def test_parse_filename_fallback_stem() -> None:
    dt_str, title = _parse_filename(Path("plain_name.txt"))
    assert dt_str is None
    assert title == "plain_name"


def test_build_settings_applies_overrides(tmp_path, monkeypatch) -> None:
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

    args = argparse.Namespace(
        model="new-model",
        language="zh",
        newest_first=False,
        transcript_dir=str(tmp_path / "out"),
        archive_dir=str(tmp_path / "aud"),
        archive=False,
    )
    monkeypatch.setattr(cli, "load_settings", lambda: base)
    s = build_settings(args)

    assert s.whisperkit_model == "new-model"
    assert s.language == "zh"
    assert s.processing_order == "oldest-first"
    assert s.transcript_dir == tmp_path / "out"
    assert s.archive_enabled is True
    assert s.archive_dir == tmp_path / "aud"


def test_list_recordings_dedups_same_time_and_title(tmp_path, monkeypatch, capsys) -> None:
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    archive = tmp_path / "archive"
    archive.mkdir()

    settings = Settings(
        container_root=tmp_path,
        recordings_dir=recordings,
        metadata_db=tmp_path / "metadata.db",
        legacy_metadata_db=None,
        transcript_dir=transcripts,
        archive_dir=archive,
        archive_enabled=False,
        inbox_dir=None,
        state_db=tmp_path / "state.sqlite",
        whisperkit_cli="whisperkit-cli",
        whisperkit_model="dummy-model",
        whisperkit_extra_args=(),
        language=None,
        processing_order="newest-first",
    )

    # Create one transcript file representing an "orphan" entry.
    transcript = transcripts / "2026-01-27_09-30-03_meeting.txt"
    transcript.write_text("t")

    # App list also has a memo at the same second with same title.
    local_tz = datetime.now().astimezone().tzinfo
    created_at = datetime(2026, 1, 27, 9, 30, 3, tzinfo=local_tz)
    memos = [
        VoiceMemo(
            guid="app-guid",
            path=recordings / "app-guid.m4a",
            title="meeting",
            created_at=created_at,
            duration_seconds=12.0,
            is_trashed=False,
        )
    ]

    monkeypatch.setattr(listing, "StateStore", FakeStateStore)
    monkeypatch.setattr(listing, "list_voice_memos", lambda _settings: memos)
    rc = _list_recordings(settings, limit=0)

    assert rc == 0
    out = capsys.readouterr().out
    assert out.count("meeting") == 1
    assert "TAS" in out.replace(" ", "")


def test_list_recordings_limit_defaults_to_10(tmp_path, monkeypatch, capsys) -> None:
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()

    settings = Settings(
        container_root=tmp_path,
        recordings_dir=recordings,
        metadata_db=tmp_path / "metadata.db",
        legacy_metadata_db=None,
        transcript_dir=transcripts,
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

    # Create 12 transcript files; ensure ordering is by timestamp in filename.
    for i in range(12):
        ts = f"2026-01-27_09-30-{i:02d}"
        (transcripts / f"{ts}_example_meeting.txt").write_text("t")

    monkeypatch.setattr(listing, "StateStore", FakeStateStore)
    monkeypatch.setattr(listing, "list_voice_memos", lambda _settings: [])

    rc = _list_recordings(settings)
    assert rc == 0
    out = capsys.readouterr().out
    assert "Title (10/12)" in out
    item_lines = [ln for ln in out.splitlines() if ln.startswith(("✓", ".")) and len(ln) >= 3]
    assert len(item_lines) == 10


def test_list_recordings_limit_can_be_overridden(tmp_path, monkeypatch, capsys) -> None:
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()

    settings = Settings(
        container_root=tmp_path,
        recordings_dir=recordings,
        metadata_db=tmp_path / "metadata.db",
        legacy_metadata_db=None,
        transcript_dir=transcripts,
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

    for i in range(12):
        ts = f"2026-01-27_09-30-{i:02d}"
        (transcripts / f"{ts}_example_meeting.txt").write_text("t")

    monkeypatch.setattr(listing, "StateStore", FakeStateStore)
    monkeypatch.setattr(listing, "list_voice_memos", lambda _settings: [])

    rc = _list_recordings(settings, limit=5)
    assert rc == 0
    out = capsys.readouterr().out
    assert "Title (5/12)" in out
    item_lines = [ln for ln in out.splitlines() if ln.startswith(("✓", ".")) and len(ln) >= 3]
    assert len(item_lines) == 5

