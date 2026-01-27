from __future__ import annotations

# pyright: reportMissingImports=false

import os
import queue
from datetime import datetime

import pytest

from voicememowhisper.metadata import VoiceMemo
from voicememowhisper.naming import sanitize_filename
from voicememowhisper.paths import ensure_directories


def test_process_memo_transcribes_and_archives_and_marks_state(service, settings) -> None:
    audio = settings.recordings_dir / "foo.m4a"
    audio.write_text("audio")
    mtime = 1_700_000_000  # deterministic timestamp
    os.utime(audio, (mtime, mtime))

    memo = service._memo_for_path(audio)
    service._process_memo(memo)

    ts_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d_%H-%M-%S")
    transcripts = list(settings.transcript_dir.glob(f"{ts_str}_foo.txt"))
    assert len(transcripts) == 1
    assert "TRANSCRIPT:foo.m4a" in transcripts[0].read_text()

    archives = list(settings.archive_dir.glob(f"{ts_str}_foo*.m4a"))  # type: ignore[union-attr]
    assert len(archives) == 1

    transcript_path, archived_path = service.state.get_state("foo")
    assert transcript_path is not None
    assert archived_path is not None


def test_transcript_filename_uses_stem_title_part_when_timestamped(service, settings) -> None:
    audio = settings.recordings_dir / "2026-01-26_15-30-03_2026-01-26 Interview_Test.m4a"
    audio.write_text("audio")

    created_at = datetime(2026, 1, 26, 15, 30, 3)
    memo = VoiceMemo(guid=audio.stem, path=audio, created_at=created_at)
    name = service._transcript_filename(memo)
    expected = (
        f"{created_at.strftime('%Y-%m-%d_%H-%M-%S')}_"
        f"{sanitize_filename('2026-01-26 Interview_Test')}.txt"
    )
    assert name == expected


def test_enqueue_path_skips_when_transcript_already_present(service, settings) -> None:
    audio = settings.recordings_dir / "bar.m4a"
    audio.write_text("audio")
    # Pre-mark transcript
    service.state.data["bar"] = (settings.transcript_dir / "bar.txt", settings.archive_dir / "bar.m4a")  # type: ignore[union-attr]

    service.enqueue_path(audio)
    assert service._queue.empty()
    assert "bar" not in service._inflight


def test_enqueue_path_for_archive_file_does_not_duplicate_archive(service, settings) -> None:
    archived = settings.archive_dir / "archived.m4a"  # type: ignore[operator]
    archived.write_text("audio")

    service.enqueue_path(archived)
    queued = service._queue.get(timeout=1)
    assert queued == archived
    assert "archived" in service._inflight


def test_scan_archive_backfills_untranscribed(service, settings) -> None:
    archived = settings.archive_dir / "needs.m4a"  # type: ignore[operator]
    archived.write_text("audio")

    service._scan_archive_for_untranscribed()
    queued = service._queue.get(timeout=1)
    assert queued == archived
    assert "needs" in service._inflight


def test_scan_archive_does_not_reprocess_files_already_archived_in_state(service, settings) -> None:
    audio = settings.recordings_dir / "foo.m4a"
    audio.write_text("audio")
    mtime = 1_700_000_000  # deterministic timestamp
    os.utime(audio, (mtime, mtime))

    memo = service._memo_for_path(audio)
    service._process_memo(memo)

    service._scan_archive_for_untranscribed()

    with pytest.raises(queue.Empty):
        service._queue.get_nowait()


def test_ensure_directories_creates_inbox_and_archive(settings_factory, tmp_path) -> None:
    settings = settings_factory(
        transcript_dir=tmp_path / "out" / "transcripts",
        inbox_dir=tmp_path / "out" / "inbox",
        archive_dir=tmp_path / "out" / "archive",
        state_db=tmp_path / "state" / "db.sqlite",
    )

    ensure_directories(settings)
    assert settings.transcript_dir.exists()
    assert settings.inbox_dir is not None and settings.inbox_dir.exists()
    assert settings.archive_dir is not None and settings.archive_dir.exists()
    assert settings.state_db.parent.exists()
