from __future__ import annotations

import os
import shutil
import sys
import tempfile
import types
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from voicememowhisper.config import Settings
from voicememowhisper.metadata import VoiceMemo
from voicememowhisper.naming import sanitize_filename
from voicememowhisper.paths import ensure_directories
import voicememowhisper.service as svc


def _install_watchdog_stubs() -> None:
    """Provide minimal watchdog stubs so tests don't need the real dependency."""
    if "watchdog.observers" in sys.modules:
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

    sys.modules["watchdog"] = watchdog
    sys.modules["watchdog.observers"] = observers
    sys.modules["watchdog.events"] = events


_install_watchdog_stubs()


class FakeState:
    def __init__(self, _path: Path) -> None:
        self.data: dict[str, tuple[Path | None, Path | None]] = {}

    def get_state(self, guid: str) -> tuple[Path | None, Path | None]:
        return self.data.get(guid, (None, None))

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


class ServiceCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())
        self.recordings = self.tmpdir / "recordings"
        self.recordings.mkdir()
        self.archive = self.tmpdir / "archive"
        self.archive.mkdir()
        self.inbox = self.tmpdir / "inbox"
        self.inbox.mkdir()
        self.transcripts = self.tmpdir / "transcripts"
        self.transcripts.mkdir()
        self.state_db = self.tmpdir / "state.db"

        self.settings = Settings(
            container_root=self.tmpdir,
            recordings_dir=self.recordings,
            metadata_db=self.tmpdir / "metadata.db",
            legacy_metadata_db=None,
            transcript_dir=self.transcripts,
            archive_dir=self.archive,
            archive_enabled=True,
            inbox_dir=self.inbox,
            state_db=self.state_db,
            whisperkit_cli="echo",
            whisperkit_model="dummy-model",
            whisperkit_extra_args=(),
            language=None,
            processing_order="newest-first",
        )

        self.patcher_transcriber = patch.object(svc, "WhisperTranscriber", FakeTranscriber)
        self.patcher_state = patch.object(svc, "StateStore", FakeState)
        self.patcher_metadata = patch.object(svc, "load_voice_memos", lambda _settings: {})

        self.patcher_transcriber.start()
        self.patcher_state.start()
        self.patcher_metadata.start()

        self.service = svc.VoiceMemoService(self.settings)

    def tearDown(self) -> None:
        self.patcher_transcriber.stop()
        self.patcher_state.stop()
        self.patcher_metadata.stop()
        shutil.rmtree(self.tmpdir)

    def test_process_memo_transcribes_and_archives_and_marks_state(self) -> None:
        audio = self.recordings / "foo.m4a"
        audio.write_text("audio")
        mtime = 1_700_000_000  # deterministic timestamp
        os.utime(audio, (mtime, mtime))

        memo = self.service._memo_for_path(audio)
        self.service._process_memo(memo)

        ts_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d_%H-%M-%S")

        # Transcript name matches timestamp + stem
        transcripts = list(self.transcripts.glob(f"{ts_str}_foo.txt"))
        self.assertEqual(len(transcripts), 1)
        self.assertIn(f"TRANSCRIPT:{ts_str}_foo.m4a", transcripts[0].read_text())

        # Archive keeps timestamped naming
        archives = list(self.archive.glob(f"{ts_str}_foo*.m4a"))
        self.assertEqual(len(archives), 1)

        # State recorded
        transcript_path, archived_path = self.service.state.get_state("foo")
        self.assertIsNotNone(transcript_path)
        self.assertIsNotNone(archived_path)

    def test_transcript_filename_uses_stem_title_part_when_timestamped(self) -> None:
        audio = self.recordings / "2026-01-26_15-30-03_2026-01-26 Interview_Test_DataEngineer.m4a"
        audio.write_text("audio")

        created_at = datetime(2026, 1, 26, 15, 30, 3)
        memo = VoiceMemo(
            guid=audio.stem,
            path=audio,
            title="IGNORED_METADATA_TITLE",
            created_at=created_at,
        )
        name = self.service._transcript_filename(audio)
        expected = (
            f"{created_at.strftime('%Y-%m-%d_%H-%M-%S')}_"
            f"{sanitize_filename('2026-01-26 Interview_Test_DataEngineer')}.txt"
        )
        self.assertEqual(name, expected)

    def test_enqueue_path_skips_when_transcript_already_present(self) -> None:
        audio = self.recordings / "bar.m4a"
        audio.write_text("audio")
        # Pre-mark transcript
        self.service.state.data["bar"] = (self.transcripts / "bar.txt", self.archive / "bar.m4a")

        self.service.enqueue_path(audio)
        self.assertTrue(self.service._queue.empty())
        self.assertNotIn("bar", self.service._inflight)

    def test_enqueue_path_for_archive_file_does_not_duplicate_archive(self) -> None:
        archived = self.archive / "archived.m4a"
        archived.write_text("audio")

        self.service.enqueue_path(archived)
        queued = self.service._queue.get(timeout=1)
        self.assertEqual(queued, archived)
        self.assertIn("archived", self.service._inflight)

    def test_scan_archive_backfills_untranscribed(self) -> None:
        archived = self.archive / "needs.m4a"
        archived.write_text("audio")

        self.service._scan_archive_for_untranscribed()
        queued = self.service._queue.get(timeout=1)
        self.assertEqual(queued, archived)
        self.assertIn("needs", self.service._inflight)


class PathsTests(unittest.TestCase):
    def test_ensure_directories_creates_inbox_and_archive(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        settings = Settings(
            container_root=tmpdir,
            recordings_dir=tmpdir / "rec",
            metadata_db=tmpdir / "metadata.db",
            legacy_metadata_db=None,
            transcript_dir=tmpdir / "out" / "transcripts",
            archive_dir=tmpdir / "out" / "archive",
            archive_enabled=True,
            inbox_dir=tmpdir / "out" / "inbox",
            state_db=tmpdir / "state" / "db.sqlite",
            whisperkit_cli="echo",
            whisperkit_model="dummy-model",
            whisperkit_extra_args=(),
            language=None,
            processing_order="newest-first",
        )

        ensure_directories(settings)
        self.assertTrue(settings.transcript_dir.exists())
        self.assertTrue(settings.inbox_dir.exists())
        self.assertTrue(settings.archive_dir.exists())
        self.assertTrue(settings.state_db.parent.exists())

        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    unittest.main()
