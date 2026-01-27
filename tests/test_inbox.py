from __future__ import annotations

import shutil
import sys
import tempfile
import types
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch


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

from voicememowhisper.config import Settings
import voicememowhisper.service as svc


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


class InboxImportTests(unittest.TestCase):
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

    def test_process_inbox_uses_filename_date_and_syncs_mtime(self) -> None:
        fixed_dt = datetime(2024, 1, 2, 3, 4, 5)
        with patch.object(svc, "_date_from_filename", lambda _p: fixed_dt):
            inbox_file = self.inbox / "2024-01-02 foo.m4a"
            inbox_file.write_text("hello")

            dest = self.service._process_inbox_file(inbox_file)

        self.assertIsNotNone(dest)
        assert dest is not None
        self.assertFalse(inbox_file.exists())
        self.assertTrue(dest.exists())
        self.assertTrue(dest.name.startswith("2024-01-02_03-04-05_2024-01-02 foo"))
        self.assertAlmostEqual(dest.stat().st_mtime, fixed_dt.timestamp(), delta=2)

    def test_process_inbox_discards_duplicate_by_hash(self) -> None:
        fixed_dt = datetime(2024, 1, 2, 0, 0, 0)
        with patch.object(svc, "_date_from_filename", lambda _p: fixed_dt):
            archive_path = self.archive / "2024-01-02_00-00-00_2024-01-02 foo.m4a"
            archive_path.write_text("same-content")

            inbox_file = self.inbox / "2024-01-02 foo.m4a"
            inbox_file.write_text("same-content")

            dest = self.service._process_inbox_file(inbox_file)

        self.assertEqual(dest, archive_path)
        self.assertFalse(inbox_file.exists())
        self.assertEqual(archive_path.read_text(), "same-content")

    def test_process_inbox_conflict_different_hash_gets_suffix(self) -> None:
        fixed_dt = datetime(2024, 1, 2, 0, 0, 0)
        with patch.object(svc, "_date_from_filename", lambda _p: fixed_dt):
            archive_path = self.archive / "2024-01-02_00-00-00_2024-01-02 foo.m4a"
            archive_path.write_text("old")

            inbox_file = self.inbox / "2024-01-02 foo.m4a"
            inbox_file.write_text("new")

            dest = self.service._process_inbox_file(inbox_file)

        self.assertIsNotNone(dest)
        assert dest is not None
        self.assertNotEqual(dest, archive_path)
        self.assertTrue(dest.name.endswith("_1.m4a"))
        self.assertFalse(inbox_file.exists())
        self.assertEqual(archive_path.read_text(), "old")
        self.assertEqual(dest.read_text(), "new")

    def test_scan_archive_backfills_untranscribed(self) -> None:
        archive_file = self.archive / "foo.m4a"
        archive_file.write_text("audio")

        # FakeState starts empty; should enqueue for transcription
        self.service._scan_archive_for_untranscribed()

        queued = self.service._queue.get(timeout=1)
        self.assertEqual(queued, archive_file)
        self.assertIn("foo", self.service._inflight)


if __name__ == "__main__":
    unittest.main()
