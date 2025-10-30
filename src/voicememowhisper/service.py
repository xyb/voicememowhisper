from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional, Set

from watchdog.observers import Observer

from .config import Settings, load_settings
from .metadata import VoiceMemo, load_voice_memos, resolve_created_at
from .paths import ensure_directories
from .state import StateStore
from .transcribe import WhisperTranscriber
from .watcher import start_watcher

LOGGER = logging.getLogger(__name__)


def _sanitize_filename(value: str) -> str:
    safe_chars = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_", " "):
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    return "".join(safe_chars).strip() or "untitled"


class VoiceMemoService:
    """Coordinate scanning, watching, and transcription of voice memos."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = ensure_directories(settings or load_settings())

        if not self.settings.recordings_dir.exists():
            raise FileNotFoundError(
                f"Voice Memo recordings directory not found at {self.settings.recordings_dir}. "
                "Open the Voice Memos app or adjust VOICE_MEMO_RECORDINGS_DIR."
            )

        try:
            next(self.settings.recordings_dir.glob("*.m4a"))
        except StopIteration:
            pass
        except PermissionError as err:
            raise PermissionError(
                f"Insufficient permissions to read {self.settings.recordings_dir}. "
                "Grant the terminal Full Disk Access (System Settings → Privacy & Security → Full Disk Access)."
            ) from err

        self.transcriber = WhisperTranscriber(self.settings)
        self._queue: "queue.Queue[Path]" = queue.Queue()
        self._stop = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._observer: Optional[Observer] = None
        self.state = StateStore(self.settings.state_db)
        self._processed: Set[str] = set(self.state.known_guids())
        self._metadata: dict[str, VoiceMemo] = {}
        self._inflight: Set[str] = set()

    def start(self, watch: bool = False) -> None:
        """Start the worker thread and optionally the filesystem watcher."""
        LOGGER.info("Starting Voice Memo transcription service")
        self._worker_thread = threading.Thread(target=self._worker_loop, name="VoiceMemoWorker", daemon=True)
        self._worker_thread.start()

        self.enqueue_existing()

        if watch:
            self._observer = start_watcher(self.settings.recordings_dir, self.enqueue_path)

    def stop(self) -> None:
        LOGGER.info("Stopping Voice Memo transcription service")
        self._stop.set()
        if self._observer:
            self._observer.stop()
            self._observer.join()
        self._queue.put(None)  # type: ignore[arg-type]
        if self._worker_thread:
            self._worker_thread.join()
        self.state.close()

    def enqueue_existing(self) -> None:
        self._refresh_metadata()
        for path in sorted(self.settings.recordings_dir.glob("*.m4a")):
            self.enqueue_path(path)

    def enqueue_path(self, path: Path) -> None:
        guid = path.stem
        if guid in self._processed or guid in self._inflight:
            return
        LOGGER.debug("Enqueueing %s", path)
        self._queue.put(path)
        self._inflight.add(guid)

    def _refresh_metadata(self) -> None:
        self._metadata = load_voice_memos(self.settings)

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                break

            path = item
            guid = path.stem
            try:
                self._process_path(path)
            except Exception:
                LOGGER.exception("Failed to process %s", path)
            finally:
                self._queue.task_done()
                self._inflight.discard(guid)

    def _memo_for_path(self, path: Path) -> VoiceMemo:
        guid = path.stem
        memo = self._metadata.get(guid)
        if memo:
            if memo.path != path:
                memo = replace(memo, path=path)
            return memo
        return VoiceMemo(guid=guid, path=path)

    def _process_path(self, path: Path) -> None:
        if not path.exists():
            LOGGER.warning("Skipping missing path %s", path)
            return

        # Newly recorded files may still be written; retry a few times.
        for attempt in range(3):
            try:
                # Ensure file readable and non-empty
                if path.stat().st_size == 0:
                    raise OSError("File size is zero while recording may still be in progress.")
                break
            except OSError as err:
                LOGGER.debug("File %s not ready (%s). Retrying...", path, err)
                time.sleep(1.0)
        else:
            LOGGER.error("Giving up on %s after repeated readiness checks", path)
            return

        self._refresh_metadata()
        memo = self._memo_for_path(path)

        if memo.guid in self._processed:
            LOGGER.debug("Skipping already processed memo %s", memo.guid)
            return

        text = self.transcriber.transcribe(path)
        self._write_transcript(memo, text)
        self._processed.add(memo.guid)

    def _write_transcript(self, memo: VoiceMemo, text: str) -> None:
        timestamp = resolve_created_at(memo)
        if timestamp is None:
            timestamp_str = "undated"
        else:
            timestamp_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        title = memo.title or memo.guid
        filename = f"{timestamp_str}_{_sanitize_filename(title)}.txt"
        output_path = self.settings.transcript_dir / filename
        LOGGER.info("Writing transcript to %s", output_path)
        output_path.write_text(text + "\n", encoding="utf-8")
        self.state.mark_processed(memo.guid, output_path)

    def join(self) -> None:
        self._queue.join()
