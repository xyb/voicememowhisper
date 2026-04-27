from __future__ import annotations

import logging
import queue
import shutil
import threading
import time
import os
from datetime import datetime
from dataclasses import replace
from pathlib import Path
from typing import Optional, Set

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from watchdog.observers import Observer  # type: ignore

from .config import Settings, load_settings
from .metadata import VoiceMemo, resolve_created_at, load_voice_memos
from .paths import ensure_directories
from .state import StateStore
from .transcribe import WhisperTranscriber
from .watcher import start_watcher
from .naming import sanitize_filename, title_from_stem
from .archive import ArchiveManager, hash_file as _archive_hash_file
from .inbox import InboxProcessor, date_from_filename as _inbox_date_from_filename
from .metadata_cache import MetadataCache
from .processor import MemoProcessor

LOGGER = logging.getLogger("service")

def _hash_file(path: Path, chunk_size: int = 1024 * 1024) -> Optional[str]:
    """Return hex digest of file content or None on error."""
    return _archive_hash_file(path, chunk_size=chunk_size)


def _date_from_filename(path: Path) -> Optional[datetime]:
    """Extract a leading YYYY-MM-DD from the filename stem if present."""
    return _inbox_date_from_filename(path)


class VoiceMemoService:
    """Coordinate scanning, watching, and transcription of voice memos."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = ensure_directories(settings or load_settings())
        self.processing_order = self.settings.processing_order

        if not self.settings.recordings_dir.exists():
            raise FileNotFoundError(
                f"Voice Memo recordings directory not found at {self.settings.recordings_dir}. "
                "Open the Voice Memos app or adjust VOICE_MEMO_RECORDINGS_DIR."
            )

        if self.settings.archive_enabled and self.settings.archive_dir and not self.settings.archive_dir.exists():
            self.settings.archive_dir.mkdir(parents=True, exist_ok=True)
            LOGGER.info("Created archive directory at %s", self.settings.archive_dir, extra={"verbosity": 1})

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
        self._observer: Optional["Observer"] = None
        self._inbox_observer: Optional["Observer"] = None
        self.state = StateStore(self.settings.state_db)
        # These are dependency-injected so tests can monkeypatch `voicememowhisper.service.*`
        self._metadata_cache = MetadataCache(self.settings, loader=load_voice_memos)
        self.archive = ArchiveManager(self.settings, copy2=shutil.copy2)
        self.processor = MemoProcessor(
            settings=self.settings,
            transcriber=self.transcriber,
            archive=self.archive,
            state=self.state,
            metadata=self._metadata_cache,
        )
        self.inbox = InboxProcessor(
            self.settings,
            self.archive,
            self.enqueue_path,
            date_from_filename_func=_date_from_filename,
        )
        self._inflight: Set[str] = set()

        # Handle Inbox directory: auto-enable archive if inbox exists
        if self.settings.inbox_dir:
            try:
                if self.settings.inbox_dir.exists():
                    # Auto-enable archive when inbox is used
                    if not self.settings.archive_enabled:
                        self.settings = replace(self.settings, archive_enabled=True)
                        LOGGER.info("Auto-enabled archive mode for Inbox processing", extra={"verbosity": 1})
                    # Ensure archive directory exists
                    if self.settings.archive_dir and not self.settings.archive_dir.exists():
                        self.settings.archive_dir.mkdir(parents=True, exist_ok=True)
                        LOGGER.info("Created archive directory at %s", self.settings.archive_dir, extra={"verbosity": 1})
            except Exception as err:
                LOGGER.debug("Inbox directory check failed (non-fatal): %s", err, extra={"verbosity": 2})

    def start(self, watch: bool = False) -> None:
        """Start the worker thread and optionally the filesystem watcher."""
        LOGGER.info("Starting Voice Memo transcription service", extra={"verbosity": 1})
        self._log_sources()
        self._worker_thread = threading.Thread(target=self._worker_loop, name="VoiceMemoWorker", daemon=True)
        self._worker_thread.start()

        self.enqueue_existing()

        if watch:
            self._observer = start_watcher(self.settings.recordings_dir, self.enqueue_path)
            
            # Also watch Inbox directory if configured
            try:
                self._inbox_observer = self.inbox.start_watcher()
                if self._inbox_observer and self.settings.inbox_dir:
                    LOGGER.info("Watching Inbox directory: %s", self.settings.inbox_dir, extra={"verbosity": 1})
            except Exception as err:
                LOGGER.debug("Could not watch Inbox directory (non-fatal): %s", err, extra={"verbosity": 2})

    def process_one(self, path: Path) -> None:
        """Process a single audio file end-to-end through the same flow
        the main scan uses (state DB + ArchiveManager + speaker pipeline).

        ``path`` may point at a Voice Memos source file, an Inbox-style
        file, or an already-archived file. The path is normalized into a
        :class:`VoiceMemo` via the metadata cache so guid / title /
        created_at land consistently with the batch flow. The state DB
        guard inside :meth:`enqueue_path` skips the run if the file has
        already been transcribed *and* archived; pass a file that needs
        either step (or both) and it will run.
        """
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"audio file not found: {path}")

        LOGGER.info("Processing single audio: %s", path, extra={"verbosity": 0})
        self._log_sources()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, name="VoiceMemoWorker", daemon=True
        )
        self._worker_thread.start()
        self._metadata_cache.refresh()
        self.enqueue_path(path)

    def stop(self) -> None:
        LOGGER.info("Stopping Voice Memo transcription service", extra={"verbosity": 1})
        self._stop.set()
        if self._observer:
            self._observer.stop()
            self._observer.join()
        if self._inbox_observer:
            self._inbox_observer.stop()
            self._inbox_observer.join()
        self._queue.put(None)  # type: ignore[arg-type]
        if self._worker_thread:
            self._worker_thread.join()
        self.state.close()

    def enqueue_existing(self) -> None:
        self._metadata_cache.refresh()
        try:
            paths = list(self.settings.recordings_dir.glob("*.m4a"))
        except PermissionError as err:
            LOGGER.warning("Unable to read recordings directory: %s", err)
            return

        memos = []
        for path in paths:
            memo = self._memo_for_path(path)
            memos.append(memo)

        memos.sort(
            key=lambda memo: resolve_created_at(memo) or datetime.fromtimestamp(0),
            reverse=self.processing_order == "newest-first",
        )

        for memo in memos:
            self.enqueue_path(memo.path)

        # Scan and process Inbox directory
        self._scan_inbox()

        # Backfill any already-moved archive files that lack transcripts
        self._scan_archive_for_untranscribed()

    def enqueue_path(self, path: Path) -> None:
        guid = path.stem
        if guid in self._inflight:
            return

        # Check state to decide if we need to process
        transcript_path, archived_path = self.state.get_state(guid)
        needs_transcription = transcript_path is None

        # Treat files already under archive_dir as archived to avoid duplicate copies
        is_already_archived = False
        if self.settings.archive_dir:
            try:
                is_already_archived = path.resolve().is_relative_to(self.settings.archive_dir.resolve())
            except Exception:
                is_already_archived = False

        needs_archiving = self.settings.archive_enabled and archived_path is None and not is_already_archived

        if not needs_transcription and not needs_archiving:
            return

        memo = self._memo_for_path(path)
        LOGGER.debug("Enqueueing %s (Transcribe: %s, Archive: %s)", 
                     self._display_name(memo), needs_transcription, needs_archiving, extra={"verbosity": 2})
        self._queue.put(path)
        self._inflight.add(guid)

    def _refresh_metadata(self) -> None:
        # Backward-compatible wrapper (prefer MetadataCache directly).
        self._metadata_cache.refresh()

    def _display_name(self, memo: VoiceMemo) -> str:
        return self._metadata_cache.display_name(memo)

    def _log_sources(self) -> None:
        recordings_override = os.environ.get("VOICE_MEMO_RECORDINGS_DIR")
        container_override = os.environ.get("VOICE_MEMO_CONTAINER")
        if recordings_override:
            LOGGER.info("Recording source override (VOICE_MEMO_RECORDINGS_DIR): %s", self.settings.recordings_dir, extra={"verbosity": 1})
        elif container_override:
            LOGGER.info("Recording source override (VOICE_MEMO_CONTAINER): %s", self.settings.recordings_dir, extra={"verbosity": 1})
        else:
            LOGGER.info("Recording source (default): %s", self.settings.recordings_dir, extra={"verbosity": 1})

        transcript_override = os.environ.get("VOICE_MEMO_TRANSCRIPT_DIR")
        if transcript_override:
            LOGGER.info("Transcript output override (VOICE_MEMO_TRANSCRIPT_DIR): %s", self.settings.transcript_dir, extra={"verbosity": 1})
        else:
            LOGGER.info("Transcript output directory (default): %s", self.settings.transcript_dir, extra={"verbosity": 1})

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                break

            path = item
            memo = self._memo_for_path(path)
            guid = memo.guid
            try:
                self._process_memo(memo)
            except Exception:
                LOGGER.exception("Failed to process %s", self._display_name(memo))
            finally:
                self._queue.task_done()
                self._inflight.discard(guid)

    def _memo_for_path(self, path: Path) -> VoiceMemo:
        return self._metadata_cache.get_memo(path)

    def _transcript_filename(self, memo: VoiceMemo) -> str:
        return self.processor.transcript_filename(memo)

    def _process_memo(self, memo: VoiceMemo) -> None:
        self.processor.process(memo)

    def _archive_memo(self, memo: VoiceMemo, archive_filename: str) -> Optional[Path]:
        display = self._display_name(memo)
        # Create a per-call manager so tests can monkeypatch `voicememowhisper.service.shutil.copy2`.
        archive = ArchiveManager(self.settings, copy2=shutil.copy2)
        return archive.archive_copy(memo.path, archive_filename, display_name=display)

    def _process_inbox_file(self, path: Path) -> Optional[Path]:
        return self.inbox.process_file(path)

    def _scan_inbox(self) -> None:
        self.inbox.scan()

    def _scan_archive_for_untranscribed(self) -> None:
        """Enqueue files already in archive_dir that have no transcript recorded."""
        if not self.settings.archive_dir:
            return
        try:
            paths = list(self.settings.archive_dir.glob("*.m4a"))
        except PermissionError as err:
            LOGGER.debug("Cannot read archive directory: %s", err)
            return

        for path in paths:
            # If this archive file is already referenced by any processed memo, it should
            # not be treated as a new work item. This avoids reprocessing the same memo
            # just because the archive filename differs from the original GUID.
            try:
                if self.state.has_archived_path(path):
                    continue
            except AttributeError:
                # Backward compatible with older StateStore implementations.
                pass

            # Self-heal: exact path didn't match, but maybe the archive directory
            # was renamed / moved. Look up by basename; if there's exactly one
            # stale row, update its archived_path to the current location and
            # treat the file as already processed. Skip healing on basename
            # collision (rare; better to let the file fall through than to
            # rewrite the wrong row).
            try:
                matches = self.state.find_by_archived_basename(path.name)
            except AttributeError:
                matches = []
            if len(matches) == 1:
                stale_guid, stale_path = matches[0]
                if stale_path != path:
                    try:
                        self.state.update_archived_path(stale_guid, path)
                        LOGGER.info(
                            "Healed stale archived_path for %s: %s → %s",
                            stale_guid, stale_path, path,
                        )
                    except AttributeError:
                        pass
                continue
            elif len(matches) > 1:
                LOGGER.warning(
                    "Cannot self-heal %s: basename matches %d rows in state DB",
                    path.name, len(matches),
                )

            guid = path.stem
            transcript_path, _archived_path = self.state.get_state(guid)
            if transcript_path:
                continue

            # Self-heal: if a transcript with the same stem already exists on
            # disk, the file was previously transcribed but the state DB row
            # was lost (manual edit, corruption, restore from backup). Re-link
            # instead of re-running Whisper — important for archives whose
            # source memo was deleted from the Voice Memos app, since the
            # archive copy is now the only evidence and a fresh transcription
            # would produce a duplicate next to the existing transcript.
            existing_transcript = self.settings.transcript_dir / f"{path.stem}.txt"
            if existing_transcript.exists():
                memo = self._memo_for_path(path)
                created_at = resolve_created_at(memo)
                created_at_str = created_at.isoformat() if created_at else None
                self.state.mark_processed(
                    guid=guid,
                    transcript_path=existing_transcript,
                    archived_path=path,
                    title=memo.title,
                    duration=memo.duration_seconds,
                    created_at=created_at_str,
                )
                LOGGER.info(
                    "Re-linked existing transcript for %s (state DB row missing)",
                    path.name,
                )
                continue

            self.enqueue_path(path)

    def _handle_inbox_file(self, path: Path) -> None:
        """Handle a new file detected in Inbox directory."""
        self.inbox.handle_new_file(path)

    def join(self) -> None:
        self._queue.join()