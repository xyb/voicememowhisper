from __future__ import annotations

import logging
import queue
import shutil
import threading
import time
import os
import re
import hashlib
from datetime import datetime
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

LOGGER = logging.getLogger("service")


def _sanitize_filename(value: str) -> str:
    safe_chars = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_", " "):
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    return "".join(safe_chars).strip() or "untitled"


def _hash_file(path: Path, chunk_size: int = 1024 * 1024) -> Optional[str]:
    """Return hex digest of file content or None on error."""
    sha = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(chunk_size)
                if not chunk:
                    break
                sha.update(chunk)
    except OSError as err:
        LOGGER.debug("Failed to hash %s: %s", path, err)
        return None
    return sha.hexdigest()


def _date_from_filename(path: Path) -> Optional[datetime]:
    """Extract a leading YYYY-MM-DD from the filename stem if present."""
    match = re.match(r"^(\d{4}-\d{2}-\d{2})", path.stem)
    if not match:
        return None
    try:
        parsed_date = datetime.strptime(match.group(1), "%Y-%m-%d")
    except ValueError:
        return None

    # Preserve the time component from the file's current mtime if available.
    try:
        mtime = path.stat().st_mtime
        current_dt = datetime.fromtimestamp(mtime)
        parsed_date = parsed_date.replace(
            hour=current_dt.hour, minute=current_dt.minute, second=current_dt.second
        )
    except OSError:
        pass
    return parsed_date


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
            LOGGER.info("Created archive directory at %s", self.settings.archive_dir)

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
        self._inbox_observer: Optional[Observer] = None
        self.state = StateStore(self.settings.state_db)
        self._metadata: dict[str, VoiceMemo] = {}
        self._inflight: Set[str] = set()

        # Handle Inbox directory: auto-enable archive if inbox exists
        if self.settings.inbox_dir:
            try:
                if self.settings.inbox_dir.exists():
                    # Auto-enable archive when inbox is used
                    if not self.settings.archive_enabled:
                        self.settings = replace(self.settings, archive_enabled=True)
                        LOGGER.info("Auto-enabled archive mode for Inbox processing")
                    # Ensure archive directory exists
                    if self.settings.archive_dir and not self.settings.archive_dir.exists():
                        self.settings.archive_dir.mkdir(parents=True, exist_ok=True)
                        LOGGER.info("Created archive directory at %s", self.settings.archive_dir)
            except Exception as err:
                LOGGER.debug("Inbox directory check failed (non-fatal): %s", err)

    def start(self, watch: bool = False) -> None:
        """Start the worker thread and optionally the filesystem watcher."""
        LOGGER.info("Starting Voice Memo transcription service")
        self._log_sources()
        self._worker_thread = threading.Thread(target=self._worker_loop, name="VoiceMemoWorker", daemon=True)
        self._worker_thread.start()

        self.enqueue_existing()

        if watch:
            self._observer = start_watcher(self.settings.recordings_dir, self.enqueue_path)
            
            # Also watch Inbox directory if configured
            if self.settings.inbox_dir:
                try:
                    if self.settings.inbox_dir.exists():
                        # Watch for multiple audio formats in Inbox
                        inbox_extensions = (".m4a", ".mp3", ".wav", ".m4v", ".aac")
                        self._inbox_observer = start_watcher(
                            self.settings.inbox_dir, 
                            self._handle_inbox_file,
                            extensions=inbox_extensions
                        )
                        LOGGER.info("Watching Inbox directory: %s", self.settings.inbox_dir)
                except Exception as err:
                    LOGGER.debug("Could not watch Inbox directory (non-fatal): %s", err)

    def stop(self) -> None:
        LOGGER.info("Stopping Voice Memo transcription service")
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
        self._refresh_metadata()
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
                     self._display_name(memo), needs_transcription, needs_archiving)
        self._queue.put(path)
        self._inflight.add(guid)

    def _refresh_metadata(self) -> None:
        try:
            self._metadata = load_voice_memos(self.settings)
        except PermissionError as err:
            LOGGER.warning("Metadata access denied: %s", err)
            self._metadata = {}

    def _display_name(self, memo: VoiceMemo) -> str:
        if memo.title:
            title = memo.title.strip()
            if title:
                return title
        stem = memo.path.stem
        return stem or memo.guid

    def _log_sources(self) -> None:
        recordings_override = os.environ.get("VOICE_MEMO_RECORDINGS_DIR")
        container_override = os.environ.get("VOICE_MEMO_CONTAINER")
        if recordings_override:
            LOGGER.info("Recording source override (VOICE_MEMO_RECORDINGS_DIR): %s", self.settings.recordings_dir)
        elif container_override:
            LOGGER.info("Recording source override (VOICE_MEMO_CONTAINER): %s", self.settings.recordings_dir)
        else:
            LOGGER.info("Recording source (default): %s", self.settings.recordings_dir)

        transcript_override = os.environ.get("VOICE_MEMO_TRANSCRIPT_DIR")
        if transcript_override:
            LOGGER.info("Transcript output override (VOICE_MEMO_TRANSCRIPT_DIR): %s", self.settings.transcript_dir)
        else:
            LOGGER.info("Transcript output directory (default): %s", self.settings.transcript_dir)

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
        guid = path.stem
        memo = self._metadata.get(guid)
        if memo and memo.title:
            if memo.path != path:
                memo = replace(memo, path=path)
                self._metadata[guid] = memo
            return memo
        self._refresh_metadata()
        memo = self._metadata.get(guid)
        if memo:
            if memo.path != path:
                memo = replace(memo, path=path)
                self._metadata[guid] = memo
            return memo
        memo = VoiceMemo(guid=guid, path=path)
        self._metadata[guid] = memo
        return memo

    def _transcript_filename(self, memo: VoiceMemo) -> str:
        timestamp = resolve_created_at(memo)
        if timestamp is None:
            timestamp_str = "undated"
        else:
            timestamp_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        title = memo.title or memo.guid
        return f"{timestamp_str}_{_sanitize_filename(title)}.txt"

    def _process_memo(self, memo: VoiceMemo) -> None:
        path = memo.path
        display = self._display_name(memo)
        if not path.exists():
            LOGGER.warning("Skipping missing memo %s", display)
            return

        # Newly recorded files may still be written; retry a few times.
        for attempt in range(3):
            try:
                # Ensure file readable and non-empty
                if path.stat().st_size == 0:
                    raise OSError("File size is zero while recording may still be in progress.")
                break
            except OSError as err:
                LOGGER.debug("Memo %s not ready (%s). Retrying...", display, err)
                time.sleep(1.0)
        else:
            LOGGER.error("Giving up on %s after repeated readiness checks", display)
            return

        self._refresh_metadata()
        memo = self._memo_for_path(path)
        display = self._display_name(memo)

        if memo.is_trashed:
            LOGGER.info("Skipping trashed memo %s", display)
            return

        transcript_path, archived_path = self.state.get_state(memo.guid)

        # Inbox files are moved into archive_dir before processing; if we are already
        # operating on a file inside the archive directory, skip duplicate archiving.
        if archived_path is None and self.settings.archive_dir:
            try:
                if path.resolve().is_relative_to(self.settings.archive_dir.resolve()):
                    archived_path = path
            except Exception:
                pass

        # 1. Transcription
        if transcript_path is None:
            filename = self._transcript_filename(memo)
            LOGGER.info("Memo title: %s", display)
            LOGGER.info("Transcript file: %s", filename)

            text = self.transcriber.transcribe(path, label=display)
            
            output_path = self.settings.transcript_dir / filename
            LOGGER.info("Writing transcript for %s to %s", display, output_path.name)
            output_path.write_text(text + "\n", encoding="utf-8")
            transcript_path = output_path

        # 2. Archiving
        if self.settings.archive_enabled and archived_path is None:
            filename = self._transcript_filename(memo)
            archived_path = self._archive_memo(memo, filename)

        # Update State (only if we have at least a transcript, which we should)
        if transcript_path:
            created_at = resolve_created_at(memo)
            created_at_str = created_at.isoformat() if created_at else None
            
            self.state.mark_processed(
                guid=memo.guid,
                transcript_path=transcript_path,
                archived_path=archived_path,
                title=memo.title,
                duration=memo.duration_seconds,
                created_at=created_at_str
            )

    def _archive_memo(self, memo: VoiceMemo, transcript_filename: str) -> Optional[Path]:
        if not self.settings.archive_dir:
            return None

        # Derive archive filename from transcript filename but with .m4a extension
        archive_name = Path(transcript_filename).with_suffix(".m4a").name
        archive_path_base = self.settings.archive_dir / archive_name
        
        final_archive_path = archive_path_base
        counter = 1
        while final_archive_path.exists():
            final_archive_path = archive_path_base.with_stem(f"{archive_path_base.stem}_{counter}")
            counter += 1

        try:
            shutil.copy2(memo.path, final_archive_path)
            LOGGER.info("Archived %s to %s", self._display_name(memo), final_archive_path.name)
            return final_archive_path
        except OSError as err:
            LOGGER.error("Failed to archive %s: %s", self._display_name(memo), err)
            return None

    def _process_inbox_file(self, path: Path) -> Optional[Path]:
        """Move a file from Inbox to archive directory and return the new path."""
        if not self.settings.archive_dir:
            LOGGER.warning("Archive directory not configured, cannot process Inbox file %s", path.name)
            return None

        # Check if file is an audio file
        if path.suffix.lower() not in (".m4a", ".mp3", ".wav", ".m4v", ".aac"):
            LOGGER.debug("Skipping non-audio file in Inbox: %s", path.name)
            return None

        # Prefer date from filename prefix; fall back to ctime, then mtime
        timestamp = _date_from_filename(path)
        if timestamp is None:
            try:
                ts = path.stat().st_ctime
                timestamp = datetime.fromtimestamp(ts)
            except OSError:
                timestamp = None
        if timestamp is None:
            try:
                ts = path.stat().st_mtime
                timestamp = datetime.fromtimestamp(ts)
            except OSError:
                timestamp = None

        timestamp_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S") if timestamp else "undated"

        # Use filename stem as title
        title = _sanitize_filename(path.stem)
        archive_name = f"{timestamp_str}_{title}.m4a"
        archive_path_base = self.settings.archive_dir / archive_name

        # If a file with the same name already exists, compare hashes to detect duplicates.
        if archive_path_base.exists():
            existing_hash = _hash_file(archive_path_base)
            incoming_hash = _hash_file(path)
            if existing_hash and incoming_hash and existing_hash == incoming_hash:
                LOGGER.warning(
                    "Inbox file %s duplicates existing archive %s; discarding inbox copy",
                    path.name,
                    archive_path_base.name,
                )
                try:
                    path.unlink()
                except OSError as unlink_err:
                    LOGGER.debug("Failed to remove duplicate Inbox file %s: %s", path.name, unlink_err)
                return archive_path_base

        # Handle filename conflicts for genuinely different files
        final_archive_path = archive_path_base
        counter = 1
        while final_archive_path.exists():
            final_archive_path = archive_path_base.with_stem(f"{archive_path_base.stem}_{counter}")
            counter += 1

        try:
            # Move file to archive directory (this removes it from Inbox)
            shutil.move(str(path), str(final_archive_path))

            # Align file times to extracted timestamp so downstream uses correct date
            if timestamp:
                ts = timestamp.timestamp()
                os.utime(final_archive_path, (ts, ts))

            LOGGER.info("Moved Inbox file %s to %s", path.name, final_archive_path.name)
            return final_archive_path
        except OSError as err:
            LOGGER.error("Failed to move Inbox file %s to archive: %s", path.name, err)
            return None

    def _scan_inbox(self) -> None:
        """Scan Inbox directory for audio files and process them."""
        if not self.settings.inbox_dir:
            return

        try:
            if not self.settings.inbox_dir.exists():
                return
        except Exception:
            # Directory doesn't exist or permission error, silently skip
            return

        try:
            # Find all audio files in Inbox
            audio_extensions = (".m4a", ".mp3", ".wav", ".m4v", ".aac")
            paths = []
            for ext in audio_extensions:
                paths.extend(self.settings.inbox_dir.glob(f"*{ext}"))
                paths.extend(self.settings.inbox_dir.glob(f"*{ext.upper()}"))

            if not paths:
                return

            LOGGER.info("Found %d file(s) in Inbox", len(paths))

            # Process each file: move to archive and enqueue for transcription
            moved_paths = []
            for path in paths:
                moved_path = self._process_inbox_file(path)
                if moved_path:
                    moved_paths.append(moved_path)

            # Enqueue moved files for transcription
            for moved_path in moved_paths:
                self.enqueue_path(moved_path)

        except PermissionError as err:
            LOGGER.warning("Unable to read Inbox directory: %s", err)
        except Exception as err:
            LOGGER.warning("Error scanning Inbox directory: %s", err)

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
            guid = path.stem
            transcript_path, _archived_path = self.state.get_state(guid)
            if transcript_path:
                continue
            self.enqueue_path(path)

    def _handle_inbox_file(self, path: Path) -> None:
        """Handle a new file detected in Inbox directory."""
        moved_path = self._process_inbox_file(path)
        if moved_path:
            self.enqueue_path(moved_path)

    def join(self) -> None:
        self._queue.join()