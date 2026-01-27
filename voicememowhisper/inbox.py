from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from watchdog.observers import Observer  # type: ignore

from .archive import ArchiveManager, hash_file, resolve_conflict_path
from .config import Settings
from .naming import sanitize_filename
from .watcher import start_watcher

LOGGER = logging.getLogger("inbox")

DEFAULT_INBOX_EXTENSIONS: tuple[str, ...] = (".m4a", ".mp3", ".wav", ".m4v", ".aac")


def date_from_filename(path: Path) -> Optional[datetime]:
    """Extract a leading YYYY-MM-DD from the filename stem if present."""
    import re

    match = re.match(r"^(\d{4}-\d{2}-\d{2})", path.stem)
    if not match:
        return None
    try:
        parsed_date = datetime.strptime(match.group(1), "%Y-%m-%d")
    except ValueError:
        return None

    try:
        mtime = path.stat().st_mtime
        current_dt = datetime.fromtimestamp(mtime)
        parsed_date = parsed_date.replace(
            hour=current_dt.hour, minute=current_dt.minute, second=current_dt.second
        )
    except OSError:
        pass
    return parsed_date


@dataclass
class InboxProcessor:
    settings: Settings
    archive: ArchiveManager
    enqueue_callback: Callable[[Path], None]
    date_from_filename_func: Callable[[Path], Optional[datetime]] = date_from_filename

    def is_audio_file(self, path: Path) -> bool:
        return path.suffix.lower() in DEFAULT_INBOX_EXTENSIONS

    def start_watcher(self) -> Optional[Observer]:
        if not self.settings.inbox_dir:
            return None
        try:
            if not self.settings.inbox_dir.exists():
                return None
        except Exception:
            return None
        return start_watcher(self.settings.inbox_dir, self.handle_new_file, extensions=DEFAULT_INBOX_EXTENSIONS)

    def scan(self) -> None:
        if not self.settings.inbox_dir:
            return
        try:
            if not self.settings.inbox_dir.exists():
                return
        except Exception:
            return

        try:
            paths: list[Path] = []
            for ext in DEFAULT_INBOX_EXTENSIONS:
                paths.extend(self.settings.inbox_dir.glob(f"*{ext}"))
                paths.extend(self.settings.inbox_dir.glob(f"*{ext.upper()}"))
            if not paths:
                return

            LOGGER.info("Found %d file(s) in Inbox", len(paths))

            moved_paths: list[Path] = []
            for path in paths:
                moved = self.process_file(path)
                if moved:
                    moved_paths.append(moved)

            for moved in moved_paths:
                self.enqueue_callback(moved)
        except PermissionError as err:
            LOGGER.warning("Unable to read Inbox directory: %s", err)
        except Exception as err:
            LOGGER.warning("Error scanning Inbox directory: %s", err)

    def handle_new_file(self, path: Path) -> None:
        moved = self.process_file(path)
        if moved:
            self.enqueue_callback(moved)

    def process_file(self, path: Path) -> Optional[Path]:
        """Move a file from Inbox to archive directory and return the new path."""
        if not self.archive.settings.archive_dir:
            LOGGER.warning("Archive directory not configured, cannot process Inbox file %s", path.name)
            return None

        if not self.is_audio_file(path):
            LOGGER.debug("Skipping non-audio file in Inbox: %s", path.name)
            return None

        timestamp = self.date_from_filename_func(path)
        if timestamp is None:
            try:
                timestamp = datetime.fromtimestamp(path.stat().st_ctime)
            except OSError:
                timestamp = None
        if timestamp is None:
            try:
                timestamp = datetime.fromtimestamp(path.stat().st_mtime)
            except OSError:
                timestamp = None

        timestamp_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S") if timestamp else "undated"
        title = sanitize_filename(path.stem)
        archive_name = f"{timestamp_str}_{title}{path.suffix.lower()}"
        archive_path_base = self.archive.settings.archive_dir / archive_name  # type: ignore[union-attr]

        if archive_path_base.exists():
            existing_hash = hash_file(archive_path_base)
            incoming_hash = hash_file(path)
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

        final_archive_path = resolve_conflict_path(archive_path_base)

        try:
            shutil.move(str(path), str(final_archive_path))
            if timestamp:
                ts = timestamp.timestamp()
                os.utime(final_archive_path, (ts, ts))
            LOGGER.info("Moved Inbox file %s to %s", path.name, final_archive_path.name)
            return final_archive_path
        except OSError as err:
            LOGGER.error("Failed to move Inbox file %s to archive: %s", path.name, err)
            return None

