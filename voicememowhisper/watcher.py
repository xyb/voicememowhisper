from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

LOGGER = logging.getLogger("watcher")


class RecordingHandler(FileSystemEventHandler):
    """Dispatch events for new or updated recording files."""

    def __init__(self, callback: Callable[[Path], None]) -> None:
        super().__init__()
        self._callback = callback

    def on_created(self, event: FileSystemEvent) -> None:  # pragma: no cover - relies on filesystem
        self._handle_event(event)

    def on_modified(self, event: FileSystemEvent) -> None:  # pragma: no cover - relies on filesystem
        self._handle_event(event)

    def _handle_event(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() != ".m4a":
            return
        self._callback(path)


def start_watcher(directory: Path, callback: Callable[[Path], None]) -> Observer:
    """Start a watchdog observer for the given directory."""
    observer = Observer()
    observer.schedule(RecordingHandler(callback), str(directory), recursive=False)
    observer.start()
    LOGGER.info("Watching %s for new recordings", directory)
    return observer
