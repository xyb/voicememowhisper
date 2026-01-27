from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

try:  # pragma: no cover - exercised in integration environments
    from watchdog.events import FileSystemEvent, FileSystemEventHandler
    from watchdog.observers import Observer
except ModuleNotFoundError:  # pragma: no cover
    FileSystemEvent = object  # type: ignore[assignment]
    FileSystemEventHandler = object  # type: ignore[assignment]
    Observer = object  # type: ignore[assignment]

LOGGER = logging.getLogger("watcher")


class RecordingHandler(FileSystemEventHandler):
    """Dispatch events for new or updated recording files."""

    def __init__(self, callback: Callable[[Path], None], extensions: tuple[str, ...] = (".m4a",)) -> None:
        super().__init__()
        self._callback = callback
        self._extensions = tuple(ext.lower() for ext in extensions)

    def on_created(self, event: FileSystemEvent) -> None:  # pragma: no cover - relies on filesystem
        self._handle_event(event)

    def on_modified(self, event: FileSystemEvent) -> None:  # pragma: no cover - relies on filesystem
        self._handle_event(event)

    def on_moved(self, event: FileSystemEvent) -> None:  # pragma: no cover - relies on filesystem
        # Handle files moved into the watched directory (common when dragging/dropping).
        self._handle_event(event)

    def _handle_event(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        # For moved events prefer the destination path when available.
        path_str = getattr(event, "dest_path", None) or event.src_path
        path = Path(path_str)
        if path.suffix.lower() not in self._extensions:
            return
        self._callback(path)


def start_watcher(directory: Path, callback: Callable[[Path], None], extensions: tuple[str, ...] = (".m4a",)) -> Observer:
    """Start a watchdog observer for the given directory."""
    if Observer is object:  # type: ignore[comparison-overlap]  # pragma: no cover
        raise ModuleNotFoundError(
            "watchdog is required for --watch mode. Install it (e.g. `pip install watchdog`) "
            "or run without --watch."
        )
    observer = Observer()
    observer.schedule(RecordingHandler(callback, extensions), str(directory), recursive=False)
    observer.start()
    LOGGER.info("Watching %s for new recordings", directory)
    return observer
