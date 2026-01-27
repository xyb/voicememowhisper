from __future__ import annotations

import hashlib
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .config import Settings
from .metadata import VoiceMemo, resolve_created_at
from .naming import sanitize_filename

LOGGER = logging.getLogger("archive")


def hash_file(path: Path, chunk_size: int = 1024 * 1024) -> Optional[str]:
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


def resolve_conflict_path(base_path: Path) -> Path:
    """Return a non-existing path by appending _<n> to the stem."""
    final_path = base_path
    counter = 1
    while final_path.exists():
        final_path = base_path.with_stem(f"{base_path.stem}_{counter}")
        counter += 1
    return final_path


@dataclass(frozen=True)
class ArchiveManager:
    settings: Settings
    copy2: Callable[[str | Path, str | Path], object] = shutil.copy2

    def ensure_directory(self) -> None:
        if self.settings.archive_dir:
            self.settings.archive_dir.mkdir(parents=True, exist_ok=True)

    def is_path_archived(self, path: Path) -> bool:
        if not self.settings.archive_dir:
            return False
        try:
            return path.resolve().is_relative_to(self.settings.archive_dir.resolve())
        except Exception:
            return False

    def archive_filename(self, memo: VoiceMemo) -> str:
        """Generate archive filename with timestamp and sanitized title."""
        timestamp = resolve_created_at(memo)
        timestamp_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S") if timestamp else "undated"
        title = memo.title or memo.guid
        return f"{timestamp_str}_{sanitize_filename(title)}.m4a"

    def archive_copy(self, src: Path, archive_filename: str, *, display_name: str) -> Optional[Path]:
        """Copy a source file into archive directory under a conflict-free name."""
        if not self.settings.archive_dir:
            return None

        base = self.settings.archive_dir / archive_filename
        final = resolve_conflict_path(base)
        try:
            self.copy2(src, final)
            LOGGER.info("Archived %s to %s", display_name, final.name)
            return final
        except OSError as err:
            LOGGER.error("Failed to archive %s: %s", display_name, err)
            return None

