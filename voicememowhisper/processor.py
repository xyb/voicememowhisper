from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from .archive import ArchiveManager
from .config import Settings
from .metadata import VoiceMemo, resolve_created_at
from .metadata_cache import MetadataCache
from .naming import sanitize_filename, title_from_stem
from .state import StateStore
from .transcribe import WhisperTranscriber

LOGGER = logging.getLogger("processor")


@dataclass
class MemoProcessor:
    settings: Settings
    transcriber: WhisperTranscriber
    archive: ArchiveManager
    state: StateStore
    metadata: MetadataCache

    def transcript_filename(self, memo: VoiceMemo) -> str:
        ts = resolve_created_at(memo)
        ts_str = ts.strftime("%Y-%m-%d_%H-%M-%S") if ts else "undated"
        title = memo.title or title_from_stem(memo.path.stem)
        return f"{ts_str}_{sanitize_filename(title)}.txt"

    def ensure_file_ready(self, path: Path, *, attempts: int = 3) -> bool:
        for _ in range(attempts):
            try:
                if path.stat().st_size == 0:
                    raise OSError("File size is zero while recording may still be in progress.")
                return True
            except OSError as err:
                LOGGER.debug("Memo %s not ready (%s). Retrying...", path.name, err)
                time.sleep(1.0)
        return False

    def process(self, memo: VoiceMemo) -> None:
        path = memo.path
        display = self.metadata.display_name(memo)
        if not path.exists():
            LOGGER.warning("Skipping missing memo %s", display)
            return

        if not self.ensure_file_ready(path):
            LOGGER.error("Giving up on %s after repeated readiness checks", display)
            return

        # Refresh metadata and re-resolve to pick up title/trashed flags.
        self.metadata.refresh()
        memo = self.metadata.get_memo(path)
        display = self.metadata.display_name(memo)

        if memo.is_trashed:
            LOGGER.info("Skipping trashed memo %s", display)
            return

        transcript_path, archived_path = self.state.get_state(memo.guid)

        # If operating directly on an archive file, don't treat it as needing archiving.
        if archived_path is None and self.archive.is_path_archived(path):
            archived_path = path

        if transcript_path is None:
            filename = self.transcript_filename(memo)
            LOGGER.info("Memo title: %s", display)
            LOGGER.info("Transcript file: %s", filename)

            text = self.transcriber.transcribe(path, label=display)
            output_path = self.settings.transcript_dir / filename
            LOGGER.info("Writing transcript for %s to %s", display, output_path.name)
            output_path.write_text(text + "\n", encoding="utf-8")
            transcript_path = output_path

        if self.settings.archive_enabled and archived_path is None:
            archive_name = self.archive.archive_filename(memo)
            archived_path = self.archive.archive_copy(path, archive_name, display_name=display)

        if transcript_path:
            created_at = resolve_created_at(memo)
            created_at_str = created_at.isoformat() if created_at else None
            self.state.mark_processed(
                guid=memo.guid,
                transcript_path=transcript_path,
                archived_path=archived_path,
                title=memo.title,
                duration=memo.duration_seconds,
                created_at=created_at_str,
            )

