from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class RecordingItem:
    """
    A recording aggregated across sources (App metadata, transcript files, archive files, state DB).
    """

    key: str  # GUID (preferred) or orphan stem
    created_at: datetime | None = None
    duration: float | None = None
    title: str | None = None
    transcript_path: Path | None = None
    archive_path: Path | None = None

    has_transcript: bool = False
    has_archive: bool = False
    has_source: bool = False

    def merge_flags_from(self, other: "RecordingItem") -> None:
        self.has_transcript = self.has_transcript or other.has_transcript
        self.has_archive = self.has_archive or other.has_archive
        self.has_source = self.has_source or other.has_source
        if self.transcript_path is None:
            self.transcript_path = other.transcript_path
        if self.archive_path is None:
            self.archive_path = other.archive_path

