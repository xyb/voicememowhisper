from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from .archive import ArchiveManager, resolve_conflict_path
from .config import Settings
from .metadata import VoiceMemo, resolve_created_at
from .metadata_cache import MetadataCache
from .naming import sanitize_filename, title_from_stem
from .speaker_pipeline import SpeakerPipeline
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
    speaker_pipeline: Optional[SpeakerPipeline] = None

    def __post_init__(self) -> None:
        if self.speaker_pipeline is None and self.settings.speaker_pipeline_enabled:
            sp = SpeakerPipeline(self.settings)
            if sp.available():
                self.speaker_pipeline = sp
                LOGGER.info("Speaker pipeline enabled", extra={"verbosity": 1})
            else:
                LOGGER.info(
                    "Speaker pipeline not available, falling back to WhisperKit",
                    extra={"verbosity": 1},
                )

    # A `runs/<stem>/` older than this is treated as historical baggage —
    # not auto-retried even if `.md` is missing. Tight enough that
    # user-deleted Voice Memos with stale runs/ from old experiments don't
    # get reprocessed against the user's intent (legacy renamed transcripts
    # paired with stale runs/ from earlier experiments that never finished
    # render are the real-world false-positive case this guards against).
    _RECENT_RUN_WINDOW_SEC = 24 * 3600

    def _has_incomplete_speaker_run(self, transcript_path: Path) -> bool:
        """Return True iff `<runs_dir>/<stem>/` has step-1 ASR cache modified
        within the recent window but the corresponding `<stem>.md` is missing.

        The mtime fence: legacy WhisperKit-only transcripts have no
        `runs/<stem>/` dir at all, but old failed-pipeline experiments DO
        leave stale runs/ that would otherwise trigger unwanted retries on
        every `voicememowhisper` invocation. Only "ran in the last 24 h"
        qualifies for auto-retry; older runs/ are left alone — for manual
        retry the user can delete the `.txt` + state row and re-run."""
        import time
        try:
            stem = transcript_path.stem
            runs_dir = Path(self.settings.speaker_runs_dir) / stem
            if not runs_dir.is_dir():
                return False
            try:
                cache_files = list(runs_dir.glob("transcript_*.json"))
            except OSError:
                return False
            if not cache_files:
                return False
            try:
                youngest = max(p.stat().st_mtime for p in cache_files)
            except OSError:
                return False
            if (time.time() - youngest) > self._RECENT_RUN_WINDOW_SEC:
                return False
            md_path = self.settings.transcript_dir / f"{stem}.md"
            return not md_path.exists()
        except Exception as err:
            LOGGER.debug(
                "Speaker-run completeness check failed for %s: %s",
                transcript_path, err, extra={"verbosity": 2},
            )
            return False

    def transcript_filename(self, memo: VoiceMemo) -> str:
        ts = resolve_created_at(memo)
        ts_str = ts.strftime("%Y-%m-%d_%H-%M-%S") if ts else "undated"
        title = memo.title or title_from_stem(memo.path.stem)
        return f"{ts_str}_{sanitize_filename(title)}.txt"

    # Below this size an `.m4a` cannot contain any usable audio frames —
    # it's almost certainly a corrupted placeholder (empty file from a
    # crashed write, drag-drop accident, archive-from-deleted-source race)
    # rather than a recording that's still being captured. Real Voice
    # Memos m4a's are kilobytes minimum even for sub-second recordings.
    # Treating these as "skip" instead of "transcribe" prevents the HTTP
    # ASR backend's 500 (which is correct) from falling back to WhisperKit
    # (which will happily process garbage and emit a junk `.txt`).
    _MIN_VALID_M4A_SIZE = 1024  # 1 KiB

    def ensure_file_ready(self, path: Path, *, attempts: int = 3) -> bool:
        for _ in range(attempts):
            try:
                size = path.stat().st_size
                if size == 0:
                    raise OSError("File size is zero while recording may still be in progress.")
                if size < self._MIN_VALID_M4A_SIZE:
                    self._quarantine_corrupt(path, size)
                    return False
                return True
            except OSError as err:
                LOGGER.debug("Memo %s not ready (%s). Retrying...", path.name, err, extra={"verbosity": 2})
                time.sleep(1.0)
        return False

    def _quarantine_corrupt(self, path: Path, size: int) -> None:
        """Move a sub-threshold m4a out of the active scan path.

        Without this, every watcher restart re-scans the file, fails the
        size check, and emits the same WARNING + "Giving up" ERROR pair
        forever. Move it to ``<archive_dir>/_corrupt/`` so a human can
        inspect or delete it later, and the live workflow stops bumping
        into it. If no archive_dir is configured, fall back to the file's
        own ``_corrupt/`` sibling so we still get the file out of the way.
        """
        import shutil
        archive_dir = self.settings.archive_dir
        quarantine_dir = (archive_dir / "_corrupt") if archive_dir else (path.parent / "_corrupt")
        try:
            quarantine_dir.mkdir(parents=True, exist_ok=True)
            target = quarantine_dir / path.name
            if target.exists():
                target = resolve_conflict_path(target)
            shutil.move(str(path), str(target))
            LOGGER.warning(
                "Quarantined corrupt placeholder %s (%d bytes < %d) → %s",
                path.name, size, self._MIN_VALID_M4A_SIZE, target,
            )
        except OSError as err:
            LOGGER.error(
                "Failed to quarantine corrupt %s (%d bytes): %s — "
                "delete it manually to stop this error.",
                path.name, size, err,
            )

    def process(self, memo: VoiceMemo) -> None:
        path = memo.path
        display = self.metadata.display_name(memo)
        if not path.exists():
            LOGGER.warning("Skipping missing memo %s", display)
            return

        if not self.ensure_file_ready(path):
            # The corrupt-placeholder branch already moved the file aside
            # (quarantine warning logged); only emit the generic "giving
            # up" line for the genuine transient-readiness fall-through.
            if path.exists():
                LOGGER.error("Giving up on %s after repeated readiness checks", display)
            return

        # Refresh metadata and re-resolve to pick up title/trashed flags.
        self.metadata.refresh()
        memo = self.metadata.get_memo(path)
        display = self.metadata.display_name(memo)

        if memo.is_trashed:
            LOGGER.info("Skipping trashed memo %s", display, extra={"verbosity": 2})
            return

        transcript_path, archived_path = self.state.get_state(memo.guid)

        # If operating directly on an archive file, don't treat it as needing archiving.
        if archived_path is None and self.archive.is_path_archived(path):
            archived_path = path

        # Detect interrupted speaker-pipeline runs and force a re-run so a
        # plain `voicememowhisper` invocation finishes them from the cached
        # ASR JSON. Tight conditions to avoid touching the ~200 legacy
        # WhisperKit-only `.txt` files (those have no `runs/<stem>/` dir):
        #   - speaker pipeline is enabled AND a transcript is recorded on
        #     disk via state DB;
        #   - the target stem has a `runs/<stem>/` directory with at least
        #     one `transcript_*.json` (step 1 ASR cache from a past run);
        #   - the corresponding `<stem>.md` is missing in transcript_dir
        #     (step 5 render never landed → either the pipeline crashed
        #     mid-way or processor fell back to WhisperKit).
        # Clearing the state row's transcript_path makes the block below
        # take the transcription branch; speaker_pipeline.transcribe() then
        # picks up the ASR cache and only runs diarize → identify → merge
        # → render to produce the missing `.md`.
        if (
            transcript_path is not None
            and self.speaker_pipeline is not None
            and self.settings.speaker_pipeline_enabled
            and self._has_incomplete_speaker_run(transcript_path)
        ):
            LOGGER.info(
                "Detected incomplete speaker-pipeline run for %s; re-running from ASR cache",
                display, extra={"verbosity": 0},
            )
            try:
                self.state.clear_transcript_path(memo.guid)
            except Exception as err:
                LOGGER.warning(
                    "Failed to clear transcript_path for %s; skipping re-run: %s",
                    display, err,
                )
            else:
                transcript_path = None

        if transcript_path is None:
            filename = self.transcript_filename(memo)
            LOGGER.info("Memo title: %s", display, extra={"verbosity": 1})
            LOGGER.info("Transcript file: %s", filename, extra={"verbosity": 1})

            text = None
            if self.speaker_pipeline:
                # No silent fallback. If the configured speaker pipeline
                # is broken (server down, ws-funasr idle timeout, auth,
                # ...) we want the failure to surface — silently retrying
                # with WhisperKit produces a degraded .txt (no speaker
                # turns, no diarization) and hides the root cause, which
                # is the bug that bit a 90-min recording on 2026-05-24
                # (10-min funasr timeout → 30 min WhisperKit, user only
                # noticed when the meetlog had no speakers).
                #
                # The legitimate "use WhisperKit" path is when the
                # speaker pipeline is unavailable to begin with — that
                # branch lives in __post_init__ which leaves
                # self.speaker_pipeline = None and falls through to the
                # WhisperKit call below.
                LOGGER.info("Using speaker pipeline for %s", display, extra={"verbosity": 0})
                target_stem = Path(filename).stem
                text = self.speaker_pipeline.transcribe(
                    path, label=display, target_stem=target_stem,
                )
                target_path = self.settings.transcript_dir / filename
                if target_path.exists():
                    transcript_path = target_path

            if text is None:
                text = self.transcriber.transcribe(path, label=display)

            if transcript_path is None:
                output_path = self.settings.transcript_dir / filename
                LOGGER.info(
                    "Writing transcript to %s",
                    output_path.name,
                    extra={"verbosity": 0},
                )
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

