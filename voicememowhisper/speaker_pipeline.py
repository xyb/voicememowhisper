"""Speaker-ID pipeline integration into the watcher's transcribe path.

In-process wrapper around `voicememowhisper.si.pipeline.run`. Replaces
the previous subprocess-into-a-separate-venv design now that the ML
deps live in this project as the `[speaker-id]` extra.

Produces:
  - <recording>.md   (with speaker labels, timestamps, frontmatter)
  - <recording>.txt  (plain text with [Speaker Name] prefixes)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from .config import Settings

LOGGER = logging.getLogger("speaker_pipeline")


class SpeakerPipeline:
    """Transcribe audio with speaker diarization and identification."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._library_dir = Path(settings.speaker_library_dir)
        self._runs_dir = Path(settings.speaker_runs_dir)
        self._output_dir = Path(settings.speaker_output_dir)

    def available(self) -> bool:
        """Return True iff the [speaker-id] extra is installed AND a non-empty
        speaker library exists. Falls back to WhisperKit otherwise."""
        try:
            from . import si  # noqa: F401
            from .si import pipeline as _pl  # noqa: F401
        except ImportError as e:
            LOGGER.debug("speaker-id extra not installed: %s", e)
            return False
        if not self._library_dir.exists():
            return False
        # Has at least one speaker directory with an embedding
        for d in self._library_dir.iterdir():
            if d.is_dir() and (d / "embedding.npy").exists():
                return True
        return False

    def transcribe(self, audio_path: Path, *, label: str | None = None) -> str:
        """Run the speaker-id pipeline. Returns the plain-text transcript.

        Side effect: also writes transcript.md to the transcript directory.
        """
        display = label or audio_path.stem
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {display}")

        try:
            audio_mb = audio_path.stat().st_size / (1024 * 1024)
            LOGGER.info(
                "Speaker pipeline: %s (audio %.1f MB)",
                display, audio_mb,
                extra={"verbosity": 0},
            )
        except OSError:
            LOGGER.info("Speaker pipeline: %s", display, extra={"verbosity": 0})

        from .si.pipeline import run as run_pipeline

        t0 = time.monotonic()
        run_pipeline(
            audio_path,
            model=self.settings.speaker_pipeline_model,
            language=self.settings.language or "zh",
            library_dir=self._library_dir,
            threshold=self.settings.speaker_pipeline_threshold,
            runs_dir=self._runs_dir,
            output_dir=self._output_dir,
            output_transcript_dir=self.settings.transcript_dir,
        )
        elapsed = time.monotonic() - t0

        LOGGER.info(
            "Speaker pipeline finished: %s in %.1fs",
            display, elapsed,
            extra={"verbosity": 0},
        )

        # Plain-text copy (for backward compatibility with the legacy txt path).
        recording_id = audio_path.stem
        for candidate in (
            self.settings.transcript_dir / f"{recording_id}.txt",
            self._output_dir / recording_id / "transcript.txt",
        ):
            if candidate.exists():
                return candidate.read_text(encoding="utf-8").strip()

        LOGGER.warning("No plain-text transcript found for %s", recording_id)
        return ""
