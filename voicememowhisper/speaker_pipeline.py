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
        speaker library exists. Logs the reason on False so "why did we fall
        back to WhisperKit?" is answerable from the log alone."""
        try:
            from . import si  # noqa: F401
            from .si import pipeline as _pl  # noqa: F401
        except ImportError as e:
            LOGGER.info(
                "Speaker pipeline unavailable: [speaker-id] extra not installed (%s). "
                "Install with `pip install -e \".[speaker-id]\"` to enable.",
                e, extra={"verbosity": 0},
            )
            return False
        if not self._library_dir.exists():
            LOGGER.info(
                "Speaker pipeline unavailable: speaker library dir does not exist (%s). "
                "Set VOICE_MEMO_SPEAKER_LIBRARY_DIR or create the dir with at least one "
                "enrolled speaker.",
                self._library_dir, extra={"verbosity": 0},
            )
            return False
        # Has at least one speaker directory with an embedding
        for d in self._library_dir.iterdir():
            if d.is_dir() and (d / "embedding.npy").exists():
                return True
        LOGGER.info(
            "Speaker pipeline unavailable: speaker library has no enrolled embeddings (%s). "
            "Run `voicememo-whisper si library add <speaker> <clip>` to enroll.",
            self._library_dir, extra={"verbosity": 0},
        )
        return False

    def transcribe(
        self,
        audio_path: Path,
        *,
        label: str | None = None,
        target_stem: str | None = None,
    ) -> str:
        """Run the speaker-id pipeline. Returns the plain-text transcript.

        Parameters
        ----------
        audio_path
            The Voice Memo audio file to process.
        label
            Human-readable name used in log lines.
        target_stem
            If given, used as the `recording_id` so that `runs/<target_stem>/`
            and `outputs/<target_stem>/` are named after the canonical
            `YYYY-MM-DD_HH-MM-SS_<title>` stem instead of the raw Voice Memos
            filename (`YYYYMMDD HHMMSS`). The copies of `transcript.md` /
            `transcript.txt` landing in `transcript_dir` are also renamed to
            this value so vault backlinks keep working.
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
        from .si.asr_config import load_asr_config, build_pipeline_kwargs

        # Load the same TOML config the `si run` CLI uses. Without this
        # the watcher-driven main flow would silently fall back to the
        # built-in defaults (faster_whisper + local_pyannote) and run
        # heavy ML locally, bypassing any self-hosted HTTP backends the
        # user configured. Precedence is TOML only — no CLI overlay
        # here; if you want per-run overrides, use `si run ... --asr-*`.
        backend_kwargs = build_pipeline_kwargs(load_asr_config())

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
            recording_id=target_stem,
            transcript_stem=target_stem,
            # The main-flow caller has its own ArchiveManager that moves
            # the source file into Audio/ with the canonical
            # `YYYY-MM-DD_HH-MM-SS_<title>.m4a` name. Letting the pipeline's
            # own auto-archive run here would race with that: pipeline moves
            # the source first using audio_path.name (the raw Voice Memos
            # `YYYYMMDD HHMMSS.m4a`), then the main-flow archive finds the
            # source missing and fails. The pipeline-side auto-archive
            # exists specifically for the direct `si run` path where no
            # other archiver is active.
            archive=False,
            **backend_kwargs,
        )
        elapsed = time.monotonic() - t0

        LOGGER.info(
            "Speaker pipeline finished: %s in %.1fs",
            display, elapsed,
            extra={"verbosity": 0},
        )

        # The pipeline now copies directly to `target_stem.{md,txt}` via
        # `transcript_stem=target_stem` above, so no post-run rename is
        # needed. We just need to find the resulting plain-text file.
        audio_stem = audio_path.stem
        final_txt = (
            self.settings.transcript_dir / f"{target_stem}.txt"
            if target_stem
            else self.settings.transcript_dir / f"{audio_stem}.txt"
        )
        for candidate in (
            final_txt,
            self._output_dir / (target_stem or audio_stem) / "transcript.txt",
        ):
            if candidate.exists():
                return candidate.read_text(encoding="utf-8").strip()

        LOGGER.warning("No plain-text transcript found for %s", audio_stem)
        return ""
