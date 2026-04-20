"""Speaker-ID pipeline integration.

Calls the unified pipeline (experiments/speaker-id/pipeline.py) as a subprocess,
keeping ML dependencies isolated in the whisperx-lab venv.

Produces both:
  - transcript.md  (with speaker labels, timestamps, frontmatter)
  - transcript.txt  (plain text with [Speaker Name] prefixes)
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path

from .config import Settings

LOGGER = logging.getLogger("speaker_pipeline")


class SpeakerPipeline:
    """Transcribe audio with speaker diarization and identification."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._python = settings.speaker_pipeline_python
        self._pipeline_dir = settings.speaker_pipeline_dir
        self._pipeline_script = self._pipeline_dir / "pipeline.py"
        self._library_dir = self._pipeline_dir / "speaker-library"

    def available(self) -> bool:
        return (
            Path(self._python).exists()
            and self._pipeline_script.exists()
            and self._library_dir.exists()
        )

    def transcribe(self, audio_path: Path, *, label: str | None = None) -> str:
        """Run the speaker-id pipeline. Returns the plain-text transcript.

        Side effect: also writes transcript.md to the transcript directory.
        Streams pipeline stdout/stderr line-by-line to LOGGER so stage
        progress is visible in real time (no more silent multi-minute waits).
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

        cmd = [
            self._python,
            str(self._pipeline_script),
            str(audio_path),
            "--model", self.settings.speaker_pipeline_model,
            "--language", self.settings.language or "zh",
            "--threshold", str(self.settings.speaker_pipeline_threshold),
            "--library", str(self._library_dir),
            "--output-transcript", str(self.settings.transcript_dir),
        ]

        LOGGER.debug("Speaker pipeline cmd: %s", cmd, extra={"verbosity": 2})

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        t0 = time.monotonic()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(self._pipeline_dir),
            env=env,
        )

        last_lines: list[str] = []
        try:
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip()
                if not line:
                    continue
                LOGGER.info("[pipeline] %s", line, extra={"verbosity": 0})
                last_lines.append(line)
                if len(last_lines) > 50:
                    last_lines = last_lines[-50:]
        finally:
            returncode = proc.wait()

        elapsed = time.monotonic() - t0

        if returncode != 0:
            tail = "\n".join(last_lines[-20:]) or "(no output)"
            LOGGER.error(
                "Speaker pipeline failed (exit=%s, elapsed=%.1fs):\n%s",
                returncode, elapsed, tail,
            )
            raise RuntimeError(
                f"Speaker pipeline failed for {display} "
                f"(exit code {returncode}). See logs for details."
            )

        LOGGER.info(
            "Speaker pipeline finished: %s in %.1fs",
            display, elapsed,
            extra={"verbosity": 0},
        )

        # Read back the plain-text transcript for backward compatibility
        recording_id = audio_path.stem
        txt_path = self.settings.transcript_dir / f"{recording_id}.txt"
        if txt_path.exists():
            return txt_path.read_text(encoding="utf-8").strip()

        # Fallback: read from pipeline outputs
        outputs_txt = self._pipeline_dir / "outputs" / recording_id / "transcript.txt"
        if outputs_txt.exists():
            return outputs_txt.read_text(encoding="utf-8").strip()

        LOGGER.warning("No plain-text transcript found for %s", recording_id)
        return ""
