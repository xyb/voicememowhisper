from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .config import Settings, load_settings

LOGGER = logging.getLogger("transcribe")


class WhisperTranscriber:
    """Transcribe audio files using the WhisperKit CLI."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or load_settings()
        self._cli = self._resolve_cli_binary(self.settings.whisperkit_cli)

    @staticmethod
    def _resolve_cli_binary(binary: str) -> str:
        path = shutil.which(binary)
        if path:
            return path
        candidate = Path(binary).expanduser()
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(
            f"Unable to locate WhisperKit CLI executable '{binary}'. "
            "Install via Homebrew (`brew install whisperkit-cli`) or set VOICE_MEMO_WHISPERKIT_CLI."
        )

    def transcribe(self, audio_path: Path, *, label: str | None = None) -> str:
        display = (label or audio_path.stem or audio_path.name).strip()
        if not display:
            display = audio_path.stem or audio_path.name

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {display}")

        cmd = [
            self._cli,
            "transcribe",
            "--model",
            self.settings.whisperkit_model,
            "--audio-path",
            str(audio_path),
        ]

        if self.settings.language:
            cmd.extend(["--language", self.settings.language])

        if self.settings.whisperkit_extra_args:
            cmd.extend(self.settings.whisperkit_extra_args)

        LOGGER.info("Transcribing %s with WhisperKit (%s)", display, self.settings.whisperkit_model)

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            LOGGER.error("WhisperKit CLI failed (%s): %s", result.returncode, result.stderr.strip())
            raise RuntimeError(
                f"WhisperKit CLI transcription failed for {display} "
                f"(exit code {result.returncode}). See logs for details."
            )

        text = result.stdout.strip()
        return text
