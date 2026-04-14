from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional

from .config import Settings, load_settings

LOGGER = logging.getLogger("transcribe")


# WhisperKit CLI downloads models from the argmaxinc/whisperkit-coreml HF repo
# into `~/Documents/huggingface/models/...` by default. When that cache already
# has the requested model, we prefer to pass --model-path directly so WhisperKit
# does not reach out to huggingface.co for metadata verification at startup
# (which fails under transient network issues and surfaces as a misleading
# "Model not found" error).
_LOCAL_MODEL_CACHE_BASES: tuple[Path, ...] = (
    Path.home() / "Documents" / "huggingface" / "models" / "argmaxinc" / "whisperkit-coreml",
)
_LOCAL_MODEL_DIR_PREFIX = "openai_whisper-"


def _find_local_model_path(model_name: str) -> Optional[Path]:
    """Return a local WhisperKit model directory for ``model_name`` if present.

    Looks under the known cache bases for a directory named
    ``openai_whisper-<model_name>`` that actually contains files. Returns
    ``None`` when nothing usable is found so the caller can fall back to the
    CLI's network download path.
    """
    for base in _LOCAL_MODEL_CACHE_BASES:
        candidate = base / f"{_LOCAL_MODEL_DIR_PREFIX}{model_name}"
        try:
            if candidate.is_dir() and any(candidate.iterdir()):
                return candidate
        except OSError:
            continue
    return None


def _has_model_path_override(extra_args: Iterable[str]) -> bool:
    """Check whether the user already supplied --model-path via extra args."""
    for arg in extra_args:
        if arg == "--model-path" or arg.startswith("--model-path="):
            return True
    return False


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

        extra_args = tuple(self.settings.whisperkit_extra_args or ())
        user_supplied_model_path = _has_model_path_override(extra_args)

        cmd: list[str] = [self._cli, "transcribe"]

        model_source_note: str
        if user_supplied_model_path:
            # Respect whatever the user injected via VOICE_MEMO_WHISPERKIT_ARGS.
            model_source_note = "user-provided --model-path via VOICE_MEMO_WHISPERKIT_ARGS"
        else:
            local_model = _find_local_model_path(self.settings.whisperkit_model)
            if local_model is not None:
                cmd.extend(["--model-path", str(local_model)])
                model_source_note = f"local cache {local_model}"
            else:
                cmd.extend(["--model", self.settings.whisperkit_model])
                model_source_note = (
                    "CLI download fallback (local cache miss; requires network)"
                )

        cmd.extend(["--audio-path", str(audio_path)])

        if self.settings.language:
            cmd.extend(["--language", self.settings.language])

        if extra_args:
            cmd.extend(extra_args)

        # Minimal mode should show only the essential progress signal.
        LOGGER.info("Transcribing %s", display, extra={"verbosity": 0})
        # More details with -v / -vv.
        LOGGER.info(
            "WhisperKit model: %s (%s)",
            self.settings.whisperkit_model,
            model_source_note,
            extra={"verbosity": 1},
        )
        LOGGER.debug("WhisperKit cmd: %s", cmd, extra={"verbosity": 2})

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            LOGGER.error("WhisperKit CLI failed (%s): %s", result.returncode, result.stderr.strip())
            raise RuntimeError(
                f"WhisperKit CLI transcription failed for {display} "
                f"(exit code {result.returncode}). See logs for details."
            )

        text = result.stdout.strip()
        return text
