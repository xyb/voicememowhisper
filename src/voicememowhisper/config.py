from __future__ import annotations

from dataclasses import dataclass
import os
import shlex
from pathlib import Path
from typing import Tuple

DEFAULT_CONTAINER = Path.home() / "Library" / "Group Containers" / "group.com.apple.VoiceMemos.shared" / "Library" / "Application Support"


def _env_path(key: str, default: Path) -> Path:
    raw = os.environ.get(key)
    return Path(raw).expanduser() if raw else default


def _env_args(key: str) -> Tuple[str, ...]:
    raw = os.environ.get(key)
    return tuple(shlex.split(raw)) if raw else ()


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for the transcription service."""

    container_root: Path = _env_path("VOICE_MEMO_CONTAINER", DEFAULT_CONTAINER)
    recordings_dir: Path = _env_path("VOICE_MEMO_RECORDINGS_DIR", container_root / "Recordings")
    metadata_db: Path = _env_path("VOICE_MEMO_METADATA_DB", container_root / "Recents.sqlite")
    transcript_dir: Path = _env_path(
        "VOICE_MEMO_TRANSCRIPT_DIR", Path.home() / "Documents" / "VoiceMemoTranscripts"
    )
    state_db: Path = _env_path("VOICE_MEMO_STATE_DB", Path.home() / ".voice-memo-whisper" / "state.sqlite")
    whisperkit_cli: str = os.environ.get("VOICE_MEMO_WHISPERKIT_CLI", "whisperkit-cli")
    whisperkit_model: str = os.environ.get("VOICE_MEMO_WHISPERKIT_MODEL", "large-v3-v20240930_turbo")
    whisperkit_extra_args: Tuple[str, ...] = _env_args("VOICE_MEMO_WHISPERKIT_ARGS")
    language: str | None = os.environ.get("VOICE_MEMO_LANGUAGE")


def load_settings() -> Settings:
    return Settings()

