from __future__ import annotations

from dataclasses import dataclass
import os
import shlex
from pathlib import Path
from typing import Tuple


def _env_path(key: str, default: Path) -> Path:
    raw = os.environ.get(key)
    return Path(raw).expanduser() if raw else default


def _env_args(key: str) -> Tuple[str, ...]:
    raw = os.environ.get(key)
    return tuple(shlex.split(raw)) if raw else ()


def _detect_default_paths() -> tuple[Path, Path, Path]:
    home = Path.home()
    candidate_roots = [
        home / "Library" / "Group Containers" / "group.com.apple.VoiceMemos.shared",
        home / "Library" / "Application Support" / "com.apple.voicememos",
        home
        / "Library"
        / "Containers"
        / "com.apple.VoiceMemos"
        / "Data"
        / "Library"
        / "Application Support"
        / "com.apple.voicememos",
    ]

    for root in candidate_roots:
        recordings = root / "Recordings"
        metadata = root / "Library" / "Application Support" / "Recents.sqlite"
        if recordings.exists():
            return root, recordings, metadata

    fallback_root = candidate_roots[0]
    return fallback_root, fallback_root / "Recordings", fallback_root / "Library" / "Application Support" / "Recents.sqlite"


DEFAULT_CONTAINER, DEFAULT_RECORDINGS, DEFAULT_METADATA = _detect_default_paths()


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for the transcription service."""

    container_root: Path = _env_path("VOICE_MEMO_CONTAINER", DEFAULT_CONTAINER)
    recordings_dir: Path = _env_path(
        "VOICE_MEMO_RECORDINGS_DIR",
        DEFAULT_RECORDINGS
        if "VOICE_MEMO_CONTAINER" not in os.environ
        else (container_root / "Recordings"),
    )
    metadata_db: Path = _env_path(
        "VOICE_MEMO_METADATA_DB",
        DEFAULT_METADATA
        if "VOICE_MEMO_CONTAINER" not in os.environ
        else (container_root / "Recents.sqlite"),
    )
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
