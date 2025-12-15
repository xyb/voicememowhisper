from __future__ import annotations

from dataclasses import dataclass
import os
import shlex
from pathlib import Path
from typing import Literal, Optional, Tuple


def _safe_exists(path: Path) -> bool:
    try:
        return path.exists()
    except PermissionError:
        return False


def _env_path(key: str, default: Path) -> Path:
    raw = os.environ.get(key)
    return Path(raw).expanduser() if raw else default


def _env_args(key: str) -> Tuple[str, ...]:
    raw = os.environ.get(key)
    return tuple(shlex.split(raw)) if raw else ()


DEFAULT_BASE_PATH = Path.home() / "Documents" / "VoiceMemoWhisper"
DEFAULT_ARCHIVE_PATH = DEFAULT_BASE_PATH / "Audio"
DEFAULT_TRANSCRIPT_PATH = DEFAULT_BASE_PATH / "Transcripts"

def _optional_env_path(key: str, default: Path | None) -> Path | None:
    raw = os.environ.get(key)
    if raw:
        return Path(raw).expanduser()
    return default


ProcessingOrder = Literal["newest-first", "oldest-first"]


def parse_processing_order(value: str | None, default: ProcessingOrder = "newest-first") -> ProcessingOrder:
    if not value:
        return default
    normalized = value.strip().lower()
    if normalized in ("newest-first", "newest", "recent-first", "desc"):
        return "newest-first"
    if normalized in ("oldest-first", "oldest", "asc"):
        return "oldest-first"
    raise ValueError("Invalid processing order. Use 'newest-first' or 'oldest-first'.")


def _detect_default_paths() -> tuple[Path, Path, Path, Optional[Path]]:
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
        cloud_db = root / "Recordings" / "CloudRecordings.db"
        recents_db = root / "Library" / "Application Support" / "Recents.sqlite"
        if _safe_exists(recordings):
            if _safe_exists(cloud_db):
                legacy = recents_db if _safe_exists(recents_db) else None
                return root, recordings, cloud_db, legacy
            if _safe_exists(recents_db):
                return root, recordings, recents_db, None
            return root, recordings, cloud_db, recents_db

    fallback_root = candidate_roots[0]
    fallback_cloud = fallback_root / "Recordings" / "CloudRecordings.db"
    metadata = fallback_cloud if _safe_exists(fallback_cloud) else fallback_root / "Library" / "Application Support" / "Recents.sqlite"
    legacy = (fallback_root / "Library" / "Application Support" / "Recents.sqlite") if metadata != fallback_root / "Library" / "Application Support" / "Recents.sqlite" else None
    return fallback_root, fallback_root / "Recordings", metadata, legacy


DEFAULT_CONTAINER, DEFAULT_RECORDINGS, DEFAULT_METADATA, DEFAULT_LEGACY_METADATA = _detect_default_paths()


def _default_recordings_dir() -> Path:
    container = os.environ.get("VOICE_MEMO_CONTAINER")
    if container:
        return Path(container).expanduser() / "Recordings"
    return DEFAULT_RECORDINGS


def _default_metadata_db() -> Path:
    container = os.environ.get("VOICE_MEMO_CONTAINER")
    if container:
        root = Path(container).expanduser()
        cloud = root / "Recordings" / "CloudRecordings.db"
        return cloud if _safe_exists(cloud) else root / "Library" / "Application Support" / "Recents.sqlite"
    return DEFAULT_METADATA


def _default_legacy_metadata_db() -> Optional[Path]:
    container = os.environ.get("VOICE_MEMO_CONTAINER")
    if container:
        root = Path(container).expanduser()
        cloud = root / "Recordings" / "CloudRecordings.db"
        recents = root / "Library" / "Application Support" / "Recents.sqlite"
        if _safe_exists(cloud) and _safe_exists(recents):
            return recents
        if _safe_exists(recents) and not _safe_exists(cloud):
            return None
        return None
    return DEFAULT_LEGACY_METADATA


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for the transcription service."""

    container_root: Path = _env_path("VOICE_MEMO_CONTAINER", DEFAULT_CONTAINER)
    recordings_dir: Path = _env_path("VOICE_MEMO_RECORDINGS_DIR", _default_recordings_dir())
    metadata_db: Path = _env_path("VOICE_MEMO_METADATA_DB", _default_metadata_db())
    legacy_metadata_db: Optional[Path] = _optional_env_path("VOICE_MEMO_LEGACY_METADATA_DB", _default_legacy_metadata_db())
    transcript_dir: Path = _env_path(
        "VOICE_MEMO_TRANSCRIPT_DIR", DEFAULT_TRANSCRIPT_PATH
    )
    archive_dir: Optional[Path] = _optional_env_path("VOICE_MEMO_ARCHIVE_DIR", DEFAULT_ARCHIVE_PATH)
    archive_enabled: bool = False
    state_db: Path = _env_path("VOICE_MEMO_STATE_DB", Path.home() / ".voice-memo-whisper" / "state.sqlite")
    whisperkit_cli: str = os.environ.get("VOICE_MEMO_WHISPERKIT_CLI", "whisperkit-cli")
    whisperkit_model: str = os.environ.get("VOICE_MEMO_WHISPERKIT_MODEL", "large-v3-v20240930_turbo")
    whisperkit_extra_args: Tuple[str, ...] = _env_args("VOICE_MEMO_WHISPERKIT_ARGS")
    language: str | None = os.environ.get("VOICE_MEMO_LANGUAGE")
    processing_order: ProcessingOrder = parse_processing_order(os.environ.get("VOICE_MEMO_PROCESSING_ORDER"))


def load_settings() -> Settings:
    return Settings()
