from __future__ import annotations

from pathlib import Path

from .config import Settings, load_settings


def ensure_directories(settings: Settings | None = None) -> Settings:
    """Ensure output directories exist and return settings."""
    settings = settings or load_settings()

    settings.transcript_dir.mkdir(parents=True, exist_ok=True)
    settings.state_db.parent.mkdir(parents=True, exist_ok=True)

    return settings


def require_accessible_path(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found at {path}")
    if not path.is_dir() and description.lower().endswith("directory"):
        raise NotADirectoryError(f"{description} expected to be a directory at {path}")
    return path

