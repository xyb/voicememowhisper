# pyright: reportMissingImports=false

from __future__ import annotations

from pathlib import Path

import pytest

from voicememowhisper import config
from voicememowhisper.config import parse_processing_order
from voicememowhisper.paths import require_accessible_path


def test_parse_processing_order_aliases() -> None:
    assert parse_processing_order("newest") == "newest-first"
    assert parse_processing_order("DESC") == "newest-first"
    assert parse_processing_order("oldest") == "oldest-first"
    assert parse_processing_order("asc") == "oldest-first"


def test_parse_processing_order_invalid_raises() -> None:
    with pytest.raises(ValueError):
        parse_processing_order("wat")


def test_env_args_parses_shell_string(monkeypatch) -> None:
    monkeypatch.setenv("VOICE_MEMO_WHISPERKIT_ARGS", "--a 1 --b 'two words'")
    assert config._env_args("VOICE_MEMO_WHISPERKIT_ARGS") == ("--a", "1", "--b", "two words")  # type: ignore[attr-defined]


def test_safe_exists_returns_false_on_permission_error(monkeypatch, tmp_path: Path) -> None:
    p = tmp_path / "x"

    def boom(self):
        raise PermissionError("no")

    monkeypatch.setattr(Path, "exists", boom, raising=True)
    assert config._safe_exists(p) is False  # type: ignore[attr-defined]


def test_require_accessible_path_errors(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    with pytest.raises(FileNotFoundError):
        require_accessible_path(missing, "Recording directory")

    file_path = tmp_path / "file.txt"
    file_path.write_text("x")
    with pytest.raises(NotADirectoryError):
        require_accessible_path(file_path, "Recording directory")

