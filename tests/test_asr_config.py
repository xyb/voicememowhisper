"""Tests for TOML-based ASR backend config loading."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from voicememowhisper.si.asr_config import (
    build_pipeline_kwargs,
    find_config_file,
    load_asr_config,
)


def _write_config(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ───────── find_config_file ────────────────────────────────────────────


def test_find_config_returns_none_when_no_file_exists(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("VMW_CONFIG", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert find_config_file() is None


def test_find_config_uses_explicit_path(tmp_path) -> None:
    p = tmp_path / "mine.toml"
    p.write_text("[asr]\n")
    assert find_config_file(str(p)) == p


def test_find_config_explicit_missing_returns_none(tmp_path) -> None:
    # Explicit path that doesn't exist → None (caller decides what to do).
    assert find_config_file(tmp_path / "nope.toml") is None


def test_find_config_env_var_takes_priority(tmp_path, monkeypatch) -> None:
    env_path = tmp_path / "via_env.toml"
    env_path.write_text("[asr]\n")
    home_default = tmp_path / "config" / "voicememowhisper" / "config.toml"
    _write_config(home_default, "[asr]\n")

    monkeypatch.setenv("VMW_CONFIG", str(env_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert find_config_file() == env_path


def test_find_config_xdg_path_is_used_when_no_env(tmp_path, monkeypatch) -> None:
    xdg = tmp_path / ".config" / "voicememowhisper" / "config.toml"
    _write_config(xdg, "[asr]\n")
    monkeypatch.delenv("VMW_CONFIG", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert find_config_file() == xdg


def test_find_config_legacy_dot_file_fallback(tmp_path, monkeypatch) -> None:
    legacy = tmp_path / ".voicememowhisper.toml"
    legacy.write_text("[asr]\n")
    monkeypatch.delenv("VMW_CONFIG", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert find_config_file() == legacy


# ───────── load_asr_config ─────────────────────────────────────────────


def test_load_returns_empty_dict_when_no_config(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("VMW_CONFIG", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert load_asr_config() == {}


def test_load_parses_openai_audio_section(tmp_path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        """
        [asr]
        backend = "openai-audio"
        language = "zh"

        [asr.http]
        url = "http://asr.internal:8000/v1/audio/transcriptions"
        model = "paraformer-large"
        host_header = "asr.internal"
        timeout_sec = 120
        """,
        encoding="utf-8",
    )
    got = load_asr_config(cfg)
    assert got == {
        "asr_backend": "openai-audio",
        "language": "zh",
        "asr_url": "http://asr.internal:8000/v1/audio/transcriptions",
        "asr_model": "paraformer-large",
        "asr_host_header": "asr.internal",
        "asr_timeout_sec": 120,
    }


def test_load_only_asr_section_no_http(tmp_path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text('[asr]\nbackend = "faster_whisper"\nlanguage = "en"\n')
    assert load_asr_config(cfg) == {
        "asr_backend": "faster_whisper",
        "language": "en",
    }


def test_load_unknown_http_key_is_ignored(tmp_path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        """
        [asr]
        backend = "openai-audio"

        [asr.http]
        url = "http://x/v1"
        model = "m"
        typo_key = "will be ignored"
        """,
        encoding="utf-8",
    )
    got = load_asr_config(cfg)
    assert "typo_key" not in got
    assert got["asr_url"] == "http://x/v1"
    assert got["asr_model"] == "m"


def test_load_missing_file_returns_empty(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("VMW_CONFIG", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert load_asr_config(tmp_path / "nope.toml") == {}


def test_models_by_language_is_loaded_and_forwarded(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[asr]\n'
        'backend = "openai-audio"\n'
        'language = "zh"\n'
        '\n'
        '[asr.http]\n'
        'url = "http://x/v1/audio/transcriptions"\n'
        'model = "paraformer-large"\n'
        '\n'
        '[asr.http.models_by_language]\n'
        'zh = "paraformer-large"\n'
        'en = "sensevoice-small"\n'
    )
    merged = load_asr_config(cfg)
    assert merged["asr_models_by_language"] == {
        "zh": "paraformer-large",
        "en": "sensevoice-small",
    }
    kwargs = build_pipeline_kwargs(merged)
    assert kwargs["asr_backend_config"]["models_by_language"] == {
        "zh": "paraformer-large",
        "en": "sensevoice-small",
    }


def test_no_models_by_language_leaves_config_clean(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[asr]\n'
        'backend = "openai-audio"\n'
        '\n'
        '[asr.http]\n'
        'url = "http://x/v1/audio/transcriptions"\n'
        'model = "paraformer-large"\n'
    )
    merged = load_asr_config(cfg)
    assert "asr_models_by_language" not in merged
    kwargs = build_pipeline_kwargs(merged)
    assert "models_by_language" not in kwargs["asr_backend_config"]
