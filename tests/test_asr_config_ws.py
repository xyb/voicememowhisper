"""Tests for the ws-funasr branch of TOML config loading + pipeline kwargs.

Sibling of ``test_asr_config.py``; isolated here so the new section
doesn't bloat the existing file.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

from pathlib import Path

import pytest

from voicememowhisper.si.asr_config import build_pipeline_kwargs, load_asr_config


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_load_parses_ws_section(tmp_path) -> None:
    cfg = _write(
        tmp_path / "config.toml",
        """
[asr]
backend = "ws-funasr"

[asr.ws]
url = "wss://example/ws/v1/asr"
idle_timeout_sec = 90
connect_timeout_sec = 20
sample_rate = 16000
""",
    )
    out = load_asr_config(str(cfg))
    assert out["asr_backend"] == "ws-funasr"
    assert out["asr_ws_url"] == "wss://example/ws/v1/asr"
    assert out["asr_ws_idle_timeout_sec"] == 90
    assert out["asr_ws_connect_timeout_sec"] == 20
    assert out["asr_ws_sample_rate"] == 16000


def test_load_ws_unknown_key_ignored_silently(tmp_path) -> None:
    cfg = _write(
        tmp_path / "config.toml",
        """
[asr]
backend = "ws-funasr"

[asr.ws]
url = "wss://example/ws/v1/asr"
mystery_key = "this should be ignored"
""",
    )
    out = load_asr_config(str(cfg))
    assert "mystery_key" not in out
    assert out["asr_ws_url"] == "wss://example/ws/v1/asr"


def test_build_kwargs_ws_backend_emits_asr_backend_config() -> None:
    kwargs = build_pipeline_kwargs(
        {
            "asr_backend": "ws-funasr",
            "asr_ws_url": "wss://example/ws/v1/asr",
            "asr_ws_idle_timeout_sec": 90.0,
        }
    )
    assert kwargs["asr_backend"] == "ws-funasr"
    assert kwargs["asr_backend_config"] == {
        "url": "wss://example/ws/v1/asr",
        "idle_timeout_sec": 90.0,
    }


def test_build_kwargs_ws_underscore_alias_also_works() -> None:
    kwargs = build_pipeline_kwargs(
        {"asr_backend": "ws_funasr", "asr_ws_url": "wss://example/ws"}
    )
    assert kwargs["asr_backend"] == "ws_funasr"
    assert kwargs["asr_backend_config"] == {"url": "wss://example/ws"}


def test_build_kwargs_openai_audio_backend_unchanged() -> None:
    """Regression: ws-funasr support must not break the openai-audio path."""
    kwargs = build_pipeline_kwargs(
        {
            "asr_backend": "openai-audio",
            "asr_url": "http://example/v1/audio/transcriptions",
            "asr_model": "paraformer-large",
        }
    )
    assert kwargs["asr_backend"] == "openai-audio"
    assert kwargs["asr_backend_config"] == {
        "url": "http://example/v1/audio/transcriptions",
        "model": "paraformer-large",
    }
    # ws keys must not leak into the openai-audio config
    assert "idle_timeout_sec" not in kwargs["asr_backend_config"]


def test_build_kwargs_default_backend_stays_faster_whisper() -> None:
    """Empty merged dict → faster_whisper default unchanged.

    'ws-funasr default' is opt-in via config or --asr-backend, never
    forced — keeps behaviour stable for users without a configured
    WS server.
    """
    kwargs = build_pipeline_kwargs({})
    assert kwargs["asr_backend"] == "faster_whisper"
    assert "asr_backend_config" not in kwargs
