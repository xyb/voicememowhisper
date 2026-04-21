"""Tests for the CLI's --asr-* flag → pipeline.run kwargs translation.

The full pipeline.run is heavy (imports pyannote / torch), so these
tests only exercise the thin argparse-to-dict helper.
"""

from __future__ import annotations

import argparse

import pytest

from voicememowhisper.si.cli import _build_asr_backend_kwargs


@pytest.fixture(autouse=True)
def _isolate_config(monkeypatch, tmp_path):
    """Make sure these tests never pick up a real config file on the
    developer's machine — the file-loading branch is covered separately
    in test_asr_config.py."""
    monkeypatch.delenv("VMW_CONFIG", raising=False)
    from pathlib import Path

    monkeypatch.setattr(Path, "home", lambda: tmp_path)


def _ns(**kw) -> argparse.Namespace:
    """Build a Namespace with every --asr-* flag defaulted to None."""
    defaults = dict(
        asr_backend=None,
        asr_config=None,
        asr_url=None,
        asr_model=None,
        asr_api_key=None,
        asr_host_header=None,
        asr_response_format=None,
        asr_timeout_sec=None,
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def test_default_is_faster_whisper_with_no_config() -> None:
    kw = _build_asr_backend_kwargs(_ns())
    assert kw == {"asr_backend": "faster_whisper"}
    assert "asr_backend_config" not in kw


def test_explicit_faster_whisper_also_has_no_config() -> None:
    kw = _build_asr_backend_kwargs(_ns(asr_backend="faster_whisper"))
    assert kw == {"asr_backend": "faster_whisper"}


def test_openai_audio_populates_config_from_flags() -> None:
    kw = _build_asr_backend_kwargs(
        _ns(
            asr_backend="openai-audio",
            asr_url="http://x/v1",
            asr_model="paraformer-large",
            asr_host_header="asr.example",
            asr_api_key="sk-test",
            asr_response_format="verbose_json",
            asr_timeout_sec=30.0,
        )
    )
    assert kw["asr_backend"] == "openai-audio"
    assert kw["asr_backend_config"] == {
        "url": "http://x/v1",
        "model": "paraformer-large",
        "host_header": "asr.example",
        "api_key": "sk-test",
        "response_format": "verbose_json",
        "timeout_sec": 30.0,
    }


def test_openai_audio_omits_missing_fields() -> None:
    """Unset flags (None) must NOT end up in the config — they would
    override the dataclass defaults with None and e.g. blow up
    timeout_sec which must be float."""
    kw = _build_asr_backend_kwargs(
        _ns(
            asr_backend="openai-audio",
            asr_url="http://x/v1",
            asr_model="paraformer-large",
        )
    )
    cfg = kw["asr_backend_config"]
    assert cfg == {"url": "http://x/v1", "model": "paraformer-large"}
    assert "api_key" not in cfg
    assert "timeout_sec" not in cfg


def test_openai_audio_underscore_alias_also_recognized() -> None:
    """Both 'openai-audio' and 'openai_audio' should produce a config."""
    kw = _build_asr_backend_kwargs(
        _ns(asr_backend="openai_audio", asr_url="http://x", asr_model="m")
    )
    assert kw["asr_backend"] == "openai_audio"
    assert kw["asr_backend_config"] == {"url": "http://x", "model": "m"}
