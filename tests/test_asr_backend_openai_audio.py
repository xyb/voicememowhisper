"""Unit tests for the ``openai-audio`` ASR backend.

The goal is to cover request-construction and response-parsing in
isolation, without a live server. We stub ``urllib.request.urlopen``
at the module level and use synthetic audio bytes.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest import mock

import pytest

from voicememowhisper.si.asr_backends import openai_audio
from voicememowhisper.si.asr_backends.openai_audio import (
    OpenAIAudioConfig,
    _encode_multipart,
    _segments_from_openai_response,
    transcribe,
)


# ───────── multipart encoder ───────────────────────────────────────────


def test_encode_multipart_includes_text_fields_and_file(tmp_path: Path) -> None:
    audio = tmp_path / "sample.m4a"
    audio.write_bytes(b"FAKEAUDIO")
    body, content_type = _encode_multipart(
        {"model": "paraformer-large", "language": "zh"},
        "file",
        audio,
    )
    assert content_type.startswith("multipart/form-data; boundary=")
    decoded = body.decode("utf-8", errors="replace")
    assert 'name="model"' in decoded
    assert "paraformer-large" in decoded
    assert 'name="language"' in decoded
    assert "zh" in decoded
    assert 'name="file"; filename="sample.m4a"' in decoded
    assert "FAKEAUDIO" in decoded
    assert decoded.rstrip().endswith("--")


def test_encode_multipart_drops_none_fields(tmp_path: Path) -> None:
    audio = tmp_path / "sample.m4a"
    audio.write_bytes(b"X")
    body, _ = _encode_multipart(
        {"model": "whisper-1", "prompt": None},
        "file",
        audio,
    )
    assert b'name="prompt"' not in body


# ───────── response parser ─────────────────────────────────────────────


def test_segments_from_verbose_json_happy_path() -> None:
    payload = {
        "text": "hello world",
        "duration": 3.0,
        "language": "zh",
        "segments": [
            {"start": 0.0, "end": 1.5, "text": "hello"},
            {"start": 1.5, "end": 3.0, "text": "world"},
        ],
        "words": [
            {"start": 0.0, "end": 0.5, "word": "he", "probability": 0.9},
            {"start": 0.5, "end": 1.5, "word": "llo"},
            {"start": 1.5, "end": 3.0, "word": "world"},
        ],
    }
    segs = _segments_from_openai_response(payload)
    assert len(segs) == 2
    assert segs[0].text == "hello"
    assert segs[1].start == 1.5
    # Word-level timings get bucketed into the right segment.
    assert segs[0].words is not None and len(segs[0].words) == 2
    assert segs[1].words is not None and len(segs[1].words) == 1
    assert segs[0].words[0].probability == pytest.approx(0.9)


def test_segments_from_plain_json_fallback() -> None:
    """When the server returns ``{"text": ...}`` only, synthesize one segment."""
    payload = {"text": "only one line", "duration": 7.5}
    segs = _segments_from_openai_response(payload)
    assert len(segs) == 1
    assert segs[0].text == "only one line"
    assert segs[0].start == 0.0
    assert segs[0].end == pytest.approx(7.5)
    assert segs[0].words is None


def test_segments_empty_response_yields_nothing() -> None:
    assert _segments_from_openai_response({}) == []
    assert _segments_from_openai_response({"text": ""}) == []


# ───────── transcribe() end-to-end with mocked HTTP ────────────────────


class _FakeResponse:
    """Context-manager response object that mimics ``http.client.HTTPResponse``."""

    def __init__(self, status: int, payload: dict) -> None:
        self.status = status
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _make_audio(tmp_path: Path) -> Path:
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"\x00\x01\x02")
    return audio


def test_transcribe_builds_request_and_maps_response(tmp_path: Path) -> None:
    audio = _make_audio(tmp_path)
    cfg = OpenAIAudioConfig(
        url="http://asr.example/v1/audio/transcriptions",
        model="paraformer-large",
        api_key="sk-test",
        host_header="asr.example",
        language="zh",
    )
    fake_payload = {
        "text": "测试句子",
        "duration": 2.0,
        "language": "zh",
        "segments": [{"start": 0.0, "end": 2.0, "text": "测试句子"}],
    }

    captured: dict[str, object] = {}

    def _fake_urlopen(req, timeout):  # type: ignore[no-untyped-def]
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["method"] = req.get_method()
        captured["body_bytes"] = req.data
        captured["timeout"] = timeout
        return _FakeResponse(200, fake_payload)

    with mock.patch.object(openai_audio.urllib.request, "urlopen", _fake_urlopen):
        transcript, raw_info = transcribe(audio, cfg)

    # Request shape
    assert captured["url"] == cfg.url
    assert captured["method"] == "POST"
    headers = captured["headers"]
    assert headers.get("Authorization") == "Bearer sk-test"
    assert headers.get("Host") == "asr.example"
    assert headers.get("Content-type", "").startswith("multipart/form-data")
    assert b'name="model"' in captured["body_bytes"]
    assert b"paraformer-large" in captured["body_bytes"]
    assert b'name="language"' in captured["body_bytes"]
    # Response mapping
    assert transcript.backend == "openai_audio_http"
    assert transcript.model == "paraformer-large"
    assert transcript.language == "zh"
    assert transcript.duration_sec == pytest.approx(2.0)
    assert len(transcript.segments) == 1
    assert transcript.segments[0].text == "测试句子"
    # raw_info carries provenance
    assert raw_info["protocol"] == "openai-audio"
    assert raw_info["http_status"] == 200
    assert raw_info["num_segments"] == 1


def test_transcribe_no_auth_skips_authorization_header(tmp_path: Path) -> None:
    """Self-hosted FunASR slim needs no api_key and shouldn't get Authorization."""
    audio = _make_audio(tmp_path)
    cfg = OpenAIAudioConfig(
        url="http://lc-x3/v1/audio/transcriptions",
        model="paraformer-large",
        api_key=None,
        host_header="asr.example",
    )
    captured: dict[str, object] = {}

    def _fake_urlopen(req, timeout):  # type: ignore[no-untyped-def]
        captured["headers"] = dict(req.header_items())
        return _FakeResponse(200, {"text": "x", "duration": 1.0})

    with mock.patch.object(openai_audio.urllib.request, "urlopen", _fake_urlopen):
        transcribe(audio, cfg)

    assert "Authorization" not in captured["headers"]
    assert captured["headers"].get("Host") == "asr.example"


def test_transcribe_surfaces_http_error(tmp_path: Path) -> None:
    audio = _make_audio(tmp_path)
    cfg = OpenAIAudioConfig(url="http://asr.example/v1/audio/transcriptions",
                            model="m")

    import urllib.error

    err = urllib.error.HTTPError(
        url=cfg.url,
        code=500,
        msg="Internal Server Error",
        hdrs=None,  # type: ignore[arg-type]
        fp=io.BytesIO(b'{"detail":"boom"}'),
    )

    with mock.patch.object(openai_audio.urllib.request, "urlopen", side_effect=err):
        with pytest.raises(RuntimeError, match="HTTP 500"):
            transcribe(audio, cfg)


def test_transcribe_missing_audio_raises(tmp_path: Path) -> None:
    cfg = OpenAIAudioConfig(url="http://x/v1", model="m")
    with pytest.raises(FileNotFoundError):
        transcribe(tmp_path / "does-not-exist.m4a", cfg)


def test_transcribe_surfaces_non_json_response(tmp_path: Path) -> None:
    """When the server returns garbage (e.g. HTML error page), wrap the error."""
    audio = _make_audio(tmp_path)
    cfg = OpenAIAudioConfig(url="http://x/v1", model="m")

    class _HTMLResponse:
        status = 200

        def read(self):
            return b"<html>server down</html>"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

    with mock.patch.object(
        openai_audio.urllib.request, "urlopen", return_value=_HTMLResponse()
    ):
        with pytest.raises(RuntimeError, match="non-JSON response"):
            transcribe(audio, cfg)


def test_segments_drops_words_with_bad_timings() -> None:
    """Words missing start/end shouldn't crash; just skip them."""
    payload = {
        "duration": 2.0,
        "segments": [{"start": 0.0, "end": 2.0, "text": "x"}],
        "words": [
            {"start": 0.0, "end": 1.0, "word": "ok"},
            {"start": "invalid", "end": 1.5, "word": "skip-me"},
            {"word": "no-timings"},
        ],
    }
    segs = _segments_from_openai_response(payload)
    assert len(segs) == 1
    assert segs[0].words is not None and len(segs[0].words) == 1
    assert segs[0].words[0].text == "ok"


# ───────── CLI smoke tests ─────────────────────────────────────────────


def test_cli_requires_audio_and_url(capsys) -> None:
    from voicememowhisper.si.asr_backends.openai_audio import _cli

    with pytest.raises(SystemExit):
        _cli(["--model", "m"])  # missing --audio and --url


def test_cli_prints_summary_when_no_output_path(tmp_path: Path, capsys) -> None:
    from voicememowhisper.si.asr_backends.openai_audio import _cli

    audio = _make_audio(tmp_path)

    def _fake_urlopen(req, timeout):  # type: ignore[no-untyped-def]
        return _FakeResponse(
            200,
            {
                "text": "hi",
                "duration": 1.0,
                "language": "zh",
                "segments": [{"start": 0.0, "end": 1.0, "text": "hi"}],
            },
        )

    with mock.patch.object(openai_audio.urllib.request, "urlopen", _fake_urlopen):
        rc = _cli(
            [
                "--audio",
                str(audio),
                "--url",
                "http://x/v1",
                "--model",
                "m",
                "--language",
                "zh",
            ]
        )
    assert rc == 0
    out = capsys.readouterr().out
    assert "language  : zh" in out
    assert "segments  : 1" in out
    assert "[   0.0 →    1.0] hi" in out


def test_cli_writes_output_file(tmp_path: Path, capsys) -> None:
    from voicememowhisper.si.asr_backends.openai_audio import _cli

    audio = _make_audio(tmp_path)
    out_path = tmp_path / "t.json"

    def _fake_urlopen(req, timeout):  # type: ignore[no-untyped-def]
        return _FakeResponse(
            200,
            {"text": "x", "duration": 0.5, "segments": [{"start": 0, "end": 0.5, "text": "x"}]},
        )

    with mock.patch.object(openai_audio.urllib.request, "urlopen", _fake_urlopen):
        rc = _cli(
            [
                "--audio",
                str(audio),
                "--url",
                "http://x/v1",
                "--model",
                "m",
                "--output",
                str(out_path),
            ]
        )
    assert rc == 0
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["backend"] == "openai_audio_http"
    assert data["segments"][0]["text"] == "x"


def test_cli_returns_nonzero_on_error(tmp_path: Path, capsys) -> None:
    from voicememowhisper.si.asr_backends.openai_audio import _cli

    # Missing audio → transcribe raises FileNotFoundError → CLI prints + returns 2.
    rc = _cli(
        [
            "--audio",
            str(tmp_path / "nope.m4a"),
            "--url",
            "http://x/v1",
            "--model",
            "m",
        ]
    )
    assert rc == 2
    assert "FileNotFoundError" in capsys.readouterr().err
