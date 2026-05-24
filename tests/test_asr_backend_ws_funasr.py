"""Unit tests for the ``ws-funasr`` ASR backend.

This backend speaks the aliyun SpeechTranscriber WebSocket protocol that
funasr-aipod exposes at ``/ws/v1/asr``. It exists because the
synchronous OpenAI-Audio HTTP backend has a hard 600s client timeout
that bites long recordings — the WS path streams audio in PCM chunks
and the server pushes partial events back, so we can monitor real
progress with an idle-timer instead of a wall-clock deadline.

Tests stub ``websocket.create_connection`` and the PCM decoder so we
can drive the message sequence end-to-end without a live server or
ffmpeg.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from voicememowhisper.si.asr_backends import ws_funasr
from voicememowhisper.si.asr_backends.ws_funasr import (
    PROTOCOL,
    BACKEND_ID,
    IdleTimeoutError,
    TaskFailedError,
    WsFunasrConfig,
    _segments_from_sentences,
    _split_pcm_chunks,
    transcribe,
)


# ───────── fake websocket ──────────────────────────────────────────────


class _FakeWebSocket:
    """Replays a scripted server side over send()/recv().

    Each entry in ``recv_script`` is either:
      - a str/bytes to return on the next recv()
      - a callable that returns one
      - the sentinel ``_TIMEOUT`` to raise WebSocketTimeoutException
      - the sentinel ``_CLOSED`` to raise WebSocketConnectionClosedException
    """

    def __init__(self, recv_script: list[Any]) -> None:
        self.recv_script = deque(recv_script)
        self.sent: list[Any] = []
        self.closed = False
        self._timeout: float | None = None

    def settimeout(self, t: float) -> None:
        self._timeout = t

    def send(self, data: Any, opcode: int | None = None) -> None:
        self.sent.append((opcode, data))

    def recv(self) -> Any:
        if not self.recv_script:
            from websocket import WebSocketTimeoutException  # type: ignore
            raise WebSocketTimeoutException("recv_script exhausted")
        item = self.recv_script.popleft()
        if callable(item):
            item = item()
        if item is _TIMEOUT:
            from websocket import WebSocketTimeoutException  # type: ignore
            raise WebSocketTimeoutException("scripted timeout")
        if item is _CLOSED:
            from websocket import WebSocketConnectionClosedException  # type: ignore
            raise WebSocketConnectionClosedException("scripted close")
        return item

    def close(self) -> None:
        self.closed = True


class _Sentinel:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.name}>"


_TIMEOUT = _Sentinel("TIMEOUT")
_CLOSED = _Sentinel("CLOSED")


def _evt(name: str, payload: dict | None = None, task_id: str = "T1") -> str:
    """Build a server-event JSON string in the SpeechTranscriber shape."""
    return json.dumps({
        "header": {
            "message_id": f"m-{name}",
            "task_id": task_id,
            "namespace": "SpeechTranscriber",
            "name": name,
            "status": 20000000,
            "status_message": "OK",
        },
        "payload": payload or {},
    })


# ───────── pure helpers ────────────────────────────────────────────────


def test_split_pcm_chunks_exact_multiple() -> None:
    chunks = _split_pcm_chunks(b"x" * 38400, chunk_bytes=19200)
    assert len(chunks) == 2
    assert all(len(c) == 19200 for c in chunks)


def test_split_pcm_chunks_trailing_partial() -> None:
    chunks = _split_pcm_chunks(b"x" * (19200 + 1234), chunk_bytes=19200)
    assert len(chunks) == 2
    assert len(chunks[1]) == 1234


def test_segments_from_sentences_ms_to_sec() -> None:
    sentences = [(0, 6240, "今天天气不错"), (6240, 12000, "明天可能会下雨")]
    segs = _segments_from_sentences(sentences)
    assert len(segs) == 2
    assert segs[0].start == 0.0
    assert segs[0].end == pytest.approx(6.24, abs=1e-6)
    assert segs[0].text == "今天天气不错"
    assert segs[0].words is None
    assert segs[1].start == pytest.approx(6.24, abs=1e-6)


# ───────── transcribe() — message protocol shape ───────────────────────


def _stub_decode(monkeypatch, pcm: bytes) -> None:
    monkeypatch.setattr(
        ws_funasr, "_decode_to_pcm_bytes", lambda audio, sample_rate: pcm
    )


def _stub_ws(monkeypatch, recv_script: list[Any]) -> _FakeWebSocket:
    fake = _FakeWebSocket(recv_script)
    monkeypatch.setattr(
        ws_funasr.websocket,
        "create_connection",
        lambda url, **kwargs: fake,
    )
    return fake


def test_transcribe_happy_path_one_sentence(tmp_path, monkeypatch) -> None:
    """Cover the full Start → chunks → Stop → SentenceEnd → Completed flow."""
    audio = tmp_path / "clip.m4a"
    audio.write_bytes(b"FAKE")
    pcm = b"\x00\x00" * (16000 * 6)  # 6 seconds of silence
    _stub_decode(monkeypatch, pcm)

    # 6s of audio = 10 chunks of 0.6s. Reserve 12 TIMEOUTs so each
    # chunk's drain phase finds nothing, then SentenceEnd + Completed
    # arrive only after StopTranscription is sent.
    script = (
        [_evt("TranscriptionStarted")]
        + [_TIMEOUT] * 12
        + [
            _evt("SentenceEnd", {"index": 1, "begin_time": 0, "time": 6000, "result": "你好世界"}),
            _evt("TranscriptionCompleted"),
        ]
    )
    fake = _stub_ws(monkeypatch, script)

    cfg = WsFunasrConfig(url="wss://example/ws/v1/asr", idle_timeout_sec=2.0)
    transcript, raw = transcribe(audio, cfg)

    # message order on the wire: 1 JSON Start, N binary chunks, 1 JSON Stop.
    json_sends = [d for (op, d) in fake.sent if isinstance(d, str)]
    bin_sends = [d for (op, d) in fake.sent if isinstance(d, (bytes, bytearray))]
    assert len(json_sends) == 2
    first = json.loads(json_sends[0])
    last = json.loads(json_sends[1])
    assert first["header"]["name"] == "StartTranscription"
    assert first["payload"]["sample_rate"] == 16000
    assert first["payload"]["format"] == "pcm"
    assert last["header"]["name"] == "StopTranscription"
    # task_id reused on Stop
    assert last["header"]["task_id"] == first["header"]["task_id"]
    # chunks: 6s @ 16kHz s16le = 192000 bytes ÷ 19200 = 10 chunks
    assert len(bin_sends) == 10
    assert sum(len(c) for c in bin_sends) == len(pcm)

    # transcript shape
    assert transcript.backend == BACKEND_ID
    assert len(transcript.segments) == 1
    assert transcript.segments[0].text == "你好世界"
    assert transcript.segments[0].start == 0.0
    assert transcript.segments[0].end == 6.0
    assert raw["protocol"] == PROTOCOL
    assert raw["num_segments"] == 1
    assert fake.closed is True


def test_transcribe_collects_multiple_sentence_ends(tmp_path, monkeypatch) -> None:
    audio = tmp_path / "clip.m4a"; audio.write_bytes(b"FAKE")
    _stub_decode(monkeypatch, b"\x00\x00" * 16000)  # 1s = 2 chunks

    # 1s audio → 2 chunks; reserve 3 TIMEOUTs to cover the send-loop
    # drain calls, then events arrive after StopTranscription.
    script = (
        [_evt("TranscriptionStarted")]
        + [_TIMEOUT] * 3
        + [
            _evt("SentenceBegin", {"index": 1, "time": 0}),
            _evt("TranscriptionResultChanged", {"index": 1, "time": 200, "result": "今"}),
            _evt("SentenceEnd", {"index": 1, "begin_time": 0, "time": 500, "result": "今天"}),
            _evt("SentenceEnd", {"index": 2, "begin_time": 500, "time": 1000, "result": "天气好"}),
            _evt("TranscriptionCompleted"),
        ]
    )
    _stub_ws(monkeypatch, script)

    cfg = WsFunasrConfig(url="wss://example/ws/v1/asr", idle_timeout_sec=2.0)
    transcript, _ = transcribe(audio, cfg)

    assert [(s.start, s.end, s.text) for s in transcript.segments] == [
        (0.0, 0.5, "今天"),
        (0.5, 1.0, "天气好"),
    ]


def test_transcribe_task_failed_raises(tmp_path, monkeypatch) -> None:
    audio = tmp_path / "clip.m4a"; audio.write_bytes(b"FAKE")
    _stub_decode(monkeypatch, b"\x00\x00" * 1600)

    script = [
        _evt("TranscriptionStarted"),
        _TIMEOUT,
        # server reports fatal failure
        json.dumps({
            "header": {"namespace": "SpeechTranscriber", "name": "TaskFailed",
                       "status": 40000000, "status_text": "model OOM",
                       "task_id": "T1", "message_id": "m"}
        }),
    ]
    _stub_ws(monkeypatch, script)

    cfg = WsFunasrConfig(url="wss://example/ws/v1/asr", idle_timeout_sec=2.0)
    with pytest.raises(TaskFailedError, match="model OOM"):
        transcribe(audio, cfg)


def test_transcribe_idle_timeout_in_final_phase(tmp_path, monkeypatch) -> None:
    """No further events after Stop → raise IdleTimeoutError within idle window."""
    audio = tmp_path / "clip.m4a"; audio.write_bytes(b"FAKE")
    _stub_decode(monkeypatch, b"\x00\x00" * 1600)  # 0.1s, 1 chunk

    # Started, then nothing forever — must trip idle timeout
    script = [_evt("TranscriptionStarted")] + [_TIMEOUT] * 100
    _stub_ws(monkeypatch, script)

    cfg = WsFunasrConfig(
        url="wss://example/ws/v1/asr",
        idle_timeout_sec=0.2,  # short so test is fast
        send_drain_poll_sec=0.01,
    )
    t0 = time.time()
    with pytest.raises(IdleTimeoutError):
        transcribe(audio, cfg)
    # bounded by idle_timeout_sec + some headroom
    assert time.time() - t0 < 5.0


def test_transcribe_idle_timer_resets_on_each_event(tmp_path, monkeypatch) -> None:
    """Steady partials → must NOT trip the idle timer."""
    audio = tmp_path / "clip.m4a"; audio.write_bytes(b"FAKE")
    _stub_decode(monkeypatch, b"\x00\x00" * 1600)  # 0.1s, 1 chunk

    # Partials trickle in *during* chunk send (resetting idle timer
    # below the 0.5s threshold), then SentenceEnd + Completed land
    # only after StopTranscription.
    script = [_evt("TranscriptionStarted")]
    # 1s pcm = 2 chunks; spread partials + extra TIMEOUTs across them
    script += [_evt("TranscriptionResultChanged",
                    {"index": 1, "time": i * 100, "result": "你"}) for i in range(2)]
    script += [_TIMEOUT] * 4
    script += [
        _evt("SentenceEnd", {"index": 1, "begin_time": 0, "time": 500, "result": "你好"}),
        _evt("TranscriptionCompleted"),
    ]
    _stub_ws(monkeypatch, script)

    cfg = WsFunasrConfig(
        url="wss://example/ws/v1/asr",
        idle_timeout_sec=0.5,
        send_drain_poll_sec=0.01,
    )
    transcript, _ = transcribe(audio, cfg)
    assert transcript.segments[0].text == "你好"


def test_transcribe_skips_stop_when_server_completes_during_send(tmp_path, monkeypatch) -> None:
    """If server emits TranscriptionCompleted mid-send (e.g. aggressive
    VAD on a short clip), the client must NOT send a redundant
    StopTranscription afterwards — that would either confuse the server
    or trigger an idle-timeout waiting for a second Completed.
    """
    audio = tmp_path / "clip.m4a"; audio.write_bytes(b"FAKE")
    # 6s = 10 chunks; server returns Completed after chunk 2's drain.
    pcm = b"\x00\x00" * (16000 * 6)
    _stub_decode(monkeypatch, pcm)

    script = (
        [_evt("TranscriptionStarted")]
        + [_TIMEOUT]  # chunk 1: nothing
        + [
            # chunk 2's drain picks up everything in one go
            _evt("SentenceEnd", {"index": 1, "begin_time": 0, "time": 1000, "result": "短句"}),
            _evt("TranscriptionCompleted"),
        ]
    )
    fake = _stub_ws(monkeypatch, script)

    cfg = WsFunasrConfig(url="wss://example/ws/v1/asr", idle_timeout_sec=2.0)
    transcript, _ = transcribe(audio, cfg)

    # Only ONE JSON send: StartTranscription. No StopTranscription.
    json_sends = [d for (op, d) in fake.sent if isinstance(d, str)]
    assert len(json_sends) == 1
    assert json.loads(json_sends[0])["header"]["name"] == "StartTranscription"
    assert transcript.segments[0].text == "短句"


def test_transcribe_propagates_server_closed(tmp_path, monkeypatch) -> None:
    """If server closes before TranscriptionCompleted with no SentenceEnd, raise."""
    audio = tmp_path / "clip.m4a"; audio.write_bytes(b"FAKE")
    _stub_decode(monkeypatch, b"\x00\x00" * 1600)

    script = [_evt("TranscriptionStarted"), _TIMEOUT, _CLOSED]
    _stub_ws(monkeypatch, script)

    cfg = WsFunasrConfig(url="wss://example/ws/v1/asr", idle_timeout_sec=1.0)
    with pytest.raises(RuntimeError, match="closed"):
        transcribe(audio, cfg)
