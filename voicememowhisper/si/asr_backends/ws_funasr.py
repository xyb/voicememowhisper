"""ASR backend speaking the aliyun SpeechTranscriber WebSocket protocol.

This is the streaming-WebSocket sibling of ``openai_audio.py``. The two
backends target the same FunASR aipod server, but at different
endpoints:

- ``openai_audio.py`` → ``POST /v1/audio/transcriptions``: synchronous
  multipart upload, the client waits for one HTTP response that may
  take 10+ minutes for a 90-min recording. A hard wall-clock timeout
  is the only failure signal — if the server is healthy but slow, the
  client gives up anyway.

- this module → ``WSS /ws/v1/asr``: client streams 16 kHz PCM chunks,
  server pushes partial results back. We monitor real server activity
  with an **idle timer** (default 60s of silence = dead), so a long
  recording with steady progress runs indefinitely while a truly
  hung server fails fast.

Protocol (reverse-engineered from the funasr test page at
``/ws/v1/asr/test`` and verified end-to-end):

    1. client → ``StartTranscription`` (JSON header+payload)
       payload: format=pcm, sample_rate=16000, enable_intermediate_result, ...
    2. server → ``TranscriptionStarted`` (status=20000000)
    3. client → binary PCM chunks (raw s16le, mono, 16 kHz)
       suggested chunk: 9600 samples × 2 bytes = 19200 bytes ≈ 0.6 s
    4. server → ``SentenceBegin`` / ``TranscriptionResultChanged`` (partials)
                / ``SentenceEnd`` (final per-sentence, with begin_time + time + result)
    5. client → ``StopTranscription`` (JSON header)
    6. server → trailing ``SentenceEnd`` + ``TranscriptionCompleted``
    7. client closes

The ``SentenceEnd`` events are the source of truth for the transcript;
``TranscriptionResultChanged`` is the running tally we use only to keep
the idle timer happy.

No external Python deps beyond ``websocket-client`` (stdlib has no WS
client). Audio decoding shells out to ``ffmpeg`` — same dep posture as
the rest of the project. Falls under ``[speaker-id]`` optional install
since the rest of the speaker-id stack is already opt-in there.

Naming: the public API mirrors ``openai_audio.transcribe(audio, config)
→ (Transcript, raw_info)`` so the dispatch layer can swap one for the
other with no other changes.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import json
import logging
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import websocket  # websocket-client

from .. import contracts

LOGGER = logging.getLogger("ws_funasr")

# Protocol id used in configs / logs.
PROTOCOL = "ws-funasr"

# Backend id written into contracts.Transcript.backend. Distinct from
# PROTOCOL: backend = "how we got it" (websocket client), protocol =
# "what API shape the server speaks" (aliyun SpeechTranscriber).
BACKEND_ID = "ws_funasr"

# 9600 samples × 2 bytes (int16) = 19200 bytes = 0.6 s @ 16 kHz mono.
# Matches the chunk size the funasr test page uses (chunkStride=9600).
_DEFAULT_CHUNK_BYTES = 19200


class IdleTimeoutError(RuntimeError):
    """Server sent no event for ``idle_timeout_sec`` — assume dead."""


class TaskFailedError(RuntimeError):
    """Server sent a ``TaskFailed`` event (model error, bad input, ...)."""


@dataclass
class WsFunasrConfig:
    """All knobs for the ws-funasr client.

    ``idle_timeout_sec`` is the only liveness check — there is **no**
    overall wall-clock deadline. A 90-minute recording with steady
    server-side progress runs to completion; a truly stuck server fails
    after ``idle_timeout_sec`` of silence.
    """

    # Full ws/wss endpoint URL (e.g. wss://your-funasr-host/ws/v1/asr).
    url: str
    sample_rate: int = 16000
    idle_timeout_sec: float = 60.0
    connect_timeout_sec: float = 15.0
    enable_intermediate_result: bool = True
    enable_punctuation_prediction: bool = True
    enable_inverse_text_normalization: bool = True
    chunk_bytes: int = _DEFAULT_CHUNK_BYTES
    # Pacing between chunk sends. 0 = send as fast as the WS will take
    # them, let the server's intake buffer apply backpressure naturally.
    # Useful >0 only if a specific server can't handle bursts.
    chunk_send_interval_sec: float = 0.0
    # How long ``recv()`` blocks during the chunk-send drain phase
    # before we send the next chunk. Tiny so the send-loop stays
    # responsive; idle detection uses ``idle_timeout_sec`` directly.
    send_drain_poll_sec: float = 0.05


# ───────── pure helpers ────────────────────────────────────────────────


def _split_pcm_chunks(pcm: bytes, *, chunk_bytes: int) -> list[bytes]:
    """Slice a contiguous PCM blob into ``chunk_bytes``-sized chunks.

    Trailing partial chunk is kept (server tolerates short final).
    """
    if chunk_bytes <= 0:
        raise ValueError(f"chunk_bytes must be > 0, got {chunk_bytes}")
    return [pcm[i : i + chunk_bytes] for i in range(0, len(pcm), chunk_bytes)]


def _segments_from_sentences(
    sentences: list[tuple[int, int, str]],
) -> list[contracts.Segment]:
    """Convert collected (begin_ms, end_ms, text) tuples to Segments.

    aliyun protocol gives timestamps in milliseconds; ``contracts.Segment``
    uses seconds. No word-level timestamps are emitted by this protocol
    so ``words=None`` always.
    """
    out: list[contracts.Segment] = []
    for begin_ms, end_ms, text in sentences:
        out.append(
            contracts.Segment(
                start=float(begin_ms) / 1000.0,
                end=float(end_ms) / 1000.0,
                text=text,
                words=None,
            )
        )
    return out


def _decode_to_pcm_bytes(audio_path: Path, *, sample_rate: int) -> bytes:
    """ffmpeg subprocess: any audio → raw s16le mono PCM at sample_rate.

    Output is the exact byte stream the WS server expects (no WAV header,
    no padding). Read fully into memory — a 90-min recording at 16 kHz
    mono int16 is ~170 MB, well within RAM budget on a modern laptop.
    """
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(audio_path),
            "-ac", "1",
            "-ar", str(sample_rate),
            "-c:a", "pcm_s16le",
            "-f", "s16le",
            "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg decode failed (rc={proc.returncode}) for {audio_path}: "
            f"{proc.stderr.decode('utf-8', errors='replace')[:500]}"
        )
    return proc.stdout


# ───────── event handling ──────────────────────────────────────────────


def _parse_event(raw: Any) -> dict | None:
    """Decode a server message. Binary frames or junk → None (ignored)."""
    if isinstance(raw, (bytes, bytearray)):
        # Server normally pushes JSON text; if it sends binary we don't
        # know what to do — log and skip rather than crash.
        LOGGER.debug("ws-funasr: dropping unexpected binary frame %d bytes", len(raw))
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        LOGGER.warning("ws-funasr: non-JSON text frame: %r", raw[:200] if raw else raw)
        return None


def _name(event: dict) -> str:
    return str((event.get("header") or {}).get("name") or "")


def _absorb_sentence_end(event: dict, sink: list[tuple[int, int, str]]) -> None:
    payload = event.get("payload") or {}
    text = str(payload.get("result") or "").strip()
    if not text:
        return
    begin = int(payload.get("begin_time") or 0)
    end = int(payload.get("time") or begin)
    sink.append((begin, end, text))


def _check_failed(event: dict) -> None:
    if _name(event) == "TaskFailed":
        header = event.get("header") or {}
        msg = header.get("status_text") or header.get("status_message") or "TaskFailed"
        raise TaskFailedError(f"funasr TaskFailed: {msg}")


# ───────── main entry ──────────────────────────────────────────────────


def transcribe(
    audio_path: Path,
    config: WsFunasrConfig,
) -> tuple[contracts.Transcript, dict[str, Any]]:
    """Stream ``audio_path`` to the funasr WS server and return a Transcript.

    Returns ``(Transcript, raw_info)`` matching ``openai_audio.transcribe``.
    Raises ``IdleTimeoutError`` if the server stops sending events for
    longer than ``config.idle_timeout_sec``; raises ``TaskFailedError``
    on server-reported failure. Connection failures bubble up as the
    underlying ``websocket`` exceptions.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")

    pcm = _decode_to_pcm_bytes(audio_path, sample_rate=config.sample_rate)
    duration_sec = len(pcm) / (config.sample_rate * 2)
    chunks = _split_pcm_chunks(pcm, chunk_bytes=config.chunk_bytes)

    t0 = time.perf_counter()
    ws = websocket.create_connection(
        config.url,
        timeout=config.connect_timeout_sec,
    )

    task_id = uuid.uuid4().hex
    sentences: list[tuple[int, int, str]] = []
    partial_count = 0
    last_event_ts = time.monotonic()

    def _record_event_ts() -> None:
        nonlocal last_event_ts
        last_event_ts = time.monotonic()

    def _drain(timeout: float) -> bool:
        """Read all events available within ``timeout``; return True if
        a TranscriptionCompleted was seen (caller should stop looping).
        """
        nonlocal partial_count
        ws.settimeout(timeout)
        completed = False
        while True:
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                break
            except websocket.WebSocketConnectionClosedException:
                raise RuntimeError("ws-funasr: server closed connection unexpectedly")
            event = _parse_event(raw)
            if event is None:
                continue
            _record_event_ts()
            _check_failed(event)
            name = _name(event)
            if name == "SentenceEnd":
                _absorb_sentence_end(event, sentences)
            elif name == "TranscriptionResultChanged":
                partial_count += 1
            elif name == "TranscriptionCompleted":
                completed = True
                # consume any tail message but stop the outer loop
            # else: SentenceBegin / TranscriptionStarted / etc — purely
            # informational, but they reset the idle timer above.
            if completed:
                break
        return completed

    try:
        # 1. StartTranscription
        start_msg = {
            "header": {
                "message_id": uuid.uuid4().hex,
                "task_id": task_id,
                "namespace": "SpeechTranscriber",
                "name": "StartTranscription",
            },
            "payload": {
                "format": "pcm",
                "sample_rate": config.sample_rate,
                "enable_intermediate_result": config.enable_intermediate_result,
                "enable_punctuation_prediction": config.enable_punctuation_prediction,
                "enable_inverse_text_normalization":
                    config.enable_inverse_text_normalization,
            },
        }
        ws.send(json.dumps(start_msg))

        # Wait for TranscriptionStarted with the same idle budget.
        ws.settimeout(config.idle_timeout_sec)
        started = False
        deadline = time.monotonic() + config.idle_timeout_sec
        while not started:
            if time.monotonic() > deadline:
                raise IdleTimeoutError(
                    f"ws-funasr: no TranscriptionStarted within "
                    f"{config.idle_timeout_sec}s"
                )
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            except websocket.WebSocketConnectionClosedException:
                raise RuntimeError("ws-funasr: server closed before TranscriptionStarted")
            event = _parse_event(raw)
            if event is None:
                continue
            _record_event_ts()
            _check_failed(event)
            if _name(event) == "TranscriptionStarted":
                started = True

        # 2. send chunks, draining partial events between sends
        completed_in_send_phase = False
        for idx, ch in enumerate(chunks):
            ws.send(ch, websocket.ABNF.OPCODE_BINARY)
            if config.chunk_send_interval_sec > 0:
                time.sleep(config.chunk_send_interval_sec)
            # Short non-blocking drain — keep idle timer happy. If the
            # server completes mid-send (uncommon but possible on very
            # short clips or with aggressive server-side VAD), break
            # out and skip the StopTranscription step.
            if _drain(timeout=config.send_drain_poll_sec):
                completed_in_send_phase = True
                break
            if time.monotonic() - last_event_ts > config.idle_timeout_sec:
                raise IdleTimeoutError(
                    f"ws-funasr: no server event for {config.idle_timeout_sec}s "
                    f"during chunk send (chunk {idx + 1}/{len(chunks)})"
                )

        # 3. StopTranscription — only if the server hasn't already
        #    declared TranscriptionCompleted on its own.
        if not completed_in_send_phase:
            stop_msg = {
                "header": {
                    "message_id": uuid.uuid4().hex,
                    "task_id": task_id,
                    "namespace": "SpeechTranscriber",
                    "name": "StopTranscription",
                }
            }
            ws.send(json.dumps(stop_msg))

            # 4. wait for trailing SentenceEnd + TranscriptionCompleted
            #    with idle-timer protection (no wall-clock cap).
            while True:
                if _drain(timeout=config.send_drain_poll_sec):
                    break
                if time.monotonic() - last_event_ts > config.idle_timeout_sec:
                    raise IdleTimeoutError(
                        f"ws-funasr: no server event for {config.idle_timeout_sec}s "
                        f"after StopTranscription (have {len(sentences)} sentences)"
                    )
    finally:
        try:
            ws.close()
        except Exception:
            pass

    elapsed = time.perf_counter() - t0
    segments = _segments_from_sentences(sentences)

    transcript = contracts.Transcript(
        recording_id=audio_path.stem,
        backend=BACKEND_ID,
        model="paraformer-large",  # funasr-aipod default; not negotiable per-call here
        language="zh",
        duration_sec=duration_sec,
        segments=segments,
    )
    raw_info = {
        "protocol": PROTOCOL,
        "url": config.url,
        "task_id": task_id,
        "wall_clock_sec": round(elapsed, 2),
        "num_segments": len(segments),
        "num_partials": partial_count,
        "audio_duration_sec": round(duration_sec, 2),
        "num_chunks_sent": len(chunks),
    }
    return transcript, raw_info
