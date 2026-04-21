"""ASR backend speaking the OpenAI Audio API protocol.

Covers any server that implements ``POST /v1/audio/transcriptions`` with
``multipart/form-data`` — OpenAI Whisper, FunASR slim (self-hosted),
groq, deepinfra, various proxies. Protocol id in configs: ``openai-audio``.

Why stdlib only
---------------
Uses ``urllib.request`` for the HTTP call so this module can be imported
without the ``[speaker-id]`` extra installed. A thin multipart encoder
lives in ``_encode_multipart`` (the stdlib doesn't provide one).

Usage as a library
------------------

    from pathlib import Path
    from voicememowhisper.si.asr_backends.openai_audio import (
        OpenAIAudioConfig, transcribe,
    )

    cfg = OpenAIAudioConfig(
        url="http://asr.internal:8000/v1/audio/transcriptions",
        model="paraformer-large",
        host_header="asr.internal",
        language="zh",
    )
    transcript, raw = transcribe(Path("clip.m4a"), cfg)
    transcript.to_json(Path("transcript.json"))

Usage as a command
------------------

    python -m voicememowhisper.si.asr_backends.openai_audio \\
        --audio /tmp/clip.m4a \\
        --url http://asr.internal:8000/v1/audio/transcriptions \\
        --model paraformer-large \\
        --host-header asr.internal \\
        --language zh \\
        --output /tmp/clip.transcript.json
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import contracts

# Protocol id used in configs / logs.
PROTOCOL = "openai-audio"

# Backend id written into contracts.Transcript.backend. Distinct from
# PROTOCOL: backend = "how we got it" (openai_audio-http), protocol =
# "what API shape the server speaks". Today they map 1:1, but if later
# we add e.g. a native FunASR WebSocket adapter, backend would change
# while PROTOCOL stays "openai-audio" for HTTP users.
BACKEND_ID = "openai_audio_http"


@dataclass
class OpenAIAudioConfig:
    """All knobs for the openai-audio HTTP client.

    ``url`` is the full endpoint (e.g. ends in ``/v1/audio/transcriptions``)
    because different vendors use different paths even within the same
    protocol family.

    ``host_header`` lets callers route through a reverse proxy that
    dispatches on Host (e.g. a Traefik / nginx rule that selects the
    backend by hostname). Leave ``None`` for direct calls.

    ``api_key`` becomes ``Authorization: Bearer <key>`` when set. For
    FunASR slim and other no-auth self-hosted servers, leave ``None``.

    ``timeout_sec`` is the per-request ceiling; OpenAI-compatible servers
    can be slow when cold, so default is generous.
    """

    url: str
    model: str = "whisper-1"
    api_key: str | None = None
    host_header: str | None = None
    language: str | None = "zh"
    response_format: str = "verbose_json"
    timeout_sec: float = 600.0
    # Extra multipart fields the server may accept (prompt, temperature,
    # vendor extensions). None values are dropped before sending.
    extra_form_fields: dict[str, str] = field(default_factory=dict)


# ───────── multipart encoder (stdlib-only) ─────────────────────────────


def _encode_multipart(
    fields: dict[str, str],
    file_field: str,
    file_path: Path,
) -> tuple[bytes, str]:
    """Encode one file + text fields as ``multipart/form-data``.

    Returns ``(body_bytes, content_type_header)``.

    Kept deliberately simple: one file, string-valued text fields, no
    nested parts, no streaming. Audio files for meeting recordings are
    typically <50 MB, which fits fine in memory. If that ever changes,
    swap in ``requests`` or ``httpx`` which handle streaming uploads.
    """
    boundary = f"----voicememowhisper-{uuid.uuid4().hex}"
    crlf = b"\r\n"
    parts: list[bytes] = []

    for name, value in fields.items():
        if value is None:
            continue
        parts.append(f"--{boundary}".encode())
        parts.append(f'Content-Disposition: form-data; name="{name}"'.encode())
        parts.append(b"")
        parts.append(str(value).encode("utf-8"))

    mime_type, _ = mimetypes.guess_type(str(file_path))
    if mime_type is None:
        mime_type = "application/octet-stream"
    parts.append(f"--{boundary}".encode())
    parts.append(
        f'Content-Disposition: form-data; name="{file_field}"; '
        f'filename="{file_path.name}"'.encode()
    )
    parts.append(f"Content-Type: {mime_type}".encode())
    parts.append(b"")
    parts.append(file_path.read_bytes())
    parts.append(f"--{boundary}--".encode())
    parts.append(b"")

    body = crlf.join(parts)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


# ───────── response mapping ────────────────────────────────────────────


def _segments_from_openai_response(payload: dict[str, Any]) -> list[contracts.Segment]:
    """Map OpenAI Audio ``verbose_json`` response to ``contracts.Segment``.

    Handles two shapes:

    - OpenAI / FunASR slim: top-level ``segments`` list with per-segment
      ``start`` / ``end`` / ``text``.
    - Some servers (e.g. if the client asked for ``json`` not
      ``verbose_json``) return only top-level ``text`` with no segments —
      we emit one synthetic segment spanning the whole recording.

    Word-level timestamps are read from top-level ``words`` (the OpenAI
    shape) and bucketed into the segment whose time range contains them.
    """
    segments_raw = payload.get("segments") or []
    words_raw = payload.get("words") or []

    if not segments_raw:
        text = (payload.get("text") or "").strip()
        if not text:
            return []
        duration = float(payload.get("duration") or 0.0)
        return [contracts.Segment(start=0.0, end=duration, text=text, words=None)]

    # Bucket words into their segment (or leave as None on that segment
    # if the server didn't return word timings).
    def _words_for(seg_start: float, seg_end: float) -> list[contracts.Word] | None:
        bucket: list[contracts.Word] = []
        for w in words_raw:
            try:
                ws = float(w["start"])
                we = float(w["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if ws >= seg_start and we <= seg_end + 1e-3:
                bucket.append(
                    contracts.Word(
                        start=ws,
                        end=we,
                        text=str(w.get("word") or w.get("text") or ""),
                        probability=(
                            float(w["probability"])
                            if w.get("probability") is not None
                            else None
                        ),
                    )
                )
        return bucket or None

    out: list[contracts.Segment] = []
    for s in segments_raw:
        start = float(s["start"])
        end = float(s["end"])
        out.append(
            contracts.Segment(
                start=start,
                end=end,
                text=str(s.get("text") or "").strip(),
                words=_words_for(start, end),
                avg_logprob=(
                    float(s["avg_logprob"])
                    if s.get("avg_logprob") is not None
                    else None
                ),
                no_speech_prob=(
                    float(s["no_speech_prob"])
                    if s.get("no_speech_prob") is not None
                    else None
                ),
            )
        )
    return out


# ───────── main entry ──────────────────────────────────────────────────


def transcribe(
    audio_path: Path,
    config: OpenAIAudioConfig,
) -> tuple[contracts.Transcript, dict[str, Any]]:
    """POST the audio to an OpenAI Audio API-compatible endpoint.

    Returns ``(Transcript, raw_info)``. ``raw_info`` carries diagnostic
    metadata (wall-clock duration, HTTP status, server-declared language,
    etc.) that the caller can log or stuff into a StageMeta.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")

    fields: dict[str, str] = {
        "model": config.model,
        "response_format": config.response_format,
    }
    if config.language:
        fields["language"] = config.language
    for k, v in (config.extra_form_fields or {}).items():
        if v is not None:
            fields[k] = str(v)

    body, content_type = _encode_multipart(fields, "file", audio_path)

    headers: dict[str, str] = {
        "Content-Type": content_type,
        "Accept": "application/json",
    }
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    if config.host_header:
        headers["Host"] = config.host_header

    req = urllib.request.Request(config.url, data=body, headers=headers, method="POST")

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=config.timeout_sec) as resp:
            status = resp.status
            raw = resp.read()
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"openai-audio HTTP {e.code} from {config.url}: {err_body[:500]}"
        ) from e
    elapsed = time.perf_counter() - t0

    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"openai-audio non-JSON response from {config.url}: "
            f"{raw[:200]!r}"
        ) from e

    segments = _segments_from_openai_response(payload)
    duration = float(payload.get("duration") or 0.0)
    language = str(payload.get("language") or config.language or "")

    transcript = contracts.Transcript(
        recording_id=audio_path.stem,
        backend=BACKEND_ID,
        model=config.model,
        language=language,
        duration_sec=duration,
        segments=segments,
    )
    raw_info = {
        "protocol": PROTOCOL,
        "url": config.url,
        "http_status": status,
        "wall_clock_sec": round(elapsed, 2),
        "response_format": config.response_format,
        "host_header": config.host_header,
        "server_language": language,
        "server_duration_sec": duration,
        "num_segments": len(segments),
        "num_words": sum(len(s.words or []) for s in segments),
    }
    return transcript, raw_info


# ───────── CLI entry: ``python -m ...openai_audio ...`` ────────────────


def _cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m voicememowhisper.si.asr_backends.openai_audio",
        description=(
            "One-shot transcribe via any OpenAI Audio API compatible server "
            "(OpenAI Whisper / FunASR slim / groq / deepinfra / ...)."
        ),
    )
    ap.add_argument("--audio", required=True, type=Path, help="input audio file")
    ap.add_argument(
        "--url",
        required=True,
        help="full endpoint URL, e.g. "
        "http://asr.internal:8000/v1/audio/transcriptions",
    )
    ap.add_argument("--model", required=True, help="model name the server accepts")
    ap.add_argument(
        "--api-key", default=None, help="Bearer token (omit for no-auth servers)"
    )
    ap.add_argument(
        "--host-header",
        default=None,
        help="Host header override (needed when routing via a "
        "reverse proxy like Traefik)",
    )
    ap.add_argument("--language", default="zh", help="language hint (default: zh)")
    ap.add_argument(
        "--response-format",
        default="verbose_json",
        choices=["json", "verbose_json", "text", "srt", "vtt"],
        help="server response format (default: verbose_json for timestamps)",
    )
    ap.add_argument(
        "--timeout-sec", type=float, default=600.0, help="per-request timeout"
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write Transcript JSON to this path (default: print summary to stdout)",
    )
    args = ap.parse_args(argv)

    config = OpenAIAudioConfig(
        url=args.url,
        model=args.model,
        api_key=args.api_key,
        host_header=args.host_header,
        language=args.language,
        response_format=args.response_format,
        timeout_sec=args.timeout_sec,
    )

    try:
        transcript, raw_info = transcribe(args.audio, config)
    except Exception as e:  # noqa: BLE001 — CLI needs to print any error clearly
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    if args.output:
        transcript.to_json(args.output)
        print(f"wrote {args.output} ({raw_info['num_segments']} segments)")
    else:
        print(f"model     : {transcript.model}")
        print(f"language  : {transcript.language}")
        print(f"duration  : {transcript.duration_sec:.1f}s")
        print(f"segments  : {len(transcript.segments)}")
        print(f"wall clock: {raw_info['wall_clock_sec']}s")
        if transcript.segments:
            print("\nfirst 3 segments:")
            for s in transcript.segments[:3]:
                print(f"  [{s.start:6.1f} → {s.end:6.1f}] {s.text}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
