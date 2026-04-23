"""HTTP diarization backend.

Posts audio to a self-hosted pyannote service (FastAPI, POST /diarize,
multipart) and rebuilds a ``contracts.Diarization`` + embeddings .npz
from the JSON response, so downstream stages (identify / merge / render)
work unchanged.

The protocol is project-specific (pyannote has no public HTTP standard).
It matches ``diarize-service/server.py`` one-to-one:

  Request:  multipart
      file: audio bytes
      include_embeddings: "true" | "false"
      num_speakers / min_speakers / max_speakers: optional ints

  Response: JSON
      {
        "task": "diarize",
        "model": "...",
        "device": "cuda",
        "duration_sec": 1383.4,
        "infer_sec": 90.2,
        "num_speakers": 11,
        "speakers": ["SPEAKER_00", ...],
        "segments": [{"start": ..., "end": ..., "speaker": "..."}, ...],
        "embeddings": {"SPEAKER_00": [...256 floats...], ...} | null
      }

The segments are already the exclusive (non-overlapping) version, same as
``voicememowhisper.si.diarize.run_diarization`` produces locally.
"""

from __future__ import annotations

import json
import mimetypes
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import contracts

PROTOCOL = "pyannote-http"
BACKEND_ID = "pyannote_http"


@dataclass
class DiarizeHTTPConfig:
    """Knobs for the HTTP diarization client.

    ``url`` is the full endpoint, e.g.
    ``http://127.0.0.1:40792/diarize`` (possibly via an SSH tunnel to the
    GPU host's Traefik).

    ``host_header`` is the Host header the reverse proxy uses for
    routing. Leave ``None`` for direct calls.

    ``include_embeddings`` defaults to True because the identify stage
    downstream needs them. Set False for a quick segmentation-only call.
    """

    url: str
    host_header: str | None = None
    api_key: str | None = None
    include_embeddings: bool = True
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    timeout_sec: float = 900.0
    # Model name is decided server-side (the image pins it). The client
    # stores the expected string so it can be stamped onto the
    # contracts.Diarization.model field for traceability.
    model_hint: str = "pyannote/speaker-diarization-community-1"
    extra_form_fields: dict[str, str] = field(default_factory=dict)


# ───────── multipart encoder (stdlib-only) ─────────────────────────────


def _encode_multipart(
    fields: dict[str, str],
    file_field: str,
    file_path: Path,
) -> tuple[bytes, str]:
    """Encode one file + text fields as ``multipart/form-data``.

    Kept stdlib-only so the pipeline can invoke this backend without
    dragging in ``requests`` / ``httpx`` for a single HTTP POST.
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


# ───────── main entry ──────────────────────────────────────────────────


def run_diarization(
    audio_path: Path,
    config: DiarizeHTTPConfig,
    embeddings_out_path: Path | None = None,
) -> tuple[contracts.Diarization, float, list[str]]:
    """Drop-in replacement for ``diarize.run_diarization`` backed by HTTP.

    Returns the same 3-tuple the local backend returns:
    ``(Diarization, duration_sec, speaker_labels_ordered)``.

    If ``embeddings_out_path`` is given and the server returned
    embeddings, writes an ``.npz`` with one key per speaker label so the
    identify stage can consume it unchanged.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")

    fields: dict[str, str] = {
        "include_embeddings": "true" if config.include_embeddings else "false",
    }
    if config.num_speakers is not None:
        fields["num_speakers"] = str(config.num_speakers)
    if config.min_speakers is not None:
        fields["min_speakers"] = str(config.min_speakers)
    if config.max_speakers is not None:
        fields["max_speakers"] = str(config.max_speakers)
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
            raw = resp.read()
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"diarize HTTP {e.code} from {config.url}: {err_body[:500]}"
        ) from e
    wall_clock = time.perf_counter() - t0

    try:
        payload: dict[str, Any] = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"diarize non-JSON response from {config.url}: {raw[:200]!r}"
        ) from e

    segments: list[contracts.SpeakerSegment] = []
    labels_seen: list[str] = []
    for s in payload.get("segments", []):
        label = str(s["speaker"])
        if label not in labels_seen:
            labels_seen.append(label)
        segments.append(
            contracts.SpeakerSegment(
                start=float(s["start"]),
                end=float(s["end"]),
                label=label,
                confidence=None,
            )
        )

    # Server may also send a canonical speakers list — prefer that as the
    # ordered label list so downstream identify sees a stable order.
    server_speakers = payload.get("speakers")
    if isinstance(server_speakers, list) and server_speakers:
        speaker_labels_ordered = [str(x) for x in server_speakers]
    else:
        speaker_labels_ordered = labels_seen

    # Dump embeddings to .npz matching the local backend's format.
    if embeddings_out_path is not None:
        emb = payload.get("embeddings") or {}
        if emb:
            import numpy as np  # local import — heavy dep, only load when needed

            arrays = {
                str(label): np.asarray(vec, dtype=np.float32)
                for label, vec in emb.items()
            }
            embeddings_out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(str(embeddings_out_path), **arrays)

    duration_sec = float(payload.get("duration_sec") or 0.0)
    num_speakers = int(payload.get("num_speakers") or len(speaker_labels_ordered))
    server_model = str(payload.get("model") or config.model_hint)

    diar = contracts.Diarization(
        recording_id=audio_path.stem,
        backend=BACKEND_ID,
        model=server_model,
        num_speakers=num_speakers,
        segments=segments,
    )

    # Stash a few diagnostic fields on the return in case the caller
    # wants to log them. The pipeline currently ignores these, but it
    # matches the local backend's habit of logging peak RSS.
    diar._wall_clock_sec = round(wall_clock, 2)  # type: ignore[attr-defined]
    diar._infer_sec = payload.get("infer_sec")  # type: ignore[attr-defined]

    return diar, duration_sec, speaker_labels_ordered
