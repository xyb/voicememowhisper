# ASR Backends

The speaker-id pipeline's transcription stage is pluggable. Different
ASR models have very different strengths — Whisper's English is better
than its Chinese, Paraformer's Chinese business terms are better than
Whisper's, small models miss rare words that larger ones catch. Running
the audio through more than one model and cross-checking is the only
reliable way to keep meeting notes accurate.

This document covers just the **code**: what backends exist, how to
invoke them, how to add a new one. Methodology (when to run which,
how to merge diverging outputs, model speed benchmarks) lives in
operator notes outside this repo.

## Status

Two dimensions, independent:

| Dimension | Values |
|---|---|
| **Deployment** | `local` (in-process Python), `cli` (shell out), `self-hosted HTTP`, `external HTTP API` |
| **Protocol** | `openai-audio` (implemented), `deepgram` / `azure-stt` / `google-stt` / `aliyun-nls` / `funasr-websocket` / `whisper-cpp-server` (not implemented) |

Currently shipped:

- **`faster_whisper`** (local Python): the existing default. Lives in
  `voicememowhisper/si/transcribe.py`. Migration into `asr_backends/`
  is a later commit; today it's unchanged.
- **`openai_audio_http`** (HTTP, `openai-audio` protocol): works with
  any server that implements `POST /v1/audio/transcriptions` with
  `multipart/form-data` — OpenAI Whisper, a self-hosted FunASR slim,
  groq, deepinfra, and any compatible proxy.

Wired into `pipeline.run()` and `voicememo-whisper si run`. Pass
`--asr-backend openai-audio` plus `--asr-url` / `--asr-model` to
swap the stage-1 engine.

## Why two axes — and why the config must name the protocol

Deployment tells you *where* the model runs (latency, privacy,
dependencies). Protocol tells you *what bytes cross the wire*. Same
deployment can speak multiple protocols; same protocol can run
self-hosted or remote.

The config must declare the protocol explicitly. `openai-audio` and
`deepgram` both POST audio to an HTTPS URL, but the request body
format, auth header, and response schema are completely different.
Auto-detecting is fragile — one server (FunASR) can expose both
`/v1/audio/transcriptions` (openai-audio) and `/ws/v1/asr` (FunASR
WebSocket) at once.

## `openai-audio` backend

### As a library

```python
from pathlib import Path
from voicememowhisper.si.asr_backends.openai_audio import (
    OpenAIAudioConfig, transcribe,
)

cfg = OpenAIAudioConfig(
    url="http://asr.internal:8000/v1/audio/transcriptions",
    model="paraformer-large",
    host_header="asr.internal",   # needed when routing via a reverse proxy
    language="zh",
)
transcript, raw = transcribe(Path("clip.m4a"), cfg)
transcript.to_json(Path("clip.transcript.json"))
```

Returns a standard `contracts.Transcript` with `backend="openai_audio_http"`.
Downstream pipeline stages don't need to know or care which backend
produced it.

### As a one-shot CLI

```sh
python -m voicememowhisper.si.asr_backends.openai_audio \
  --audio /tmp/clip.m4a \
  --url http://asr.internal:8000/v1/audio/transcriptions \
  --model paraformer-large \
  --host-header asr.internal \
  --output /tmp/clip.transcript.json
```

Useful for ad-hoc cross-checking a known-problem segment against a
different model without touching pipeline state.

### Via a config file (recommended for daily use)

Writing `--asr-url / --asr-model / --asr-host-header ...` every time
gets tedious. Put the defaults in a TOML file and the CLI picks them
up automatically:

```toml
# ~/.config/voicememowhisper/config.toml
[asr]
backend = "openai-audio"
language = "zh"

[asr.http]
url = "http://asr.internal:8000/v1/audio/transcriptions"
model = "paraformer-large"
host_header = "asr.internal"
# api_key = "sk-..."           # optional
# response_format = "verbose_json"
# timeout_sec = 600
```

With the file in place, just run:

```sh
voicememo-whisper si run /path/to/meeting.m4a
```

Search order for the config file:

1. `--asr-config <path>` on the CLI
2. `$VMW_CONFIG` environment variable
3. `~/.config/voicememowhisper/config.toml`
4. `~/.voicememowhisper.toml`

First hit wins. No file found → falls back to the built-in default
(`faster_whisper`), so existing users see no change.

Precedence: CLI flags > config file > defaults. Pass `--asr-model
other-model` once to override the config for a single run.

### Via CLI flags (ad-hoc override)

```sh
voicememo-whisper si run /path/to/meeting.m4a \
  --from 1 --to 1 --force \
  --asr-backend openai-audio \
  --asr-url http://asr.internal:8000/v1/audio/transcriptions \
  --asr-model paraformer-large \
  --asr-host-header asr.internal

# Then finish stages 2–5 from cache (diarize still runs locally)
voicememo-whisper si run /path/to/meeting.m4a --from 2 --to 5
```

Cache file name is keyed by backend: `transcript_faster_whisper.json`
vs `transcript_openai_audio.json`, so different engines' outputs
coexist under the same `runs/<id>/` without clobbering. Downstream
stages work transparently — they consume the `Transcript` object in
memory, not the filename.

### Config fields

| Field | Required | Notes |
|---|---|---|
| `url` | yes | Full endpoint. Different vendors use different paths. |
| `model` | yes | Server-specific. `whisper-1` (OpenAI), `paraformer-large` / `sensevoice-small` / `fun-asr-nano` (FunASR slim). |
| `api_key` | no | Becomes `Authorization: Bearer <key>`. Omit for no-auth self-hosted servers. |
| `host_header` | no | Overrides the `Host` header. Needed when routing via a reverse proxy (Traefik / nginx / Caddy) that dispatches on hostname. |
| `language` | no | Default `"zh"`. Set to `None` to let the server auto-detect. |
| `response_format` | no | Default `"verbose_json"` (gets timestamps). Other values: `"json"`, `"text"`, `"srt"`, `"vtt"`. |
| `timeout_sec` | no | Default 600. Some servers are slow on first request (cold model load). |
| `extra_form_fields` | no | Dict of extra multipart fields (e.g. `prompt`, `temperature`) for servers that accept them. |

### Full CLI flag list

| Flag | Default | Meaning |
|---|---|---|
| `--asr-backend` | `faster_whisper` | `faster_whisper` or `openai-audio` |
| `--asr-url` | — | HTTP endpoint (openai-audio only) |
| `--asr-model` | — | Server-side model name (openai-audio only) |
| `--asr-api-key` | — | Bearer token for external APIs |
| `--asr-host-header` | — | Override `Host` header (reverse-proxy routing) |
| `--asr-response-format` | `verbose_json` | `json` / `verbose_json` / `text` / `srt` / `vtt` |
| `--asr-timeout-sec` | 600 | Per-request timeout |

### Routing behind a reverse proxy

If the server is behind a reverse proxy (Traefik / nginx / Caddy) that
selects the backend by hostname, set both the URL (pointing at the
proxy) and `host_header` (the hostname the proxy expects). If the
proxy only listens on localhost on a remote host, use an SSH port
forward from the client:

```sh
ssh -f -N -L <local-port>:127.0.0.1:<proxy-port> <user>@<proxy-host>
```

Then point `--asr-url` at `http://127.0.0.1:<local-port>/...` and
keep `--asr-host-header` set to the hostname the proxy expects.

## Adding a new protocol

Create a sibling module, e.g. `asr_backends/deepgram.py`. Expose the
same shape:

```python
PROTOCOL = "deepgram"
BACKEND_ID = "deepgram_http"

@dataclass
class DeepgramConfig: ...

def transcribe(
    audio_path: Path, config: DeepgramConfig
) -> tuple[contracts.Transcript, dict[str, Any]]: ...
```

Map the vendor's response shape into `contracts.Transcript` —
`segments[].{start,end,text}` with optional `words[]`. Downstream
code already handles missing `avg_logprob` / `no_speech_prob`.

Add a `if __name__ == "__main__"` block for the one-shot CLI.

Add tests under `tests/test_asr_backend_<protocol>.py` that mock
`urlopen` with a vendor-shaped response payload.

## Roadmap

1. ✅ `openai-audio` adapter + tests + one-shot CLI.
2. ✅ Pipeline integration: `si run --asr-backend openai-audio ...`.
3. Multi-backend mode: run two or three backends in parallel and
   emit a `CanonicalTranscript` with per-segment divergences flagged.
4. Protocol expansion as real needs arise (deepgram / whisper.cpp-server
   are the likely next two).

See `contracts.CanonicalTranscript` and `contracts.Divergence` —
those types already exist and are the intended output of step 3.
