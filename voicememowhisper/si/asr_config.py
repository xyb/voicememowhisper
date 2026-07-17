"""Load ASR backend config from a TOML file.

The goal is to let users stop typing a long `--asr-url / --asr-model /
--asr-host-header ...` each time. Once a config file is in place,
``voicememo-whisper si run <audio>`` picks up the preferred backend
automatically, same ergonomics as before the HTTP backend existed.

Search order
------------
1. ``--asr-config <path>`` on the CLI (explicit).
2. ``$VMW_CONFIG`` environment variable.
3. ``~/.config/voicememowhisper/config.toml``.
4. ``~/.voicememowhisper.toml``.

The first file that exists wins. No merging across multiple locations —
keep it simple, one file.

File format (TOML)
------------------

    [asr]
    backend = "openai-audio"          # or "faster_whisper"
    language = "zh"                   # optional, default "zh"

    [asr.http]                        # only read when backend = "openai-audio"
    url = "http://asr.internal:8000/v1/audio/transcriptions"
    model = "paraformer-large"
    host_header = "asr.internal"      # optional
    # api_key = "sk-..."              # optional
    # response_format = "verbose_json"
    # timeout_sec = 600

    [asr.http.models_by_language]     # optional; see below
    zh = "paraformer-large"
    en = "sensevoice-small"

Per-language models
-------------------
Most self-hosted ASR servers carry several models and only some of them
speak a given language. Passing ``language = "en"`` to a Chinese-only
model does **not** make it transcribe English — the server accepts the
field and ignores it, and you get plausible-looking Chinese-decoded
garbage back with no error. So the language hint alone is not enough:
the model has to change with it.

``[asr.http.models_by_language]`` maps a language code to the model to
ask. When the run's language has an entry here, it wins over the plain
``model`` key; otherwise ``model`` is used as before. This keeps
``--language en`` a single flag that actually works, instead of asking
the caller to remember a second one.

Precedence
----------
CLI flags > config file > built-in defaults. A CLI flag that the user
didn't pass stays ``None`` and doesn't override the config.

Output shape
------------
``load_asr_config`` returns a flat dict keyed by the same names as the
CLI argparse namespace — ``asr_backend`` / ``asr_url`` / ``asr_model`` /
``asr_api_key`` / ``asr_host_header`` / ``asr_response_format`` /
``asr_timeout_sec`` / ``language``. This way the CLI layer can merge
config and args with a single ``dict`` update.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# tomllib is stdlib since Python 3.11. If you're on 3.9/3.10 and want
# to use the config file feature, install the `tomli` package — we fall
# back to it at import time.
try:
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover — 3.9 / 3.10 fallback
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]


# Mapping from config file key → CLI namespace key. Keys the user types
# into the TOML file on the left; how the CLI layer sees them on the right.
_HTTP_KEY_MAP = {
    "url": "asr_url",
    "model": "asr_model",
    "api_key": "asr_api_key",
    "host_header": "asr_host_header",
    "response_format": "asr_response_format",
    "timeout_sec": "asr_timeout_sec",
}

# [asr.ws] → CLI namespace keys for the ws-funasr backend.
# Keys must not collide with [asr.http] — the user is allowed to keep
# both sections in the same config file (one for OpenAI-Audio HTTP
# fallback, one for the streaming WS path) and pick between them via
# ``[asr] backend = "..."``.
_WS_KEY_MAP = {
    "url": "asr_ws_url",
    "idle_timeout_sec": "asr_ws_idle_timeout_sec",
    "connect_timeout_sec": "asr_ws_connect_timeout_sec",
    "sample_rate": "asr_ws_sample_rate",
    "chunk_bytes": "asr_ws_chunk_bytes",
    "chunk_send_interval_sec": "asr_ws_chunk_send_interval_sec",
    "enable_intermediate_result": "asr_ws_enable_intermediate_result",
    "enable_punctuation_prediction": "asr_ws_enable_punctuation_prediction",
    "enable_inverse_text_normalization": "asr_ws_enable_inverse_text_normalization",
}

# [diarize.http] → CLI namespace keys.
_DIARIZE_HTTP_KEY_MAP = {
    "url": "diarize_url",
    "api_key": "diarize_api_key",
    "host_header": "diarize_host_header",
    "timeout_sec": "diarize_timeout_sec",
    "include_embeddings": "diarize_include_embeddings",
    "num_speakers": "diarize_num_speakers",
    "min_speakers": "diarize_min_speakers",
    "max_speakers": "diarize_max_speakers",
}


def default_config_paths() -> list[Path]:
    """Search paths in priority order. First existing file wins."""
    paths: list[Path] = []
    env = os.environ.get("VMW_CONFIG")
    if env:
        paths.append(Path(env).expanduser())
    paths.append(Path.home() / ".config" / "voicememowhisper" / "config.toml")
    paths.append(Path.home() / ".voicememowhisper.toml")
    return paths


def find_config_file(explicit: str | Path | None = None) -> Path | None:
    """Return the first config path that exists, or None.

    ``explicit`` (from ``--asr-config``) short-circuits the search. If
    given but missing, returns None — the caller decides whether to
    error out or fall back.
    """
    if explicit is not None:
        p = Path(explicit).expanduser()
        return p if p.exists() else None
    for p in default_config_paths():
        if p.exists():
            return p
    return None


def load_asr_config(explicit: str | Path | None = None) -> dict[str, Any]:
    """Load and flatten the [asr] section of the TOML config file.

    Returns an empty dict if no config file is found — callers treat
    that as "no config-level defaults, use built-in defaults".

    Unknown keys under [asr.http] are ignored with no warning. The
    rationale: the TOML file is user-authored; a typo shouldn't block
    a transcription run. Unknown keys that matter will surface as "my
    setting didn't take effect", at which point the user re-reads the
    docs.
    """
    path = find_config_file(explicit)
    if path is None:
        return {}

    with path.open("rb") as f:
        data = tomllib.load(f)

    asr = data.get("asr") or {}
    out: dict[str, Any] = {}

    if "backend" in asr:
        out["asr_backend"] = str(asr["backend"])
    if "language" in asr:
        out["language"] = str(asr["language"])

    http = asr.get("http") or {}
    for k, v in http.items():
        mapped = _HTTP_KEY_MAP.get(k)
        if mapped is None:
            continue  # unknown key — ignore silently
        out[mapped] = v

    models_by_language = http.get("models_by_language") or {}
    if models_by_language:
        out["asr_models_by_language"] = {
            str(lang): str(model) for lang, model in models_by_language.items()
        }

    ws = asr.get("ws") or {}
    for k, v in ws.items():
        mapped = _WS_KEY_MAP.get(k)
        if mapped is None:
            continue  # unknown key — ignore silently
        out[mapped] = v

    diarize = data.get("diarize") or {}
    if "backend" in diarize:
        out["diarize_backend"] = str(diarize["backend"])
    diar_http = diarize.get("http") or {}
    for k, v in diar_http.items():
        mapped = _DIARIZE_HTTP_KEY_MAP.get(k)
        if mapped is None:
            continue
        out[mapped] = v

    return out


def build_pipeline_kwargs(merged: dict[str, Any]) -> dict[str, Any]:
    """Translate a merged config dict (CLI-namespace keys) into the
    kwargs shape ``si.pipeline.run`` expects.

    ``merged`` is expected to already reflect the final precedence
    (e.g. CLI > config). Only keys that are present are forwarded —
    so the built-in defaults in ``pipeline.run`` keep applying for
    anything missing.

    Returns a dict with ``asr_backend`` / ``asr_backend_config`` /
    ``diarize_backend`` / ``diarize_backend_config`` ready to splat
    into ``run_pipeline(..., **kwargs)``. Both the CLI entry point
    and the watcher wrapper (``speaker_pipeline.py``) call this so
    the two paths pick up identical backend resolution — avoiding
    the footgun where one goes through the TOML config and the
    other silently falls back to built-in local defaults.
    """
    kwargs: dict[str, Any] = {}

    backend = merged.get("asr_backend") or "faster_whisper"
    kwargs["asr_backend"] = backend
    if backend in ("openai-audio", "openai_audio"):
        cfg: dict[str, Any] = {}
        for merged_key, http_key in (
            ("asr_url", "url"),
            ("asr_model", "model"),
            ("asr_api_key", "api_key"),
            ("asr_host_header", "host_header"),
            ("asr_response_format", "response_format"),
            ("asr_timeout_sec", "timeout_sec"),
        ):
            if merged_key in merged:
                cfg[http_key] = merged[merged_key]
        if "asr_models_by_language" in merged:
            cfg["models_by_language"] = merged["asr_models_by_language"]
        kwargs["asr_backend_config"] = cfg
    elif backend in ("ws-funasr", "ws_funasr"):
        wcfg: dict[str, Any] = {}
        for merged_key, ws_key in (
            ("asr_ws_url", "url"),
            ("asr_ws_idle_timeout_sec", "idle_timeout_sec"),
            ("asr_ws_connect_timeout_sec", "connect_timeout_sec"),
            ("asr_ws_sample_rate", "sample_rate"),
            ("asr_ws_chunk_bytes", "chunk_bytes"),
            ("asr_ws_chunk_send_interval_sec", "chunk_send_interval_sec"),
            ("asr_ws_enable_intermediate_result", "enable_intermediate_result"),
            ("asr_ws_enable_punctuation_prediction", "enable_punctuation_prediction"),
            ("asr_ws_enable_inverse_text_normalization",
             "enable_inverse_text_normalization"),
        ):
            if merged_key in merged:
                wcfg[ws_key] = merged[merged_key]
        kwargs["asr_backend_config"] = wcfg

    d_backend = merged.get("diarize_backend") or "local_pyannote"
    kwargs["diarize_backend"] = d_backend
    if d_backend == "http":
        dcfg: dict[str, Any] = {}
        for merged_key, http_key in (
            ("diarize_url", "url"),
            ("diarize_api_key", "api_key"),
            ("diarize_host_header", "host_header"),
            ("diarize_timeout_sec", "timeout_sec"),
            ("diarize_include_embeddings", "include_embeddings"),
            ("diarize_num_speakers", "num_speakers"),
            ("diarize_min_speakers", "min_speakers"),
            ("diarize_max_speakers", "max_speakers"),
        ):
            if merged_key in merged:
                dcfg[http_key] = merged[merged_key]
        kwargs["diarize_backend_config"] = dcfg

    return kwargs
