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

    return out
