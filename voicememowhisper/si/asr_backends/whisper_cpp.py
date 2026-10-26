"""ASR backend that shells out to whisper.cpp's ``whisper-cli``.

The model file is a ggml bin, not a faster-whisper or WhisperKit model.
The Vimpo download ``ggml-large-v3-turbo-q5_0.bin`` is the default path.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..._lock import acquire_compute_lock
from .. import contracts

BACKEND_ID = "whisper_cpp"
DEFAULT_MODEL = (
    Path.home()
    / "Library/Application Support/Vimpo/models/ggml-large-v3-turbo-q5_0.bin"
)


@dataclass
class WhisperCppConfig:
    model_path: str = str(DEFAULT_MODEL)
    cli: str = "whisper-cli"
    language: str = "zh"
    threads: int = 4
    timeout_sec: float = 3600


def parse_whisper_json(payload: dict) -> tuple[list[contracts.Segment], float, str]:
    """Turn whisper.cpp ``-oj`` JSON into segments, duration, language."""
    segments: list[contracts.Segment] = []
    last_ms = 0
    for row in payload.get("transcription") or []:
        offsets = row.get("offsets") or {}
        start_ms = int(offsets.get("from") or 0)
        end_ms = int(offsets.get("to") or start_ms)
        last_ms = max(last_ms, end_ms)
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        segments.append(
            contracts.Segment(start=start_ms / 1000, end=end_ms / 1000, text=text)
        )
    language = str((payload.get("result") or {}).get("language") or "")
    return segments, last_ms / 1000, language


def _to_16k_wav(audio_path: Path, dest: Path) -> None:
    proc = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(audio_path),
            "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le",
            str(dest),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {audio_path}: {proc.stderr[-400:]}"
        )


def transcribe(
    audio_path: Path,
    config: WhisperCppConfig,
) -> tuple[contracts.Transcript, dict[str, Any]]:
    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")
    model = Path(config.model_path).expanduser()
    if not model.exists():
        raise FileNotFoundError(f"whisper.cpp model not found: {model}")
    cli = config.cli
    if shutil.which(cli) is None and not Path(cli).exists():
        raise FileNotFoundError(
            f"whisper-cli not found ({cli}). Install whisper.cpp or pass cli=."
        )

    acquire_compute_lock(what=f"whisper.cpp ({model.name})")
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="vmw-whisper-cpp-") as tmp:
        tmp_path = Path(tmp)
        wav = tmp_path / "audio.wav"
        _to_16k_wav(audio_path, wav)
        out_base = tmp_path / "out"
        proc = subprocess.run(
            [
                cli,
                "-m", str(model),
                "-l", config.language,
                "-t", str(config.threads),
                "-f", str(wav),
                "-oj",
                "-of", str(out_base),
            ],
            capture_output=True,
            text=True,
            timeout=config.timeout_sec,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"whisper-cli exited {proc.returncode}: {proc.stderr[-500:]}"
            )
        payload = json.loads((out_base.with_suffix(".json")).read_text(encoding="utf-8"))
    elapsed = time.perf_counter() - t0
    segments, duration, language = parse_whisper_json(payload)
    transcript = contracts.Transcript(
        recording_id=audio_path.stem,
        backend=BACKEND_ID,
        model=model.name,
        language=language or config.language,
        duration_sec=duration,
        segments=segments,
    )
    return transcript, {
        "wall_clock_sec": round(elapsed, 2),
        "model_path": str(model),
        "times_realtime": round(duration / elapsed, 2) if elapsed else None,
    }
