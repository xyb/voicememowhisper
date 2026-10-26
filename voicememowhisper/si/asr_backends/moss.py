"""ASR backend that shells out to transcribe.cpp's ``transcribe-cli``.

This runs a MOSS GGUF. With ``--diarize`` the same pass returns anonymous
speaker ids. Those ids are not speaker-library names. The identify stage
can still embed the spans afterwards.

Pass the model file with ``--asr-model-path`` or ``[asr.moss] model_path``.
``transcribe-cli`` must be on ``PATH`` or passed in ``cli``. There is no
built-in model location or executable path.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..._lock import acquire_compute_lock
from .. import contracts

BACKEND_ID = "moss"
MODEL_REPO = "handy-computer/MOSS-Transcribe-Diarize-gguf"
MODEL_NAME = "MOSS-Transcribe-Diarize-Q5_K_M.gguf"
MODEL_BYTES = 700_314_304


def download_hint(*, mirror: bool = True) -> str:
    """Tell a user who has no MOSS GGUF where to get one, and the command to run."""
    from . import MIRROR_HINT

    text = (
        f"No model file is included. Download {MODEL_NAME} ({MODEL_BYTES:,} bytes):\n"
        f"  hf download {MODEL_REPO} {MODEL_NAME} --local-dir .\n"
        f"  https://huggingface.co/{MODEL_REPO}\n"
        "Pass the saved file with --asr-model-path or [asr.moss] model_path. "
        "transcribe-cli must be on PATH or passed with --asr-cli."
    )
    if mirror:
        text += "\n" + MIRROR_HINT
    return text


_SEGMENT = re.compile(
    r"\[\s*([0-9]+(?:\.[0-9]+)?)\s*->\s*([0-9]+(?:\.[0-9]+)?)\]"
    r"\s*(S\d+)\s*:\s*(.*\S)\s*$"
)
_DURATION = re.compile(r"duration:\s*([0-9]+(?:\.[0-9]+)?)\s*s")


@dataclass
class MossConfig:
    model_path: str = ""
    cli: str = "transcribe-cli"
    language: str = "zh"
    threads: int = 4
    backend: str = "auto"
    timeout_sec: float = 3600


def resolve_cli(cli: str) -> str:
    if Path(cli).exists() or shutil.which(cli):
        return cli
    raise FileNotFoundError(
        f"transcribe-cli not found ({cli}). Install it or pass --asr-cli."
    )


def parse_moss_output(text: str) -> tuple[list[contracts.Segment], list[contracts.SpeakerSegment], float]:
    """Parse transcribe-cli's speaker lines.

    Returns transcript segments, diarization segments, and duration.
    """
    segments: list[contracts.Segment] = []
    speakers: list[contracts.SpeakerSegment] = []
    for line in text.splitlines():
        match = _SEGMENT.search(line)
        if match is None:
            continue
        start = float(match.group(1))
        end = float(match.group(2))
        label = match.group(3)
        spoken = match.group(4).strip()
        segments.append(contracts.Segment(start=start, end=end, text=spoken))
        speakers.append(contracts.SpeakerSegment(start=start, end=end, label=label))
    duration_match = _DURATION.search(text)
    if duration_match:
        duration = float(duration_match.group(1))
    elif segments:
        duration = segments[-1].end
    else:
        duration = 0.0
    return segments, speakers, duration


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
    config: MossConfig,
) -> tuple[contracts.Transcript, contracts.Diarization, dict[str, Any]]:
    if not str(config.model_path or "").strip():
        raise ValueError(
            "MOSS model path is required. Pass --asr-model-path or set "
            "[asr.moss] model_path in the config file.\n"
            + download_hint()
        )
    model = Path(config.model_path).expanduser()
    if not model.exists():
        raise FileNotFoundError(f"MOSS model not found: {model}\n{download_hint()}")
    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")
    cli = resolve_cli(config.cli)

    acquire_compute_lock(what=f"MOSS ({model.name})")
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="vmw-moss-") as tmp:
        wav = Path(tmp) / "audio.wav"
        _to_16k_wav(audio_path, wav)
        proc = subprocess.run(
            [
                cli,
                "-m", str(model),
                "-l", config.language,
                "--threads", str(config.threads),
                "--backend", config.backend,
                "--timestamps", "segment",
                "--diarize",
                str(wav),
            ],
            capture_output=True,
            text=True,
            timeout=config.timeout_sec,
        )
    elapsed = time.perf_counter() - t0
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(
            f"transcribe-cli exited {proc.returncode}: {combined[-500:]}"
        )
    segments, speakers, duration = parse_moss_output(combined)
    if not segments:
        raise RuntimeError(
            "transcribe-cli returned no speaker segments: " + combined[-500:]
        )
    transcript = contracts.Transcript(
        recording_id=audio_path.stem,
        backend=BACKEND_ID,
        model=model.name,
        language=config.language,
        duration_sec=duration,
        segments=segments,
    )
    diarization = contracts.Diarization(
        recording_id=audio_path.stem,
        backend=BACKEND_ID,
        model=model.name,
        num_speakers=len({s.label for s in speakers}),
        segments=speakers,
    )
    return transcript, diarization, {
        "wall_clock_sec": round(elapsed, 2),
        "model_path": str(model),
        "times_realtime": round(duration / elapsed, 2) if elapsed else None,
    }
