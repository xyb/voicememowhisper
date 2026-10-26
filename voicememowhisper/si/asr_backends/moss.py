"""ASR backend that shells out to transcribe.cpp's ``transcribe-cli``.

This runs a MOSS GGUF. With ``--diarize`` the same pass returns anonymous
speaker ids. Those ids are not speaker-library names. The identify stage
can still embed the spans afterwards.

``transcribe-cli`` is not installed system-wide. The 2026-10-26 build lives
at ``/tmp/transcribe.cpp/build/bin/transcribe-cli`` until it is installed
or passed in ``cli``.
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
DEFAULT_MODEL = (
    Path.home()
    / "Library/Application Support/Vimpo/models"
    / "MOSS-Transcribe-Diarize-Q5_K_M.gguf"
)
_BUILT_CLI = Path("/tmp/transcribe.cpp/build/bin/transcribe-cli")
_SEGMENT = re.compile(
    r"\[\s*([0-9]+(?:\.[0-9]+)?)\s*->\s*([0-9]+(?:\.[0-9]+)?)\]"
    r"\s*(S\d+)\s*:\s*(.*\S)\s*$"
)
_DURATION = re.compile(r"duration:\s*([0-9]+(?:\.[0-9]+)?)\s*s")


@dataclass
class MossConfig:
    model_path: str = str(DEFAULT_MODEL)
    cli: str = "transcribe-cli"
    language: str = "zh"
    threads: int = 4
    backend: str = "auto"
    timeout_sec: float = 3600


def resolve_cli(cli: str) -> str:
    if Path(cli).exists() or shutil.which(cli):
        return cli
    if cli == "transcribe-cli" and _BUILT_CLI.exists():
        return str(_BUILT_CLI)
    raise FileNotFoundError(
        f"transcribe-cli not found ({cli}). "
        f"Build transcribe.cpp or pass cli=."
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
    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")
    model = Path(config.model_path).expanduser()
    if not model.exists():
        raise FileNotFoundError(f"MOSS model not found: {model}")
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
