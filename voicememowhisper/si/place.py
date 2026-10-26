"""Choose local or remote compute for one run.

``remote`` keeps the servers in the config file.
``local`` transcribes with whisper.cpp on this Mac and diarizes with
local pyannote.
``auto`` keeps remote transcription. It moves only diarization onto this
Mac, and only when the machine is plugged in, the battery is high enough,
memory is free, and the CPU is not already busy.

The thresholds are a first cut. They are not a claim that local diarization
is faster. On this Mac it is about as long as the audio.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass

from .._lock import compute_lock_busy

BATTERY_MIN_PERCENT = 80
MEMORY_FREE_MIN_GB = 8.0


@dataclass(frozen=True)
class MachineSnapshot:
    power_known: bool
    on_ac: bool
    battery_percent: int | None
    memory_free_gb: float | None
    load_1min: float | None
    cpu_count: int
    compute_lock_held: bool


def local_diarize_blockers(snapshot: MachineSnapshot) -> list[str]:
    """Reasons ``auto`` must keep diarization on the configured server."""
    blockers: list[str] = []
    if not snapshot.power_known:
        blockers.append("power state unreadable")
    elif not snapshot.on_ac:
        percent = (
            f" ({snapshot.battery_percent}%)"
            if snapshot.battery_percent is not None
            else ""
        )
        blockers.append(f"on battery{percent}")
    elif (
        snapshot.battery_percent is not None
        and snapshot.battery_percent < BATTERY_MIN_PERCENT
    ):
        blockers.append(
            f"battery {snapshot.battery_percent}% < {BATTERY_MIN_PERCENT}%"
        )
    if snapshot.memory_free_gb is None:
        blockers.append("free memory unreadable")
    elif snapshot.memory_free_gb < MEMORY_FREE_MIN_GB:
        blockers.append(
            f"free memory {snapshot.memory_free_gb:.1f} GB "
            f"< {MEMORY_FREE_MIN_GB:.0f} GB"
        )
    if snapshot.compute_lock_held:
        blockers.append("another voicememo-whisper model is running")
    if snapshot.load_1min is not None and snapshot.cpu_count:
        limit = max(2.0, snapshot.cpu_count * 0.5)
        if snapshot.load_1min >= limit:
            blockers.append(
                f"load {snapshot.load_1min:.1f} >= {limit:.1f}"
            )
    return blockers


def parse_pmset(text: str) -> tuple[bool, bool, int | None]:
    """Return ``(power_known, on_ac, battery_percent)`` from ``pmset -g batt``."""
    source = re.search(r"Now drawing from '([^']+)'", text)
    if source is None:
        return False, False, None
    on_ac = source.group(1) == "AC Power"
    percent_match = re.search(r"(\d+)%", text)
    percent = int(percent_match.group(1)) if percent_match else None
    return True, on_ac, percent


def _memory_free_gb() -> float | None:
    try:
        out = subprocess.run(
            ["memory_pressure"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"System-wide memory free percentage:\s*(\d+)%", out)
    if match is None:
        return None
    try:
        total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        return None
    return total * int(match.group(1)) / 100 / 1e9


def read_snapshot() -> MachineSnapshot:
    try:
        text = subprocess.run(
            ["pmset", "-g", "batt"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        text = ""
    power_known, on_ac, percent = parse_pmset(text)
    try:
        load = os.getloadavg()[0]
    except OSError:
        load = None
    return MachineSnapshot(
        power_known=power_known,
        on_ac=on_ac,
        battery_percent=percent,
        memory_free_gb=_memory_free_gb(),
        load_1min=load,
        cpu_count=os.cpu_count() or 1,
        compute_lock_held=compute_lock_busy(),
    )


def apply_place(merged: dict, place: str, snapshot: MachineSnapshot | None) -> tuple[dict, str]:
    """Return the flat config dict after applying ``--place``.

    Call this before ``build_pipeline_kwargs``. ``snapshot`` is required for
    ``auto``. ``local`` and ``remote`` ignore it.
    """
    if place == "remote":
        return merged, "remote: use the servers in the config file"
    if place == "local":
        out = dict(merged)
        out["asr_backend"] = "whisper-cpp"
        out["diarize_backend"] = "local_pyannote"
        return out, "local: whisper.cpp on this Mac, pyannote on this Mac"
    if place != "auto":
        raise ValueError(f"unknown place: {place!r}")
    if snapshot is None:
        raise ValueError("place='auto' needs a machine snapshot")
    blockers = local_diarize_blockers(snapshot)
    if blockers:
        return merged, "auto: " + "; ".join(blockers) + "; keep configured diarize"
    out = dict(merged)
    out["diarize_backend"] = "local_pyannote"
    return out, (
        "auto: AC power, battery and memory are sufficient, CPU is free; "
        "diarize on this Mac"
    )
