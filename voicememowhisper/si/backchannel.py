"""Detect short listener-feedback (backchannel) segments in a recording.

Backchannel signals are brief listener responses such as Mandarin "嗯/啊/对/
哦/呃" or English "uh-huh / mm-hmm / yeah" that a listener emits to
acknowledge the speaker. They are a problem for speaker enrollment: when
extracting an enrollment clip for the speaker, the listener's backchannel
gets included and pollutes the speaker library.

This module detects backchannel candidates so callers can exclude them
from enrollment clips. Two pipelines, picked by the recall vs precision
tradeoff:

- ``detect_high_recall``: RMS energy gate ∪ ASR-detected backchannel words
  that miss the RMS gate. Wider candidate list intended for human review.
- ``detect_high_precision``: RMS energy gate AND multi-ASR-run consensus.
  Narrower auto-trusted subset.

Single-recording caveat: the default thresholds here come from one short
reference recording. Cross-recording generalisation is not yet validated.
Recalibrate RMS thresholds, and re-run a slowdown sweep, when applying to
a recording with a different microphone, room SNR, or talker.

The slowdown trick: at 1.0x speed short backchannels often get swallowed
by the speaker's word boundaries; slowing the audio (e.g. 0.25x via
``ffmpeg atempo`` cascade) gives the ASR more frames per token and lets it
emit the backchannel as its own segment. Detected timestamps are mapped
back to the original timeline by multiplying by the slowdown factor.
"""
from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)

# RMS energy gate. These thresholds were tuned for one quiet meeting-room
# recording with mid-distance mic. Recalibrate per recording.
RMS_LOW_DB = -40.0
RMS_HIGH_DB = -27.0
RMS_DURATION_MIN = 0.15
RMS_DURATION_MAX = 0.5

# Default audio slowdown factor for ASR. 0.25x worked best in the reference
# experiment but the optimum is recording-dependent — sweep when in doubt.
DEFAULT_SLOWDOWN = 0.25

# Backchannel keywords. Mandarin single chars + a small phrase whitelist.
# These are language features (not personal data) so they are kept verbatim.
BC_CHARS = set("嗯啊哦呃唉是对")
BC_PHRASES = [
    "嗯", "嗯嗯", "嗯嗯嗯", "啊", "哦", "呃", "唉",
    "对", "对对", "是的", "明白", "好", "好的", "对吧",
]
_PUNCT_RE = re.compile(r"[，。！？、,.!?\s\-]")


@dataclass(frozen=True)
class BackchannelSegment:
    """A detected backchannel candidate, on the original-speed timeline."""

    start: float
    end: float
    source: str  # "rms" / "asr" / "rms+asr"
    rms_db: float | None = None
    asr_text: str | None = None
    asr_votes: int = 0

    @property
    def duration(self) -> float:
        return self.end - self.start


def is_backchannel_text(text: str) -> bool:
    """True if ``text`` looks like a single short listener feedback word."""
    n = _PUNCT_RE.sub("", text or "").strip()
    if not n or len(n) > 4:
        return False
    return all(c in BC_CHARS for c in n) or n in BC_PHRASES


def _decode_to_wav_16k_mono(audio: Path, out_wav: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(audio),
        "-ac", "1", "-ar", "16000",
        str(out_wav),
    ]
    subprocess.run(cmd, check=True)


def _atempo_filter(speed: float) -> str:
    """Build an ``atempo=...`` chain for arbitrary speed in (0, 1).

    Each ``atempo`` filter accepts speed in [0.5, 100]. To go below 0.5x,
    chain multiple filters: 0.25x = ``atempo=0.5,atempo=0.5``.
    """
    if speed >= 1.0 or speed <= 0:
        raise ValueError("speed must be in (0, 1) for slowdown")
    chain: list[float] = []
    remaining = speed
    while remaining < 0.5:
        chain.append(0.5)
        remaining = remaining / 0.5
    chain.append(remaining)
    return ",".join(f"atempo={s}" for s in chain)


def slowdown_audio(audio: Path, out_audio: Path, speed: float) -> None:
    """ffmpeg atempo cascade. Output preserves content, stretched in time."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(audio),
        "-ac", "1", "-ar", "16000",
        "-filter:a", _atempo_filter(speed),
        str(out_audio),
    ]
    subprocess.run(cmd, check=True)


def detect_rms_candidates(
    audio: Path,
    low_db: float = RMS_LOW_DB,
    high_db: float = RMS_HIGH_DB,
    dur_min: float = RMS_DURATION_MIN,
    dur_max: float = RMS_DURATION_MAX,
) -> list[BackchannelSegment]:
    """Find connected runs whose RMS energy and duration look backchannel-like.

    Defaults are tuned for one reference recording. Recalibrate when the
    RMS baseline shifts (different mic distance, room SNR, talker).
    """
    import librosa

    tmp_wav = Path("/tmp") / f"_bc_{audio.stem}.wav"
    _decode_to_wav_16k_mono(audio, tmp_wav)
    try:
        y, sr = librosa.load(str(tmp_wav), sr=16000)
    finally:
        tmp_wav.unlink(missing_ok=True)

    rms = librosa.feature.rms(y=y, frame_length=400, hop_length=160)[0]
    db = 20.0 * np.log10(rms + 1e-10)
    times = librosa.frames_to_time(np.arange(len(db)), sr=sr, hop_length=160)
    mask = (db >= low_db) & (db < high_db)

    out: list[BackchannelSegment] = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            s, e = float(times[i]), float(times[j - 1])
            d = e - s
            if dur_min <= d <= dur_max:
                seg_db = float(db[i:j].mean())
                out.append(BackchannelSegment(s, e, "rms", rms_db=seg_db))
            i = j
        else:
            i += 1
    return out


# ASR backend interface. The caller passes a function that takes a slowed
# audio path and returns a list of (start, end, text) tuples on the slowed
# timeline. The module restores timestamps to the original speed.


def detect_asr_candidates(
    audio: Path,
    asr_fn,
    speed: float = DEFAULT_SLOWDOWN,
) -> list[BackchannelSegment]:
    """Slow audio, transcribe, keep backchannel-only segments."""
    tmp_slow = Path("/tmp") / f"_bc_slow_{audio.stem}_{speed}.wav"
    slowdown_audio(audio, tmp_slow, speed)
    try:
        raw = asr_fn(tmp_slow)
    finally:
        tmp_slow.unlink(missing_ok=True)

    out: list[BackchannelSegment] = []
    for s_slow, e_slow, text in raw:
        if not is_backchannel_text(text):
            continue
        out.append(BackchannelSegment(
            start=s_slow * speed,
            end=e_slow * speed,
            source="asr",
            asr_text=text.strip(),
        ))
    return out


def _merge_close(
    segs: list[BackchannelSegment],
    gap: float = 0.1,
) -> list[BackchannelSegment]:
    """Merge candidates whose intervals are within ``gap`` seconds."""
    if not segs:
        return []
    segs = sorted(segs, key=lambda s: s.start)
    out = [segs[0]]
    for s in segs[1:]:
        last = out[-1]
        if s.start <= last.end + gap:
            sources = sorted({last.source, s.source})
            out[-1] = BackchannelSegment(
                start=last.start,
                end=max(last.end, s.end),
                source="+".join(sources),
                rms_db=last.rms_db if last.rms_db is not None else s.rms_db,
                asr_text=last.asr_text or s.asr_text,
                asr_votes=max(last.asr_votes, s.asr_votes),
            )
        else:
            out.append(s)
    return out


def _near(a: BackchannelSegment, b: BackchannelSegment, tol: float = 0.5) -> bool:
    a_mid = (a.start + a.end) / 2
    b_mid = (b.start + b.end) / 2
    return abs(a_mid - b_mid) <= tol


def detect_high_recall(
    audio: Path,
    asr_fn,
    speed: float = DEFAULT_SLOWDOWN,
) -> list[BackchannelSegment]:
    """RMS gate plus ASR-only gaps. Wide candidate list for human review."""
    rms_segs = detect_rms_candidates(audio)
    asr_segs = detect_asr_candidates(audio, asr_fn, speed=speed)
    asr_only = [a for a in asr_segs if not any(_near(a, r) for r in rms_segs)]
    return _merge_close(rms_segs + asr_only)


def detect_high_precision(
    audio: Path,
    asr_fns: list,
    speeds: list[float] | None = None,
    min_votes: int = 3,
) -> list[BackchannelSegment]:
    """RMS gate AND multi-ASR-run consensus. Narrow high-confidence subset.

    Each entry of ``asr_fns`` is paired with a speed in ``speeds``. A
    candidate from the RMS gate is kept only if at least ``min_votes`` of
    the ASR runs detect a backchannel near it.
    """
    if speeds is None:
        speeds = [DEFAULT_SLOWDOWN] * len(asr_fns)
    if len(asr_fns) != len(speeds):
        raise ValueError("asr_fns and speeds must align")

    rms_segs = detect_rms_candidates(audio)
    all_asr: list[BackchannelSegment] = []
    for fn, sp in zip(asr_fns, speeds):
        all_asr.extend(detect_asr_candidates(audio, fn, speed=sp))

    out: list[BackchannelSegment] = []
    for r in rms_segs:
        votes = sum(1 for a in all_asr if _near(r, a, tol=0.4))
        if votes >= min_votes:
            out.append(BackchannelSegment(
                start=r.start, end=r.end, source="rms+asr",
                rms_db=r.rms_db, asr_votes=votes,
            ))
    return out


__all__ = [
    "BackchannelSegment",
    "is_backchannel_text",
    "slowdown_audio",
    "detect_rms_candidates",
    "detect_asr_candidates",
    "detect_high_recall",
    "detect_high_precision",
    "RMS_LOW_DB",
    "RMS_HIGH_DB",
    "RMS_DURATION_MIN",
    "RMS_DURATION_MAX",
    "DEFAULT_SLOWDOWN",
]
