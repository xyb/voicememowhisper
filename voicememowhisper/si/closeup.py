"""Close-up analysis: spend heavily on a few seconds to see them clearly.

This is a magnifying glass, not a pipeline stage. Everything here is far too
expensive to run over a whole recording — it slows the audio down and
transcribes it several times over. That is the point: on a ten-second window
the cost is seconds, and in exchange you see things a single 1.0x pass cannot.

Two jobs it exists for:

1. **Vetting an enrollment clip.** Before a clip goes into the speaker
   library, you need to know who is actually audible in it. A listener's
   "嗯 / 对 / uh-huh" lasts 0.15-0.5s — short enough that diarization folds it
   into the speaker's block, and it rides along into the clip. Enroll that and
   the speaker's centroid is contaminated with a second voice, quietly, for
   every future recording. :func:`analyze_window` finds those moments and
   :meth:`WindowReport.clean_subranges` hands back the stretches with nobody
   else in them.

2. **Reading a few seconds the main pass got wrong.** When a passage matters
   and the 1.0x transcript is mush, point this at it: several slowed ASR runs
   vote, and each segment reports how many runs backed it, so you can see what
   is solid and what is guesswork.

What it is *not*: a step in the main transcription flow. Nothing in `si run`
calls this, and nothing should — the cost model only works because the window
is small and a human asked for it.

**The slowdown trick.** At 1.0x, a 0.2s "嗯" hides inside the speaker's word
boundaries and the ASR never emits it. Slowing to 0.25x (an ffmpeg ``atempo``
cascade) hands the model 4x the frames per token and it comes out as its own
segment. Timestamps are mapped back to the original timeline.

**Thresholds are relative, on purpose.** What marks a backchannel is that it
sits between *this window's* noise floor and *this window's* speech level. An
absolute dB gate breaks the moment the mic moves; :func:`adaptive_thresholds`
derives the gate from the window itself.
"""
from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

LOGGER = logging.getLogger(__name__)

# Where the gate sits between the window's own noise floor and its speech
# level. A backchannel is audible (above the floor) but markedly weaker than
# the person actually holding the floor.
FLOOR_PCT = 10.0
SPEECH_PCT = 90.0
GATE_LOW_FRAC = 0.25
GATE_HIGH_FRAC = 0.55

RMS_DURATION_MIN = 0.15
RMS_DURATION_MAX = 0.5

# 0.25x won the sweep on the reference recording. The optimum is
# recording-dependent; sweep if a window stays stubbornly unreadable.
DEFAULT_SLOWDOWN = 0.25

# Language features, not personal data.
BC_CHARS = set("嗯啊哦呃唉是对")
BC_PHRASES = [
    "嗯", "嗯嗯", "嗯嗯嗯", "啊", "哦", "呃", "唉",
    "对", "对对", "是的", "明白", "好", "好的", "对吧",
]
_PUNCT_RE = re.compile(r"[，。！？、,.!?\s\-]")


@dataclass(frozen=True)
class CloseupSegment:
    """One thing heard inside the window, on the recording's own timeline."""

    start: float
    end: float
    text: str = ""
    source: str = "asr"          # "rms" / "asr" / "rms+asr"
    votes: int = 0               # how many ASR runs backed this
    rms_db: float | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def is_backchannel(self) -> bool:
        return is_backchannel_text(self.text)


@dataclass
class WindowReport:
    """What a close-up pass found in one window."""

    start: float
    end: float
    segments: list[CloseupSegment] = field(default_factory=list)

    def backchannels(self) -> list[CloseupSegment]:
        return [s for s in self.segments if s.is_backchannel]

    def clean_subranges(self, min_duration: float = 3.0) -> list[tuple[float, float]]:
        """Stretches of the window with no backchannel in them.

        This is the answer to "which part of this block is safe to enroll
        from": cut the intruders out, keep what is still long enough to be
        worth embedding.
        """
        intruders = sorted(
            ((s.start, s.end) for s in self.backchannels()),
            key=lambda p: p[0],
        )
        out: list[tuple[float, float]] = []
        cursor = self.start
        for lo, hi in intruders:
            if lo > cursor and (lo - cursor) >= min_duration:
                out.append((cursor, lo))
            cursor = max(cursor, hi)
        if self.end > cursor and (self.end - cursor) >= min_duration:
            out.append((cursor, self.end))
        return out


def is_backchannel_text(text: str) -> bool:
    """True if ``text`` looks like a single short listener acknowledgement."""
    n = _PUNCT_RE.sub("", text or "").strip()
    if not n or len(n) > 4:
        return False
    return all(c in BC_CHARS for c in n) or n in BC_PHRASES


# ---------- audio plumbing ------------------------------------------------


def atempo_filter(speed: float) -> str:
    """Build an ``atempo=...`` chain for a speed in (0, 1).

    A single ``atempo`` bottoms out at 0.5x, so anything slower is a cascade:
    0.25x = ``atempo=0.5,atempo=0.5``.
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


def cut_window(audio: Path, out_wav: Path, start: float, end: float) -> None:
    """Extract [start, end) as 16k mono wav."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start}", "-to", f"{end}", "-i", str(audio),
            "-ac", "1", "-ar", "16000", str(out_wav),
        ],
        check=True,
    )


def slowdown_audio(audio: Path, out_audio: Path, speed: float) -> None:
    """ffmpeg atempo cascade. Same content, stretched in time."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(audio),
            "-ac", "1", "-ar", "16000",
            "-filter:a", atempo_filter(speed),
            str(out_audio),
        ],
        check=True,
    )


# ---------- RMS gate (pure core, so it is testable without audio) ---------


def adaptive_thresholds(db: np.ndarray) -> tuple[float, float]:
    """Derive the energy gate from the window's own dynamic range.

    Returns (low_db, high_db). Both track the window: move the mic 20 dB
    closer and both thresholds move 20 dB with it.
    """
    import numpy as np

    floor = float(np.percentile(db, FLOOR_PCT))
    speech = float(np.percentile(db, SPEECH_PCT))
    span = speech - floor
    return floor + GATE_LOW_FRAC * span, floor + GATE_HIGH_FRAC * span


def segments_from_db(
    db: np.ndarray,
    times: np.ndarray,
    low_db: float,
    high_db: float,
    dur_min: float = RMS_DURATION_MIN,
    dur_max: float = RMS_DURATION_MAX,
) -> list[CloseupSegment]:
    """Connected runs inside the gate whose length looks backchannel-like.

    Too long and it is not an acknowledgement — it is someone talking.
    """
    mask = (db >= low_db) & (db < high_db)
    out: list[CloseupSegment] = []
    i, n = 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        s, e = float(times[i]), float(times[j - 1])
        if dur_min <= (e - s) <= dur_max:
            out.append(CloseupSegment(
                start=s, end=e, source="rms", rms_db=float(db[i:j].mean()),
            ))
        i = j
    return out


def detect_rms_candidates(
    audio: Path,
    low_db: float | None = None,
    high_db: float | None = None,
    dur_min: float = RMS_DURATION_MIN,
    dur_max: float = RMS_DURATION_MAX,
) -> list[CloseupSegment]:
    """Energy-gate an audio file. Thresholds default to window-adaptive."""
    import librosa

    y, sr = librosa.load(str(audio), sr=16000)
    rms = librosa.feature.rms(y=y, frame_length=400, hop_length=160)[0]
    db = 20.0 * np.log10(rms + 1e-10)
    times = librosa.frames_to_time(np.arange(len(db)), sr=sr, hop_length=160)

    if low_db is None or high_db is None:
        auto_low, auto_high = adaptive_thresholds(db)
        low_db = auto_low if low_db is None else low_db
        high_db = auto_high if high_db is None else high_db

    return segments_from_db(db, times, low_db, high_db, dur_min, dur_max)


# ---------- the entry point -----------------------------------------------


def _same_moment(a: CloseupSegment, b: CloseupSegment, tol: float = 0.4) -> bool:
    return abs((a.start + a.end) / 2 - (b.start + b.end) / 2) <= tol


def analyze_window(
    audio: Path,
    start: float,
    end: float,
    asr_fns: list,
    speed: float = DEFAULT_SLOWDOWN,
    tmp_dir: Path | None = None,
    cut_fn=cut_window,
) -> WindowReport:
    """Look hard at [start, end) and report what is in it.

    Each entry in ``asr_fns`` takes the slowed wav and returns
    ``[(start, end, text), ...]`` on the *slowed* timeline. Pass the same
    backend several times (or several different backends) to get a vote per
    segment: agreement across runs is the confidence signal.

    Timestamps come back absolute to the recording, so a hit 2s into a window
    cut at 100s reports as 102s.
    """
    if end <= start:
        raise ValueError("end must be after start")
    if not asr_fns:
        raise ValueError("at least one ASR function is required")

    tmp_dir = tmp_dir or Path("/tmp")
    win = tmp_dir / f"_closeup_{start:.0f}_{end:.0f}.wav"
    slow = tmp_dir / f"_closeup_{start:.0f}_{end:.0f}_{speed}.wav"

    cut_fn(audio, win, start, end)
    try:
        slowdown_audio(win, slow, speed)
    except Exception:
        win.unlink(missing_ok=True)
        raise

    try:
        heard: list[CloseupSegment] = []
        for asr in asr_fns:
            for s_slow, e_slow, text in asr(slow):
                # Slowed timeline → window-relative → absolute.
                heard.append(CloseupSegment(
                    start=start + s_slow * speed,
                    end=start + e_slow * speed,
                    text=(text or "").strip(),
                    source="asr",
                    votes=1,
                ))
    finally:
        win.unlink(missing_ok=True)
        slow.unlink(missing_ok=True)

    return WindowReport(start=start, end=end, segments=_tally(heard))


def _agree(a: CloseupSegment, b: CloseupSegment) -> bool:
    """Did two runs hear the same thing here?

    Identical text counts, obviously. So does two runs both hearing *some*
    acknowledgement at the same moment — one transcribes 嗯, the other 嗯嗯,
    and demanding they match character-for-character would throw away the
    agreement that matters. On a 0.2s grunt no ASR reproduces the exact
    token; what the vote establishes is that somebody was there.
    """
    if not _same_moment(a, b):
        return False
    return a.text == b.text or (a.is_backchannel and b.is_backchannel)


def _tally(heard: list[CloseupSegment]) -> list[CloseupSegment]:
    """Fold runs that heard the same thing at the same moment into one vote."""
    merged: list[CloseupSegment] = []
    for seg in sorted(heard, key=lambda s: s.start):
        for i, kept in enumerate(merged):
            if _agree(kept, seg):
                merged[i] = CloseupSegment(
                    start=min(kept.start, seg.start),
                    end=max(kept.end, seg.end),
                    text=kept.text,
                    source=kept.source,
                    votes=kept.votes + 1,
                    rms_db=kept.rms_db,
                )
                break
        else:
            merged.append(seg)
    return merged


__all__ = [
    "CloseupSegment",
    "WindowReport",
    "analyze_window",
    "is_backchannel_text",
    "adaptive_thresholds",
    "segments_from_db",
    "detect_rms_candidates",
    "atempo_filter",
    "cut_window",
    "slowdown_audio",
    "DEFAULT_SLOWDOWN",
]
