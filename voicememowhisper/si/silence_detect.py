"""Detect silence/noise clusters that pyannote mistook as real speakers.

pyannote diarize occasionally assigns a long near-silent segment (meeting
pre-roll, a participant's muted stretch, ambient noise) its own cluster
label. It propagates into identify → merge → render, and shows up in the
final transcript as a phantom SPEAKER_xx with ASR hallucination text.

This module compares the mean dB energy of each unresolved (UNMATCHED)
speaker against the mean of all identified speakers. Labels whose energy
is far below the baseline are relabeled to "[静音]" so the final transcript
shows the gap explicitly instead of confusing the reader.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np

from . import contracts

LOGGER = logging.getLogger(__name__)

DEFAULT_SILENCE_THRESHOLD_DB = 12.0
SILENCE_SPEAKER_NAME = "[静音]"


def _load_audio_mono_16k(audio: Path) -> tuple[int, np.ndarray]:
    """Decode audio to mono 16 kHz float32 PCM via ffmpeg (in-memory)."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(audio),
        "-ac", "1", "-ar", "16000",
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    samples = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return 16000, samples


def _rms_dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    if rms <= 1e-10:
        return -120.0
    return 20.0 * float(np.log10(rms))


def _speaker_rms_by_label(
    sr: int,
    samples: np.ndarray,
    merged: contracts.MergedTranscript,
) -> dict[str, float]:
    per_label: dict[str, list[np.ndarray]] = {}
    for seg in merged.segments:
        if seg.speaker_label == "UNKNOWN":
            continue
        s_idx = max(0, int(seg.start * sr))
        e_idx = min(len(samples), int(seg.end * sr))
        if e_idx <= s_idx:
            continue
        per_label.setdefault(seg.speaker_label, []).append(samples[s_idx:e_idx])
    return {
        label: _rms_dbfs(np.concatenate(chunks))
        for label, chunks in per_label.items()
        if chunks
    }


def detect_silence_speakers(
    audio_path: Path,
    merged: contracts.MergedTranscript,
    threshold_db: float = DEFAULT_SILENCE_THRESHOLD_DB,
) -> list[tuple[str, float, float]]:
    """Relabel unresolved speakers whose energy is far below identified baseline.

    Mutates ``merged`` in place: matching segments get ``speaker_name`` set
    to ``SILENCE_SPEAKER_NAME`` and ``needs_review`` cleared. Relabeled
    labels are removed from ``unresolved_labels``.

    Returns a list of ``(label, label_dbfs, baseline_dbfs)`` tuples for
    anything that was relabeled, so callers can print a one-line summary.
    """
    if not merged.unresolved_labels:
        return []

    identified_labels = {
        s.speaker_label for s in merged.segments if s.speaker_name is not None
    }
    if not identified_labels:
        LOGGER.debug("silence-detect: no identified speakers, no baseline → skip")
        return []

    try:
        sr, samples = _load_audio_mono_16k(audio_path)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        LOGGER.warning("silence-detect: ffmpeg decode failed (%s), skipping", exc)
        return []

    per_label_db = _speaker_rms_by_label(sr, samples, merged)
    if not per_label_db:
        return []

    baseline_values = [per_label_db[lbl] for lbl in identified_labels if lbl in per_label_db]
    if not baseline_values:
        return []
    baseline = float(np.mean(baseline_values))

    relabeled: list[tuple[str, float, float]] = []
    for label in list(merged.unresolved_labels):
        label_db = per_label_db.get(label)
        if label_db is None:
            continue
        if baseline - label_db >= threshold_db:
            for seg in merged.segments:
                if seg.speaker_label == label:
                    seg.speaker_name = SILENCE_SPEAKER_NAME
                    seg.needs_review = False
            relabeled.append((label, label_db, baseline))

    if relabeled:
        silenced_labels = {lbl for lbl, _, _ in relabeled}
        merged.unresolved_labels = [
            lbl for lbl in merged.unresolved_labels if lbl not in silenced_labels
        ]
    return relabeled
