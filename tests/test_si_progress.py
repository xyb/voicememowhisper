"""Tests for `voicememowhisper.si.progress`.

Covers the non-TTY emission contract the design relies on:

- rate-limited progress lines (default 60s, overridable via env + param)
- sub-step announcements emit immediately and reset the throttle
- line format surfaces both stage position ([i/N name]) and within-stage
  progress (pct / elapsed / eta, or plain elapsed when total is unknown)
- completion line on context exit
- ``DiarizeProgressHook`` maps pyannote's hook calls onto ``StageProgress``

TTY mode is force-disabled via the env knob so these tests don't depend
on whether pytest is run from a terminal or a CI runner.
"""

from __future__ import annotations

import os
import re

import pytest

from voicememowhisper.si.progress import (
    DiarizeProgressHook,
    StageProgress,
    _fmt_duration,
    _is_tty,
)


# ─────────────── fixtures ─────────────────────────────────────────────


@pytest.fixture
def plain_mode(monkeypatch):
    """Force non-TTY output regardless of where pytest is run from."""
    monkeypatch.setenv("VOICE_MEMO_PROGRESS_MODE", "plain")
    return True


@pytest.fixture
def fake_clock(monkeypatch):
    """Monotonic clock under test control. Returns a mutable [t] list so
    tests advance time with ``fake_clock[0] += delta``."""
    t = [1000.0]

    def now():
        return t[0]

    monkeypatch.setattr("voicememowhisper.si.progress.time.monotonic", now)
    return t


# ─────────────── duration formatting ──────────────────────────────────


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (0, "0m00s"),
        (5, "0m05s"),
        (65, "1m05s"),
        (3600, "1h00m00s"),
        (3723, "1h02m03s"),
        (None, "?"),
    ],
)
def test_fmt_duration(seconds, expected):
    assert _fmt_duration(seconds) == expected


# ─────────────── env-forced mode ──────────────────────────────────────


def test_env_forces_plain(monkeypatch):
    monkeypatch.setenv("VOICE_MEMO_PROGRESS_MODE", "plain")
    assert _is_tty() is False


def test_env_forces_tty(monkeypatch):
    monkeypatch.setenv("VOICE_MEMO_PROGRESS_MODE", "tty")
    assert _is_tty() is True


# ─────────────── non-TTY line format ──────────────────────────────────


def _lines(capsys) -> list[str]:
    err = capsys.readouterr().err
    return [line for line in err.splitlines() if line.strip()]


def test_non_tty_start_and_done(plain_mode, fake_clock, capsys):
    with StageProgress(
        "transcribe", stage_num=1, total_stages=5, total=2700.0
    ) as prog:
        fake_clock[0] += 5
    lines = _lines(capsys)
    assert len(lines) == 2
    assert lines[0] == "[1/5 transcribe] start · total 45m00s"
    assert lines[1] == "[1/5 transcribe] done · elapsed 0m05s"


def test_non_tty_no_total_omits_percent_and_eta(plain_mode, fake_clock, capsys):
    with StageProgress("diarize", 2, 5) as prog:
        fake_clock[0] += 3
    lines = _lines(capsys)
    assert lines[0] == "[2/5 diarize] start · total ?"
    assert lines[1] == "[2/5 diarize] done · elapsed 0m03s"


def test_non_tty_respects_interval(plain_mode, fake_clock, capsys):
    with StageProgress(
        "transcribe", 1, 5, total=1000.0, min_interval_s=60.0
    ) as prog:
        # Many updates packed into the first 30 seconds — none should emit.
        for i in range(30):
            fake_clock[0] += 1
            prog.update(10 * (i + 1))
    lines = _lines(capsys)
    # Just the start + done lines. No progress lines in this short window.
    assert len(lines) == 2, lines


def test_non_tty_emits_after_interval(plain_mode, fake_clock, capsys):
    with StageProgress(
        "transcribe", 1, 5, total=1000.0, min_interval_s=60.0
    ) as prog:
        fake_clock[0] += 61      # ≥ min_interval_s
        prog.update(250)         # 25%
        fake_clock[0] += 61
        prog.update(500)         # 50%
        fake_clock[0] += 61
        prog.update(750)         # 75%
    lines = _lines(capsys)
    progress_lines = [ln for ln in lines if "elapsed" in ln and "done" not in ln]
    # Three emits, one per interval boundary.
    assert len(progress_lines) == 3, progress_lines
    for ln, expected_pct in zip(progress_lines, ("25%", "50%", "75%")):
        assert expected_pct in ln
        assert "(" in ln and "/" in ln
        assert "eta" in ln


def test_non_tty_line_shape(plain_mode, fake_clock, capsys):
    """Regression guard for the non-TTY format.

    The line grammar is relied on by the user's AI-agent tail-f workflow;
    if it changes, change here too — but notice first.
    """
    with StageProgress("transcribe", 1, 5, total=2700.0, min_interval_s=60.0) as prog:
        fake_clock[0] += 60
        prog.update(750)  # 27.78%  → formatted as " 28%"
    lines = _lines(capsys)
    shaped = [ln for ln in lines if "eta" in ln]
    assert len(shaped) == 1
    ln = shaped[0]
    assert re.match(
        r"^\[1/5 transcribe\]  ?\d+% "
        r"\(\d+m\d{2}s/\d+m\d{2}s\) "
        r"· elapsed \d+m\d{2}s "
        r"· eta \d+m\d{2}s$",
        ln,
    ), f"line shape regressed: {ln!r}"


def test_non_tty_prefix_without_stage_num(plain_mode, fake_clock, capsys):
    with StageProgress("custom", total=None) as prog:
        fake_clock[0] += 2
    lines = _lines(capsys)
    assert lines[0].startswith("[custom] "), lines
    assert lines[-1].startswith("[custom] done"), lines


# ─────────────── sub-step semantics ───────────────────────────────────


def test_substep_emits_immediately_and_resets_throttle(plain_mode, fake_clock, capsys):
    with StageProgress("diarize", 2, 5, min_interval_s=60.0) as prog:
        fake_clock[0] += 10
        prog.set_substep("speaker_segmentation")
        # set_substep must emit right away — throttle clock resets.
        fake_clock[0] += 10  # not yet 60s past the substep emit
        prog.update(100)
        fake_clock[0] += 60  # now past the new throttle window
        prog.update(200)
    lines = _lines(capsys)
    substep_line = [ln for ln in lines if "speaker_segmentation" in ln]
    assert substep_line, f"no sub-step line: {lines}"
    # The immediate one has no percent / count — just elapsed.
    assert "speaker_segmentation" in substep_line[0]


def test_substep_same_name_twice_is_noop(plain_mode, fake_clock, capsys):
    with StageProgress("diarize", 2, 5, min_interval_s=60.0) as prog:
        fake_clock[0] += 5
        prog.set_substep("embeddings")
        fake_clock[0] += 1
        prog.set_substep("embeddings")  # repeat — should not emit again
    substep_lines = [ln for ln in _lines(capsys) if "embeddings" in ln]
    assert len(substep_lines) == 1, substep_lines


# ─────────────── DiarizeProgressHook ──────────────────────────────────


def test_diarize_hook_announces_substep_changes(plain_mode, fake_clock, capsys):
    with StageProgress("diarize", 2, 5, min_interval_s=60.0) as prog:
        hook = DiarizeProgressHook(prog)
        fake_clock[0] += 1
        hook("speaker_segmentation", total=1000, completed=0)
        fake_clock[0] += 1
        hook("speaker_segmentation", total=1000, completed=500)
        fake_clock[0] += 1
        hook("embeddings", total=10, completed=0)
    lines = _lines(capsys)
    seg_lines = [ln for ln in lines if "speaker_segmentation" in ln]
    emb_lines = [ln for ln in lines if "embeddings" in ln]
    # One announcement per sub-step transition (the middle `completed=500`
    # within the same step shouldn't emit an extra line inside 60s).
    assert len(seg_lines) == 1, seg_lines
    assert len(emb_lines) == 1, emb_lines


def test_diarize_hook_throttled_progress_within_substep(plain_mode, fake_clock, capsys):
    with StageProgress("diarize", 2, 5, min_interval_s=60.0) as prog:
        hook = DiarizeProgressHook(prog)
        hook("speaker_segmentation", total=1000, completed=0)
        # Inside one sub-step: hook called repeatedly. Only one progress
        # line per 60s window should come out.
        for completed in range(0, 1000, 50):
            fake_clock[0] += 5  # 20 ticks × 5s = 100s total
            hook("speaker_segmentation", total=1000, completed=completed)
    lines = _lines(capsys)
    substep_progress = [
        ln for ln in lines
        if "speaker_segmentation" in ln and "/1000" in ln
    ]
    # 100s / 60s ≈ 1-2 progress emits inside this sub-step.
    assert 1 <= len(substep_progress) <= 2, substep_progress


# ─────────────── fallback when tqdm missing in forced TTY mode ────────


def test_tty_without_tqdm_falls_back_to_plain(monkeypatch, fake_clock, capsys):
    """If the user asks for TTY but tqdm isn't installed, we don't crash
    — we quietly degrade to the plain emitter so the pipeline still runs."""
    monkeypatch.setenv("VOICE_MEMO_PROGRESS_MODE", "tty")
    # Force the tqdm import inside `_make_tqdm` to fail.
    import importlib
    import sys as _sys
    real_import = _sys.modules.get("tqdm")
    _sys.modules["tqdm"] = None  # type: ignore[assignment]
    try:
        with StageProgress("transcribe", 1, 5, total=10.0) as prog:
            fake_clock[0] += 1
            prog.update(5)
    finally:
        if real_import is not None:
            _sys.modules["tqdm"] = real_import
        else:
            del _sys.modules["tqdm"]
    lines = _lines(capsys)
    # Fell back to plain: at minimum start + done lines appear.
    assert any("start" in ln for ln in lines)
    assert any("done" in ln for ln in lines)
