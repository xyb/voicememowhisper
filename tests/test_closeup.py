from __future__ import annotations

# pyright: reportMissingImports=false

"""Close-up analysis: spend heavily on a few seconds to see them clearly."""

import numpy as np
import pytest

from voicememowhisper.si import closeup


# ---------- backchannel text classifier (pure) ----------------------------


@pytest.mark.parametrize("text", ["嗯", "嗯嗯", "对", "对对", "哦", "是的", "好的", "呃"])
def test_backchannel_text_accepts_listener_feedback(text) -> None:
    assert closeup.is_backchannel_text(text)


@pytest.mark.parametrize("text", ["这个方案我觉得可以", "对这个方案有意见", "", "嗯我们再看看"])
def test_backchannel_text_rejects_real_speech(text) -> None:
    assert not closeup.is_backchannel_text(text)


# ---------- RMS gate: adaptive thresholds (pure, no audio needed) ---------


def test_adaptive_thresholds_scale_with_the_window() -> None:
    """A window recorded 20 dB hotter must yield thresholds 20 dB hotter.

    The absolute-dB gate this replaces broke the moment the mic moved. What
    marks a backchannel is that it sits *between* this window's noise floor
    and this window's speech level — a relative fact.
    """
    quiet = np.array([-60.0] * 50 + [-45.0] * 20 + [-30.0] * 50)
    loud = quiet + 20.0

    lo_q, hi_q = closeup.adaptive_thresholds(quiet)
    lo_l, hi_l = closeup.adaptive_thresholds(loud)

    assert lo_q < hi_q
    assert lo_l == pytest.approx(lo_q + 20.0, abs=0.01)
    assert hi_l == pytest.approx(hi_q + 20.0, abs=0.01)


def test_adaptive_thresholds_bracket_the_mid_level() -> None:
    """Floor at -60, speech at -30 → the gate must straddle the -45 middle."""
    db = np.array([-60.0] * 50 + [-45.0] * 20 + [-30.0] * 50)
    low, high = closeup.adaptive_thresholds(db)
    assert low < -45.0 < high


# ---------- RMS gate: segment extraction (pure) ---------------------------


def test_segments_from_db_picks_short_mid_energy_runs() -> None:
    """One 0.3s mid-energy run between loud speech → exactly one candidate."""
    hop = 0.01
    db = np.array([-30.0] * 100 + [-45.0] * 30 + [-30.0] * 100)  # 0.3s dip
    times = np.arange(len(db)) * hop

    segs = closeup.segments_from_db(db, times, low_db=-50.0, high_db=-40.0,
                                    dur_min=0.15, dur_max=0.5)

    assert len(segs) == 1
    assert segs[0].duration == pytest.approx(0.29, abs=0.02)
    assert segs[0].source == "rms"


def test_segments_from_db_rejects_runs_that_are_too_long() -> None:
    """A 2s mid-energy stretch is someone actually talking, not a backchannel."""
    hop = 0.01
    db = np.array([-30.0] * 50 + [-45.0] * 200 + [-30.0] * 50)  # 2s
    times = np.arange(len(db)) * hop

    segs = closeup.segments_from_db(db, times, low_db=-50.0, high_db=-40.0,
                                    dur_min=0.15, dur_max=0.5)

    assert segs == []


def test_segments_from_db_rejects_silence_and_loud_speech() -> None:
    hop = 0.01
    db = np.array([-80.0] * 50 + [-30.0] * 50)  # only floor + speech
    times = np.arange(len(db)) * hop

    segs = closeup.segments_from_db(db, times, low_db=-50.0, high_db=-40.0,
                                    dur_min=0.15, dur_max=0.5)

    assert segs == []


# ---------- atempo cascade ------------------------------------------------


@pytest.mark.parametrize("speed,expected", [
    (0.5, "atempo=0.5"),
    (0.25, "atempo=0.5,atempo=0.5"),
    (0.125, "atempo=0.5,atempo=0.5,atempo=0.5"),
])
def test_atempo_chain_reaches_below_ffmpeg_floor(speed, expected) -> None:
    """ffmpeg's atempo bottoms out at 0.5x; going slower means cascading."""
    assert closeup.atempo_filter(speed) == expected


@pytest.mark.parametrize("speed", [1.0, 1.5, 0.0, -1.0])
def test_atempo_rejects_non_slowdown(speed) -> None:
    with pytest.raises(ValueError):
        closeup.atempo_filter(speed)


# ---------- analyze_window: the high-precision entry point ----------------


def test_analyze_window_votes_across_asr_runs(monkeypatch, tmp_path) -> None:
    """Several ASR passes over the same slowed window; agreement raises votes.

    Two of three runs hear 嗯 at the same spot, all three hear the real
    sentence. Both survive; each carries how many runs backed it.
    """
    audio = tmp_path / "clip.m4a"
    audio.write_bytes(b"fake")

    # Timestamps come back on the SLOWED timeline (0.25x → 4x the numbers).
    runs = [
        [(4.0, 5.2, "嗯"), (8.0, 20.0, "这个方案我觉得可以")],
        [(4.1, 5.1, "嗯嗯"), (8.0, 20.0, "这个方案我觉得可以")],
        [(40.0, 41.0, "哦"), (8.0, 20.0, "这个方案我觉得可以")],
    ]
    calls = iter(runs)
    monkeypatch.setattr(closeup, "slowdown_audio", lambda *a, **k: None)

    report = closeup.analyze_window(
        audio, start=10.0, end=25.0,
        asr_fns=[lambda _p, r=r: r for r in runs],
        speed=0.25,
        cut_fn=lambda *a, **k: None,
    )

    texts = {s.text for s in report.segments}
    assert "这个方案我觉得可以" in texts

    sentence = next(s for s in report.segments if s.text == "这个方案我觉得可以")
    assert sentence.votes == 3
    assert not sentence.is_backchannel

    bc = next(s for s in report.segments if s.is_backchannel and s.votes >= 2)
    # Slowed 4.0s ÷ 0.25 speed → 1.0s into the window → 11.0s absolute.
    assert bc.start == pytest.approx(10.0 + 1.0, abs=0.2)


def test_analyze_window_timestamps_are_absolute_to_the_recording(monkeypatch, tmp_path) -> None:
    """A window cut at 100s must report segments at ~100s, not at ~0s."""
    audio = tmp_path / "clip.m4a"
    audio.write_bytes(b"fake")
    monkeypatch.setattr(closeup, "slowdown_audio", lambda *a, **k: None)

    report = closeup.analyze_window(
        audio, start=100.0, end=110.0,
        asr_fns=[lambda _p: [(8.0, 12.0, "开始吧")]],
        speed=0.25,
        cut_fn=lambda *a, **k: None,
    )

    seg = report.segments[0]
    assert seg.start == pytest.approx(102.0, abs=0.1)
    assert seg.end == pytest.approx(103.0, abs=0.1)


def test_analyze_window_reports_clean_subranges(monkeypatch, tmp_path) -> None:
    """The payoff for enrollment: which stretches have nobody else in them.

    Window 0-30s, a backchannel at 10-10.5s → two clean stretches around it.
    """
    audio = tmp_path / "clip.m4a"
    audio.write_bytes(b"fake")
    monkeypatch.setattr(closeup, "slowdown_audio", lambda *a, **k: None)

    report = closeup.analyze_window(
        audio, start=0.0, end=30.0,
        asr_fns=[lambda _p: [(40.0, 42.0, "嗯")]],  # slowed → 10.0-10.5s
        speed=0.25,
        cut_fn=lambda *a, **k: None,
    )

    clean = report.clean_subranges(min_duration=5.0)
    assert len(clean) == 2
    assert clean[0][0] == pytest.approx(0.0, abs=0.1)
    assert clean[0][1] == pytest.approx(10.0, abs=0.2)
    assert clean[1][0] == pytest.approx(10.5, abs=0.2)
    assert clean[1][1] == pytest.approx(30.0, abs=0.1)


def test_clean_subranges_drops_stretches_below_min_duration(monkeypatch, tmp_path) -> None:
    audio = tmp_path / "clip.m4a"
    audio.write_bytes(b"fake")
    monkeypatch.setattr(closeup, "slowdown_audio", lambda *a, **k: None)

    report = closeup.analyze_window(
        audio, start=0.0, end=30.0,
        asr_fns=[lambda _p: [(8.0, 10.0, "嗯")]],  # slowed → 2.0-2.5s
        speed=0.25,
        cut_fn=lambda *a, **k: None,
    )

    # The 0-2s head is too short to enroll from; only the tail survives.
    clean = report.clean_subranges(min_duration=5.0)
    assert len(clean) == 1
    assert clean[0][0] == pytest.approx(2.5, abs=0.2)


# ---------- CLI wiring ----------------------------------------------------


def test_vet_requires_audio(capsys) -> None:
    """--vet has to listen to the audio; without it, fail loudly."""
    from voicememowhisper.si import cli as si_cli

    rc = si_cli.main(["library", "find-candidates", "--recording-id", "nope", "--vet"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "--vet requires --audio" in err


def test_closeup_subcommand_rejects_missing_audio(capsys) -> None:
    from voicememowhisper.si import cli as si_cli

    rc = si_cli.main(["closeup", "/nonexistent/x.m4a", "--start", "0", "--end", "5"])
    assert rc == 2
    assert "audio not found" in capsys.readouterr().err


def test_analyze_window_rejects_an_empty_window(tmp_path) -> None:
    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"fake")
    with pytest.raises(ValueError):
        closeup.analyze_window(audio, start=5.0, end=5.0, asr_fns=[lambda _p: []])


def test_analyze_window_needs_at_least_one_asr(tmp_path) -> None:
    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"fake")
    with pytest.raises(ValueError):
        closeup.analyze_window(audio, start=0.0, end=5.0, asr_fns=[])
