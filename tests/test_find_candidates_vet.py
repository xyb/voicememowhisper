"""`find-candidates --vet` must be cheap enough to actually run, and must never
hand back a clip it failed to clear.

Two bugs this pins down, both found while enrolling speakers into the library:

1. Vetting ran an ASR pass (x --vet-runs) over the *whole* block. A real block
   was 859s, so vetting it cost ~28 min of ASR — to choose 20s out of it. In
   practice the command just looked hung. We only ever need one clean stretch of
   clip length, so vet a bounded window around the middle instead.

2. When vetting found no clean stretch, it printed "skip this block" — and then
   cut the clip anyway, from the unvetted middle. That is the worst possible
   output: a clip that failed vetting but looks vetted, heading straight for
   someone's voice centroid.
"""

from __future__ import annotations

import argparse

import pytest

from voicememowhisper.si import cli as si_cli


# The window rule, as the command applies it. Kept here as the executable spec:
# a long block is bounded to a centred window; a short one is vetted whole.
def window_for(block_start, block_end, need, override=None):
    win = override or max(3.0 * need, 60.0)
    if (block_end - block_start) <= win:
        return block_start, block_end
    mid = (block_start + block_end) / 2.0
    return mid - win / 2.0, mid + win / 2.0


def test_long_block_is_vetted_through_a_bounded_centred_window():
    """The 859s block that made this unusable: vet 60s of it, not all of it."""
    start, end = window_for(2260.3, 3120.2, need=20.0)
    assert end - start == pytest.approx(60.0), "must not vet the whole 859s block"
    assert (start + end) / 2 == pytest.approx((2260.3 + 3120.2) / 2), "centred"
    assert start > 2260.3 and end < 3120.2, "window sits inside the block"


def test_short_block_is_vetted_whole():
    """Below the floor there is nothing to save — vet the lot."""
    assert window_for(100.0, 140.0, need=20.0) == (100.0, 140.0)


def test_window_scales_with_clip_length():
    """3x the clip we need, so there is room to choose a clean stretch within it."""
    start, end = window_for(0.0, 600.0, need=40.0)
    assert end - start == pytest.approx(120.0)


def test_window_floor_applies_to_small_clips():
    """3 x 5s would be a 15s window — too tight to find a clean 5s inside. Floor at 60."""
    start, end = window_for(0.0, 600.0, need=5.0)
    assert end - start == pytest.approx(60.0)


def test_explicit_window_override_wins():
    start, end = window_for(0.0, 600.0, need=20.0, override=300.0)
    assert end - start == pytest.approx(300.0)


def test_vet_window_flag_is_exposed():
    """--vet-window-sec must reach the command, else a dirty middle is unfixable
    without editing source."""
    parser = si_cli.build_parser()
    args = parser.parse_args([
        "library", "find-candidates", "--audio", "a.m4a",
        "--speaker", "SPEAKER_01", "--vet", "--vet-window-sec", "180",
    ])
    assert args.vet_window_sec == 180.0


def test_vet_window_defaults_to_none_so_the_rule_picks():
    parser = si_cli.build_parser()
    args = parser.parse_args([
        "library", "find-candidates", "--audio", "a.m4a", "--speaker", "SPEAKER_01",
    ])
    assert args.vet_window_sec is None


def test_failed_vet_does_not_fall_through_to_a_cut():
    """Guards bug 2 at the source level: the 'no clean stretch' branch must not
    fall through into the cutting block. A clip that failed vetting but looks
    vetted is worse than no clip at all — it goes into a centroid unexamined.

    Asserted on the source because the cut path shells out to ffmpeg over real
    audio; the behaviour is one control-flow edge and this is what actually
    regresses if someone deletes the `continue`.
    """
    import inspect
    src = inspect.getsource(si_cli._cmd_library_find_candidates)
    marker = "no clean stretch long enough"
    assert marker in src
    after = src.split(marker, 1)[1]
    # The next control-flow statement after the failure message must be the bail-out,
    # reached before anything writes a clip.
    bail = after.find("continue")
    cut = after.find("cut →")
    assert bail != -1, "failed vet must bail out, not fall through to the cut"
    assert cut == -1 or bail < cut, "bail-out must come before any clip is written"


# --- clip quality: speech density + gain --------------------------------------
#
# Bug 3 and 4, same enrollment session. Both are things --vet structurally cannot
# catch, because --vet only ever answers "is a second person audible in here".

from voicememowhisper.si.cli import (           # noqa: E402
    _speech_fraction, MIN_SPEECH_FRACTION, QUIET_SOURCE_DBFS,
)


def _seg(start, end, text):
    return {"start": start, "end": end, "text": text}


def test_mostly_silent_clip_is_rejected():
    """The clip that started this: 2.4s of speech in a 20s window. --vet passed it
    ("nobody else audible" — of course, it's silence) and the acoustic QC passed it
    too ("92.5% single speaker"). Both green, and the embedding it produced matched
    nobody, losing even to a wrong speaker. Neither check looks at whether the
    target said anything."""
    segs = [
        _seg(927.8, 930.2, "这个口味比较重，里面的味道也重。"),   # 2.4s, a real utterance
        _seg(930.2, 977.4, "对。"),                              # 2 chars over 47s of silence
    ]
    frac = _speech_fraction(segs, 914.0, 934.0)
    assert frac < MIN_SPEECH_FRACTION, f"a 12%-speech clip must be rejected, got {frac:.0%}"


def test_dense_clip_passes():
    """The stretch that actually worked (self-matched 0.77): continuous speech."""
    segs = [
        _seg(212.8, 218.3, "这一段是连续说话的内容，长度足够填满这个区间"),
        _seg(220.8, 226.5, "接着又说了一段，中间几乎没有停顿"),
        _seg(226.5, 232.2, "然后继续讲下去，语速正常"),
        _seg(232.2, 237.9, "最后收尾的一句话，同样是连续的"),
    ]
    assert _speech_fraction(segs, 212.5, 238.0) >= MIN_SPEECH_FRACTION


def test_asr_stretching_two_words_over_silence_does_not_count_as_speech():
    """ASR sometimes emits one segment spanning 40s of silence for a 2-word
    utterance. Counting its full duration would score a silent clip as 100% speech
    — which is exactly how the bad clip would sneak back in."""
    segs = [_seg(0.0, 40.0, "对。")]      # 2 chars, 40 seconds
    frac = _speech_fraction(segs, 0.0, 40.0)
    assert frac < MIN_SPEECH_FRACTION, (
        f"a 2-char utterance cannot fill 40s; got {frac:.0%}"
    )


def test_speech_fraction_ignores_untranscribed_segments():
    """Empty text = the ASR heard nothing there. Not speech."""
    segs = [_seg(0.0, 10.0, ""), _seg(10.0, 20.0, "   ")]
    assert _speech_fraction(segs, 0.0, 20.0) == 0.0


def test_speech_fraction_needs_a_real_window():
    assert _speech_fraction([_seg(0, 1, "x")], 5.0, 5.0) is None


def test_quiet_source_threshold_separates_far_field_from_meeting_audio():
    """Measured levels: the rolling recorder's family audio at -49..-54 dB, meeting
    audio at -30 dB. Normalizing the former took a self-match from 0.54 to 0.77;
    the latter needs no touching."""
    assert -54.1 < QUIET_SOURCE_DBFS, "far-field audio must trigger normalization"
    assert -30.2 > QUIET_SOURCE_DBFS, "meeting audio must be left alone"
