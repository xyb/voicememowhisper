from pathlib import Path
import pytest
from voicememowhisper.si.contracts import Transcript, Segment, Diarization, SpeakerSegment
from voicememowhisper.si.untimed_repair import needs_repair, speaker_windows, repair


def transcript(segments):
    return Transcript('sample', 'http', 'sample-model', 'zh', 12, segments)


def test_only_nonempty_untimed_text_needs_repair():
    assert needs_repair(transcript([Segment(0, 0, 'words')]))
    assert not needs_repair(transcript([]))
    assert not needs_repair(transcript([Segment(2, 3, 'words')]))


def test_windows_do_not_join_across_another_speaker_or_an_overlap():
    d = Diarization('sample', 'test', 'test', 2, [
        SpeakerSegment(1, 2, 'A'), SpeakerSegment(2.2, 3, 'A'),
        SpeakerSegment(2.5, 4, 'B'), SpeakerSegment(4, 5, 'A')])
    assert speaker_windows(d) == [(1, 3), (2.5, 4), (4, 5)]


def test_repair_uses_real_window_bounds_not_server_zero_times(tmp_path):
    d = Diarization('sample', 'test', 'test', 2, [
        SpeakerSegment(1, 2, 'A'), SpeakerSegment(3, 4, 'B')])
    calls = []
    def recognize(path):
        calls.append(path)
        return transcript([Segment(0, 0, 'recognized')])
    def cut(source, start, end, target):
        target.write_bytes(b'sample')
    original = transcript([Segment(0, 0, 'original')])
    result = repair(Path('sample.wav'), original, d, tmp_path, recognize, cut)
    assert [(s.start, s.end, s.text) for s in result.segments] == [
        (1, 2, 'recognized'), (3, 4, 'recognized')]
    assert original.segments[0].text == 'original'
    repair(Path('sample.wav'), original, d, tmp_path, recognize, cut)
    assert len(calls) == 2  # successful window results resume without another request


def test_empty_windows_fail_instead_of_losing_original(tmp_path):
    d = Diarization('sample', 'test', 'test', 1, [])
    with pytest.raises(ValueError, match='no speaker windows'):
        repair(Path('sample.wav'), transcript([Segment(0, 0, 'words')]),
               d, tmp_path, lambda p: None, lambda *args: None)


def test_mixed_valid_and_invalid_segments_are_not_replaced():
    assert not needs_repair(transcript([Segment(0, 0, 'bad'), Segment(1, 2, 'valid')]))


def test_preserve_original_keeps_each_distinct_version(tmp_path):
    from voicememowhisper.si.untimed_repair import preserve_original
    p = tmp_path / 'transcript.json'
    p.write_text('first')
    first = preserve_original(p)
    p.write_text('second')
    second = preserve_original(p)
    assert first != second
    assert first.read_text() == 'first'
    assert second.read_text() == 'second'


def test_corrupt_and_empty_cache_can_retry(tmp_path):
    d = Diarization('sample', 'test', 'test', 1, [SpeakerSegment(1, 2, 'A')])
    audio = tmp_path / 'sample.wav'
    audio.write_bytes(b'audio')
    cache = tmp_path / 'cache'
    def cut(source, start, end, target):
        target.write_bytes(b'clip')
    original = transcript([Segment(0, 0, 'original')])
    with pytest.raises(ValueError, match='no text'):
        repair(audio, original, d, cache, lambda p: transcript([]), cut)
    assert not list(cache.glob('*.json'))
    repair(audio, original, d, cache, lambda p: transcript([Segment(0, 0, 'ok')]), cut)
    checkpoint = next(cache.glob('*.json'))
    checkpoint.write_text('{')
    result = repair(audio, original, d, cache,
                    lambda p: transcript([Segment(0, 0, 'retried')]), cut)
    assert result.segments[0].text == 'retried'


def test_partial_empty_result_is_visible_and_can_retry(tmp_path):
    from voicememowhisper.si.untimed_repair import GAP
    d = Diarization('sample', 'test', 'test', 2, [
        SpeakerSegment(1, 2, 'A'), SpeakerSegment(3, 4, 'B')])
    def cut(source, start, end, target):
        target.write_bytes(b'clip')
    calls = iter([transcript([Segment(0, 0, 'ok')]), transcript([])])
    result = repair(Path('sample.wav'), transcript([Segment(0, 0, 'original')]),
                    d, tmp_path, lambda p: next(calls), cut)
    assert [s.text for s in result.segments] == ['ok', GAP]
    assert needs_repair(result)
    resumed = repair(Path('sample.wav'), result, d, tmp_path,
                     lambda p: transcript([Segment(0, 0, 'recovered')]), cut)
    assert [s.text for s in resumed.segments] == ['ok', 'recovered']


def test_short_windows_remain_visible_without_inventing_words(tmp_path):
    from voicememowhisper.si.untimed_repair import SHORT
    d = Diarization('sample', 'test', 'test', 2, [
        SpeakerSegment(1, 1.2, 'A'), SpeakerSegment(3, 6, 'B')])
    def cut(source, start, end, target):
        target.write_bytes(b'clip')
    result = repair(Path('sample.wav'), transcript([Segment(0, 0, 'original')]),
                    d, tmp_path, lambda p: transcript([Segment(0, 0, 'ok')]),
                    cut, min_window_sec=2)
    assert [s.text for s in result.segments] == [SHORT, 'ok']
    assert needs_repair(result)
