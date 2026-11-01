import wave
from pathlib import Path

import pytest

from voicememowhisper.si import contracts, untimed_repair
from voicememowhisper.si.asr_backends import openai_audio as api


def audio(path, seconds):
    with wave.open(str(path), 'wb') as w:
        w.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        w.writeframes(b'\0\0' * int(seconds * 16000))
    return path


def test_chunks_cover_every_sample_and_offset_timestamps(tmp_path, monkeypatch):
    source = audio(tmp_path / 'sample.wav', 5.25)
    seen = []
    def recognize(path, config):
        with wave.open(str(path)) as w:
            n = w.getnframes()
        seen.append(n)
        duration = n / 16000
        return contracts.Transcript('chunk', 'openai_audio_http', 'test', 'en', duration,
            [contracts.Segment(0, duration, str(len(seen)),
                [contracts.Word(0, duration, 'word')])]), {'wall_clock_sec': 0, 'http_status': 200}
    monkeypatch.setattr(api, '_transcribe_single', recognize)
    transcript, info = api.transcribe(source, api.OpenAIAudioConfig(url='http://example.invalid', max_audio_sec=2, overlap_sec=0))
    assert seen == [32000, 32000, 20000]
    assert transcript.recording_id == 'sample'
    assert transcript.duration_sec == 5.25
    assert [(s.start, s.end) for s in transcript.segments] == [(0, 2), (2, 4), (4, 5.25)]
    assert transcript.segments[-1].words[0].start == 4
    assert info['num_chunks'] == 3


def test_text_only_chunks_still_require_timing_repair(tmp_path, monkeypatch):
    source = audio(tmp_path / 'sample.wav', 3)
    def recognize(path, config):
        return contracts.Transcript('x', 'openai_audio_http', 'test', 'en', 0,
            [contracts.Segment(0, 0, 'text')]), {'wall_clock_sec': 0, 'http_status': 200}
    monkeypatch.setattr(api, '_transcribe_single', recognize)
    transcript, _ = api.transcribe(source, api.OpenAIAudioConfig(url='http://example.invalid', max_audio_sec=2, overlap_sec=0))
    assert untimed_repair.needs_repair(transcript)
    assert [(s.start, s.end) for s in transcript.segments] == [(0, 0), (2, 2)]


def test_failed_second_chunk_does_not_return_partial_success(tmp_path, monkeypatch):
    source = audio(tmp_path / 'sample.wav', 5)
    calls = []
    def recognize(path, config):
        calls.append(path)
        if len(calls) == 2:
            raise RuntimeError('request failed')
        return contracts.Transcript('x', 'openai_audio_http', 'test', 'en', 0, []), {'wall_clock_sec': 0}
    monkeypatch.setattr(api, '_transcribe_single', recognize)
    with pytest.raises(RuntimeError, match='request failed'):
        api.transcribe(source, api.OpenAIAudioConfig(url='http://example.invalid', max_audio_sec=2, overlap_sec=0))
    assert len(calls) == 2


@pytest.mark.parametrize('limit', [0, -1, float('nan'), float('inf')])
def test_invalid_limits_fail_before_request(tmp_path, monkeypatch, limit):
    source = audio(tmp_path / 'sample.wav', 1)
    def forbidden(*args):
        pytest.fail('must not send invalid limit')
    monkeypatch.setattr(api, '_transcribe_single', forbidden)
    with pytest.raises(ValueError, match='max_audio_sec'):
        api.transcribe(source, api.OpenAIAudioConfig(url='http://example.invalid', max_audio_sec=limit))


def test_toml_limit_and_cli_override(tmp_path):
    from voicememowhisper.si.asr_config import load_asr_config, build_pipeline_kwargs
    path = tmp_path / 'config.toml'
    path.write_text('[asr]\nbackend="openai-audio"\n[asr.http]\nurl="http://example.invalid"\nmax_audio_sec=60\n')
    config = load_asr_config(path)
    assert build_pipeline_kwargs(config)['asr_backend_config']['max_audio_sec'] == 60
    config['asr_max_audio_sec'] = 30
    assert build_pipeline_kwargs(config)['asr_backend_config']['max_audio_sec'] == 30


def test_overlap_coverage_and_tail_are_exact(tmp_path, monkeypatch):
    source = audio(tmp_path / 'overlap.wav', 5.25)
    sizes = []
    def recognize(path, config):
        with wave.open(str(path)) as w:
            sizes.append(w.getnframes())
        return contracts.Transcript('x', 'openai_audio_http', 'test', 'en', 0,
            [contracts.Segment(0, 0, 'unverified boundary text')]), {'wall_clock_sec': 0}
    monkeypatch.setattr(api, '_transcribe_single', recognize)
    transcript, info = api.transcribe(source, api.OpenAIAudioConfig(url='http://example.invalid', max_audio_sec=2, overlap_sec=0.5))
    assert sizes == [32000, 32000, 32000, 12000]
    assert [c['offset_sec'] for c in info['chunks']] == [0, 1.5, 3, 4.5]
    assert transcript.duration_sec == 5.25
    assert info['overlap_review_required']
    assert untimed_repair.needs_repair(transcript)


def test_exact_multiple_does_not_send_overlap_only_tail(tmp_path, monkeypatch):
    source = audio(tmp_path / 'exact.wav', 2)
    calls = []
    def recognize(path, config):
        calls.append(path)
        return contracts.Transcript('x', 'openai_audio_http', 'test', 'en', 0, []), {}
    monkeypatch.setattr(api, '_transcribe_single', recognize)
    result, _ = api.transcribe(source, api.OpenAIAudioConfig(url='http://example.invalid', max_audio_sec=2, overlap_sec=0.5))
    assert len(calls) == 1
    assert result.duration_sec == 2


def test_overlap_agreeing_words_keep_one_copy(tmp_path, monkeypatch):
    source = audio(tmp_path / 'words.wav', 3.5)
    count = []
    def recognize(path, config):
        count.append(1)
        words = ([contracts.Word(0.2, 0.4, 'first'), contracts.Word(1.6, 1.8, 'seam')]
                 if len(count) == 1 else [contracts.Word(0.1, 0.3, 'seam'), contracts.Word(1.5, 1.7, 'last')])
        return contracts.Transcript('x', 'openai_audio_http', 'test', 'en', 2,
            [contracts.Segment(0, 2, 'text', words)]), {}
    monkeypatch.setattr(api, '_transcribe_single', recognize)
    result, info = api.transcribe(source, api.OpenAIAudioConfig(url='http://example.invalid', max_audio_sec=2, overlap_sec=0.5))
    assert [w.text for s in result.segments for w in s.words] == ['first', 'seam', 'last']
    assert not info['overlap_review_required']
    assert len(info['chunks'][0]['transcript']['segments'][0]['words']) == 2


def test_fractional_limit_cannot_silently_consume_zero_new_frames(tmp_path):
    source = audio(tmp_path / 'fractional.wav', 1.1)
    with pytest.raises(ValueError, match='PCM sample'):
        api.transcribe(source, api.OpenAIAudioConfig(url='http://example.invalid', max_audio_sec=1.00001, overlap_sec=1.000005))


def test_timing_jitter_never_deletes_both_boundary_hypotheses(tmp_path, monkeypatch):
    source = audio(tmp_path / 'jitter.wav', 3)
    calls = []
    def recognize(path, config):
        calls.append(1)
        start, end = (1.41, 1.61) if len(calls) == 1 else (0.39, 0.59)
        return contracts.Transcript('x', 'openai_audio_http', 'test', 'en', 2,
            [contracts.Segment(start, end, 'boundary', [contracts.Word(start, end, 'boundary')])]), {}
    monkeypatch.setattr(api, '_transcribe_single', recognize)
    result, info = api.transcribe(source, api.OpenAIAudioConfig(url='http://example.invalid', max_audio_sec=2, overlap_sec=1))
    assert len(result.segments) == 2
    assert info['overlap_review_required']


def test_empty_segments_do_not_disable_untimed_repair(tmp_path, monkeypatch):
    source = audio(tmp_path / 'empty.wav', 3)
    calls = []
    def recognize(path, config):
        calls.append(1)
        seg = contracts.Segment(0, 1, '') if len(calls) == 1 else contracts.Segment(0, 0, 'text')
        return contracts.Transcript('x', 'openai_audio_http', 'test', 'en', 2, [seg]), {}
    monkeypatch.setattr(api, '_transcribe_single', recognize)
    result, _ = api.transcribe(source, api.OpenAIAudioConfig(url='http://example.invalid', max_audio_sec=2, overlap_sec=0))
    assert untimed_repair.needs_repair(result)
