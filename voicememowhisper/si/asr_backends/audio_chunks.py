"""Bound HTTP model inputs without loading a recording into memory."""
from __future__ import annotations

import math
import os
import selectors
import subprocess
import tempfile
import wave
from dataclasses import asdict, replace
from pathlib import Path

from ..contracts import Transcript

SAMPLE_RATE = 16000


def _read_piece(stream, count):
    """Read bounded PCM with an idle timeout; requests may take longer independently."""
    blocks, remaining = [], count
    with selectors.DefaultSelector() as selector:
        selector.register(stream, selectors.EVENT_READ)
        while remaining:
            if not selector.select(timeout=30):
                raise TimeoutError('audio decoder produced no data for 30 seconds')
            block = os.read(stream.fileno(), min(remaining, 65536))
            if not block:
                break
            blocks.append(block)
            remaining -= len(block)
    return b''.join(blocks)


def transcribe_chunks(audio_path, config, recognize):
    limit = float(config.max_audio_sec)
    if not math.isfinite(limit) or not 1 <= limit <= 3600:
        raise ValueError('max_audio_sec must be finite and between 1 and 3600 seconds')
    overlap = float(config.overlap_sec)
    if not math.isfinite(overlap) or not 0 <= overlap < limit:
        raise ValueError('overlap_sec must be finite, nonnegative, and below max_audio_sec')
    if not audio_path.exists():
        raise FileNotFoundError(f'audio not found: {audio_path}')
    # Always decode: container duration metadata can underreport actual samples.
    # A single PCM buffer and a single temporary chunk file bound working memory.
    chunk_bytes = int(limit * SAMPLE_RATE) * 2
    overlap_bytes = int(overlap * SAMPLE_RATE) * 2
    if overlap_bytes >= chunk_bytes:
        raise ValueError('overlap_sec must leave at least one new PCM sample per chunk')
    tail = b''
    pieces, chunks = [], []
    total_frames = 0
    language = config.language or ''
    with tempfile.TemporaryDirectory(prefix='asr-chunks-') as directory, tempfile.TemporaryFile() as errors:
        path = Path(directory) / 'chunk.wav'
        process = subprocess.Popen([
            'ffmpeg', '-nostdin', '-v', 'error', '-i', str(audio_path),
            '-vn', '-ac', '1', '-ar', str(SAMPLE_RATE), '-f', 's16le', 'pipe:1',
        ], stdout=subprocess.PIPE, stderr=errors, bufsize=0)
        try:
            assert process.stdout is not None
            while True:
                fresh = _read_piece(process.stdout, chunk_bytes - len(tail))
                if not fresh:
                    break
                pcm = tail + fresh
                if len(pcm) % 2:
                    raise ValueError('decoder returned an incomplete PCM sample')
                frames = len(pcm) // 2
                start = (total_frames - len(tail) // 2) / SAMPLE_RATE
                duration = frames / SAMPLE_RATE
                shifted = []
                with wave.open(str(path), 'wb') as output:
                    output.setparams((1, 2, SAMPLE_RATE, 0, 'NONE', 'not compressed'))
                    output.writeframes(pcm)
                transcript, info = recognize(path, config)
                language = transcript.language or language
                for segment in transcript.segments:
                    if not segment.text.strip():
                        continue
                    # Zero-length timestamps must stay untimed for speaker-window repair.
                    if not (math.isfinite(segment.start) and math.isfinite(segment.end)):
                        raise ValueError('ASR returned non-finite timestamps')
                    if segment.start < 0 or segment.end < segment.start or segment.end > duration + 0.1:
                        raise ValueError('ASR timestamp lies outside its audio chunk')
                    for word in segment.words or []:
                        if not (math.isfinite(word.start) and math.isfinite(word.end)) or word.start < 0 or word.end < word.start or word.end > duration + 0.1:
                            raise ValueError('ASR word timestamp lies outside its audio chunk')
                    words = None if segment.words is None else [
                        replace(word, start=word.start + start, end=word.end + start)
                        for word in segment.words
                    ]
                    shifted.append(replace(segment, start=segment.start + start,
                                            end=segment.end + start, words=words))
                pieces.append(shifted)
                chunks.append({**info, 'offset_sec': start, 'duration_sec': duration,
                               'transcript': asdict(transcript)})
                total_frames += len(fresh) // 2
                tail = pcm[-overlap_bytes:] if overlap_bytes else b''
            status = process.wait(timeout=30)
            if status:
                errors.seek(0)
                raise RuntimeError('audio decode failed: ' + errors.read(1000).decode(errors='replace'))
            if not total_frames:
                raise ValueError('audio contains no decodable samples')
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
    segments = []
    review_required = False
    previous_word_keys = set()
    for index, piece in enumerate(pieces):
        current_word_keys = set()
        overlap_end = chunks[index - 1]['offset_sec'] + chunks[index - 1]['duration_sec'] if index else 0
        for segment in piece:
            if overlap and len(pieces) > 1 and segment.words:
                words = []
                for word in segment.words:
                    key = (round(word.start, 6), round(word.end, 6), word.text.strip())
                    current_word_keys.add(key)
                    in_overlap = index > 0 and word.start < overlap_end
                    if in_overlap and key in previous_word_keys:
                        continue  # Identical text AND source times agree in both hypotheses.
                    words.append(word)
                    if in_overlap:
                        review_required = True
                if not words:
                    continue
                if len(words) == len(segment.words):
                    segments.append(segment)
                else:
                    segments.append(replace(segment, start=words[0].start, end=words[-1].end,
                                            text=' '.join(w.text.strip() for w in words), words=words))
            else:
                # Without agreement, preserve both hypotheses. Model timing jitter must
                # never make independent midpoint decisions delete both copies of a word.
                segments.append(segment)
                review_required |= bool(overlap and len(pieces) > 1)
        previous_word_keys = current_word_keys
    if any(s.end <= s.start and s.text.strip() for s in segments):
        # Mixed timestamp quality must not bypass the all-untimed repair detector.
        segments = [replace(s, end=s.start, words=None) for s in segments]
    output = Transcript(audio_path.stem, 'openai_audio_http', config.resolved_model,
                        language, total_frames / SAMPLE_RATE, segments)
    return output, {
        'protocol': 'openai-audio', 'url': config.resolved_url,
        'model': config.resolved_model, 'http_status': 200,
        'wall_clock_sec': round(sum(c.get('wall_clock_sec', 0) for c in chunks), 2),
        'response_format': config.response_format, 'host_header': config.host_header,
        'server_language': language, 'server_duration_sec': output.duration_sec,
        'num_segments': len(segments), 'num_words': sum(len(s.words or []) for s in segments),
        'num_chunks': len(chunks), 'max_audio_sec': limit, 'overlap_sec': overlap,
        'overlap_review_required': review_required, 'chunks': chunks,
    }
