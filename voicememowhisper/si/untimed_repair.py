"""Opt-in HTTP transcript recovery using existing diarization windows.

Window timestamps describe the entire recognized clip, not individual words.
The caller retains the shared pipeline lock and owns the output cache.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .contracts import Transcript, Segment, Diarization

GAP = "[ASR未返回文字]"
SHORT = "[短应答窗口未补转录，见整场原稿]"


def needs_repair(transcript: Transcript) -> bool:
    if transcript.backend.endswith('+speaker-windows'):
        return any(s.text in (GAP, SHORT) for s in transcript.segments)
    return bool(transcript.segments) and all(
        s.text.strip() and (not math.isfinite(s.start) or
                           not math.isfinite(s.end) or s.end <= s.start)
        for s in transcript.segments
    )


def speaker_windows(diarization: Diarization) -> list[tuple[float, float]]:
    turns = []
    for s in sorted(diarization.segments, key=lambda x: (x.start, x.end)):
        if not (math.isfinite(s.start) and math.isfinite(s.end)):
            raise ValueError('non-finite diarization boundary')
        if s.start < 0 or s.end <= s.start:
            raise ValueError('invalid diarization boundary')
        if (turns and turns[-1][2] == s.label and
                0 <= s.start - turns[-1][1] <= 1):
            turns[-1] = (turns[-1][0], s.end, s.label)
        else:
            turns.append((s.start, s.end, s.label))
    return [(start, end) for start, end, _ in turns]


def cut_audio(source: Path, start: float, end: float, target: Path) -> None:
    subprocess.run([
        'ffmpeg', '-nostdin', '-v', 'error', '-n', '-ss', str(start),
        '-i', str(source), '-t', str(end - start), '-ar', '16000',
        '-ac', '1', '-c:a', 'pcm_s16le', str(target),
    ], check=True, capture_output=True, timeout=60)


def repair(audio: Path, original: Transcript, diarization: Diarization,
           cache_dir: Path, recognize, cut=cut_audio, *, cache_key='',
           progress=None, min_window_sec=0) -> Transcript:
    if not math.isfinite(min_window_sec) or min_window_sec < 0:
        raise ValueError('invalid minimum window duration')
    windows = speaker_windows(diarization)
    if not windows:
        raise ValueError('no speaker windows available for untimed repair')
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Bind successful-window checkpoints to the audio bytes and request config.
    audio_hash = hashlib.sha256(audio.read_bytes()).hexdigest() if audio.exists() else str(audio)
    segments = []
    recognized_count = 0
    for index, (start, end) in enumerate(windows, 1):
        if end - start < min_window_sec:
            segments.append(Segment(start, end, SHORT))
            if progress:
                progress(f'untimed repair {index}/{len(windows)} clips (short gap)')
            continue
        digest = hashlib.sha256(json.dumps(
            [audio_hash, cache_key, start, end], sort_keys=True).encode()).hexdigest()
        cached = cache_dir / (digest + '.json')
        text = None
        if cached.exists():
            try:
                candidate = json.loads(cached.read_text())['text']
                if isinstance(candidate, str) and candidate.strip():
                    text = candidate
            except (ValueError, KeyError, TypeError):
                pass
        if text is None:
            with tempfile.TemporaryDirectory(prefix='vmw-window-') as temp:
                clip = Path(temp) / 'clip.wav'
                cut(audio, start, end, clip)
                result = recognize(clip)
                text = ' '.join(s.text for s in result.segments).strip()
            if text:
                with tempfile.NamedTemporaryFile(mode='w', dir=cache_dir,
                                                 delete=False) as handle:
                    temporary = Path(handle.name)
                    json.dump({'text': text}, handle, ensure_ascii=False)
                os.replace(temporary, cached)
        if text:
            recognized_count += 1
        segments.append(Segment(start, end, text or GAP))
        if progress:
            progress(f'untimed repair {index}/{len(windows)} clips')
    if not recognized_count:
        raise ValueError('untimed repair returned no text; original retained')
    return Transcript(original.recording_id, original.backend + '+speaker-windows',
                      original.model, original.language,
                      max(original.duration_sec, max(e for _, e in windows)), segments)


def preserve_original(path: Path) -> Path:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    backup = path.with_name(path.stem + '.untimed-' + digest + '.json')
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup
