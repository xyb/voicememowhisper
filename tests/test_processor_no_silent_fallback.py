"""Regression: speaker pipeline failures must NOT silently fall back to WhisperKit.

Why: a failed ASR backend often means the user's chosen path is broken
(server down, timeout, auth issue, model OOM). The historical behaviour
was to swallow the exception and silently retry with the bundled
WhisperKit, which:

1. Wastes 10-30 minutes of WhisperKit time on a long recording while
   the user thinks the configured backend is working.
2. Produces only a plain ``.txt`` (no speaker turns, no diarization),
   so the downstream meetlog flow is silently degraded.
3. Hides the root cause — the user doesn't know the speaker pipeline
   broke until much later.

The 2026-05-24 case: 92 MB / 90-min recording, funasr-aipod client
hit a 600s read timeout, ``processor.py`` silently switched to
WhisperKit, the user only noticed when the resulting ``.txt`` had no
speaker turns and they'd already burned an hour of laptop time.

New behaviour: the speaker pipeline exception propagates. The single
``MemoProcessor`` only auto-uses WhisperKit when the speaker pipeline
is unavailable (not configured / not installed).
"""

# pyright: reportMissingImports=false

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from voicememowhisper.config import Settings


def _make_settings(tmp_path: Path) -> Settings:
    return Settings(
        container_root=tmp_path,
        recordings_dir=tmp_path / "recordings",
        metadata_db=tmp_path / "metadata.db",
        legacy_metadata_db=None,
        transcript_dir=tmp_path / "transcripts",
        archive_dir=tmp_path / "archive",
        archive_enabled=False,
        inbox_dir=None,
        state_db=tmp_path / "state.sqlite",
        whisperkit_cli="whisperkit-cli",
        whisperkit_model="dummy-model",
        whisperkit_extra_args=(),
        language=None,
        processing_order="newest-first",
    )


class _ExplodingSpeakerPipeline:
    """Always raises — represents a misconfigured / broken speaker backend."""

    def transcribe(self, *_a, **_k):  # noqa: D401 - test stub
        raise RuntimeError("speaker pipeline blew up (e.g. ws-funasr idle timeout)")


class _ExplodingWhisperKit:
    """Should never be called on the failure path — its invocation is the bug."""

    def transcribe(self, *_a, **_k):  # noqa: D401 - test stub
        raise AssertionError(
            "WhisperKit must not be auto-invoked after a speaker pipeline failure — "
            "silent fallback is the regression this test guards against"
        )


def test_speaker_pipeline_failure_does_not_silently_fall_back_to_whisperkit(
    tmp_path,
) -> None:
    """The exception from the speaker pipeline must propagate."""
    from voicememowhisper.processor import MemoProcessor
    from voicememowhisper.archive import ArchiveManager
    from voicememowhisper.state import StateStore
    from voicememowhisper.metadata import VoiceMemo, load_voice_memos
    from voicememowhisper.metadata_cache import MetadataCache

    settings = _make_settings(tmp_path)
    settings.transcript_dir.mkdir(parents=True, exist_ok=True)
    settings.recordings_dir.mkdir(parents=True, exist_ok=True)

    fake_audio = tmp_path / "fake.m4a"
    # >1024 bytes to bypass the "corrupt placeholder" quarantine guard.
    fake_audio.write_bytes(b"NOTAUDIO" * 200)
    from datetime import datetime
    memo = VoiceMemo(
        guid="g1",
        path=fake_audio,
        title="t1",
        created_at=datetime(2026, 5, 24, 12, 0, 0),
        duration_seconds=1.0,
    )

    proc = MemoProcessor(
        settings=settings,
        transcriber=_ExplodingWhisperKit(),
        archive=ArchiveManager(settings),
        state=StateStore(settings.state_db),
        metadata=MetadataCache(settings, loader=load_voice_memos),
        speaker_pipeline=_ExplodingSpeakerPipeline(),
    )

    with pytest.raises(RuntimeError, match="speaker pipeline blew up"):
        proc.process(memo)
