from __future__ import annotations

# pyright: reportMissingImports=false

"""The archive name must keep the source file's real extension.

`archive_filename()` hardcoded `.m4a`, so any non-m4a source (a .flac clip, a
.wav) was archived under a name that lied about its contents. Decoders sniff
the content so playback and transcription still worked, which is exactly why
this went unnoticed — but `file`-based tooling and the human reading the
directory both get misled.
"""

from datetime import datetime
from pathlib import Path

import pytest

from voicememowhisper.archive import ArchiveManager
from voicememowhisper.metadata import VoiceMemo


@pytest.mark.parametrize("suffix", [".m4a", ".flac", ".wav", ".mp3"])
def test_archive_filename_preserves_source_suffix(settings, tmp_path, suffix) -> None:
    memo = VoiceMemo(
        guid="g1",
        path=tmp_path / f"source{suffix}",
        title="a clip",
        created_at=datetime(2026, 7, 10, 9, 51, 0),
    )
    name = ArchiveManager(settings).archive_filename(memo)

    assert name == f"2026-07-10_09-51-00_a clip{suffix}"


def test_archive_filename_uppercase_suffix_is_normalized(settings, tmp_path) -> None:
    memo = VoiceMemo(
        guid="g1",
        path=tmp_path / "source.FLAC",
        title="a clip",
        created_at=datetime(2026, 7, 10, 9, 51, 0),
    )
    assert ArchiveManager(settings).archive_filename(memo).endswith(".flac")


def test_archive_filename_falls_back_to_m4a_when_source_has_no_suffix(settings, tmp_path) -> None:
    memo = VoiceMemo(
        guid="g1",
        path=Path("recording"),
        title="a clip",
        created_at=datetime(2026, 7, 10, 9, 51, 0),
    )
    assert ArchiveManager(settings).archive_filename(memo).endswith(".m4a")
