from __future__ import annotations

# pyright: reportMissingImports=false

"""Inbox must accept .flac.

reccap extracts clips as .flac and drops them in the Inbox, and the
`obsidian-transcript-notes` flow feeds them to vmw. But .flac was missing from
DEFAULT_INBOX_EXTENSIONS, so InboxProcessor skipped every one of them: the
batch scan never picked them up (they were also invisible to --dry-run) and
they piled up in the Inbox indefinitely.
"""

import pytest

import voicememowhisper.service as svc
from voicememowhisper.inbox import DEFAULT_INBOX_EXTENSIONS


def test_flac_is_an_accepted_inbox_extension() -> None:
    assert ".flac" in DEFAULT_INBOX_EXTENSIONS


@pytest.mark.parametrize("suffix", [".flac", ".m4a"])
def test_inbox_file_is_moved_into_archive(service_factory, settings, monkeypatch, suffix) -> None:
    from datetime import datetime

    monkeypatch.setattr(svc, "_date_from_filename", lambda _p: datetime(2026, 7, 10, 9, 51, 0))
    service = service_factory(settings)

    inbox_file = settings.inbox_dir / f"2026-07-10 clip{suffix}"  # type: ignore[operator]
    inbox_file.write_text("audio")

    dest = service._process_inbox_file(inbox_file)

    assert dest is not None, f"{suffix} must be accepted by the Inbox"
    assert not inbox_file.exists()
    assert dest.exists()
    assert dest.suffix == suffix


def test_process_one_consumes_an_inbox_flac(service_factory, settings, monkeypatch) -> None:
    """The positional entry point must not choke on a .flac from reccap."""
    enqueued = []
    service = service_factory(settings)
    monkeypatch.setattr(service, "enqueue_path", lambda p: enqueued.append(p))
    monkeypatch.setattr(service._metadata_cache, "refresh", lambda: None)

    inbox_file = settings.inbox_dir / "2026-07-10 clip.flac"  # type: ignore[operator]
    inbox_file.write_text("audio")

    service.process_one(inbox_file)

    assert not inbox_file.exists()
    assert len(enqueued) == 1
    assert enqueued[0].parent == settings.archive_dir
