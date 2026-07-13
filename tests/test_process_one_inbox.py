from __future__ import annotations

# pyright: reportMissingImports=false

"""`process_one` (the positional-path entry point) must consume Inbox files.

The batch scan moves an Inbox file into the archive *before* enqueueing it, so
the Inbox drains. `process_one` used to enqueue the Inbox path directly, which
sent it down the ArchiveManager *copy* path (correct for Voice Memos sources,
which must never be moved) and left the original sitting in the Inbox forever.
"""

from pathlib import Path


def _stub_run(service, monkeypatch) -> list[Path]:
    enqueued: list[Path] = []
    monkeypatch.setattr(service, "enqueue_path", lambda p: enqueued.append(p))
    monkeypatch.setattr(service._metadata_cache, "refresh", lambda: None)
    return enqueued


def test_process_one_moves_inbox_file_into_archive(service_factory, settings, monkeypatch) -> None:
    service = service_factory(settings)
    enqueued = _stub_run(service, monkeypatch)

    inbox_file = settings.inbox_dir / "2024-01-02 foo.m4a"  # type: ignore[operator]
    inbox_file.write_text("hello")

    service.process_one(inbox_file)

    assert not inbox_file.exists(), "Inbox file must be consumed, not left behind"
    assert len(enqueued) == 1
    archived = enqueued[0]
    assert archived.parent == settings.archive_dir
    assert archived.exists()
    assert archived.read_text() == "hello"


def test_process_one_leaves_voice_memos_source_in_place(service_factory, settings, monkeypatch) -> None:
    """Voice Memos sources live in a library we must not mutate — copy, never move."""
    service = service_factory(settings)
    enqueued = _stub_run(service, monkeypatch)

    source = settings.recordings_dir / "20240102 030405.m4a"
    source.write_text("hello")

    service.process_one(source)

    assert source.exists(), "Voice Memos source must stay put"
    assert enqueued == [source.resolve()]


def test_process_one_on_already_archived_file_is_a_noop(service_factory, settings, monkeypatch) -> None:
    service = service_factory(settings)
    enqueued = _stub_run(service, monkeypatch)

    archived = settings.archive_dir / "2024-01-02_03-04-05_foo.m4a"  # type: ignore[operator]
    archived.write_text("hello")

    service.process_one(archived)

    assert archived.exists()
    assert enqueued == [archived.resolve()]
