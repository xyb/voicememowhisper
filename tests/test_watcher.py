from __future__ import annotations

from pathlib import Path

import voicememowhisper.watcher as w


class _Event:
    def __init__(self, src_path: str, *, is_directory: bool = False, dest_path: str | None = None) -> None:
        self.src_path = src_path
        self.dest_path = dest_path
        self.is_directory = is_directory


def test_recording_handler_ignores_directories() -> None:
    seen: list[Path] = []

    def cb(p: Path) -> None:
        seen.append(p)

    h = w.RecordingHandler(cb, extensions=(".m4a",))
    h._handle_event(_Event("/tmp/x.m4a", is_directory=True))
    assert seen == []


def test_recording_handler_filters_extensions_case_insensitive() -> None:
    seen: list[Path] = []

    def cb(p: Path) -> None:
        seen.append(p)

    h = w.RecordingHandler(cb, extensions=(".m4a", ".mp3"))
    h._handle_event(_Event("/tmp/a.M4A"))
    h._handle_event(_Event("/tmp/b.mp3"))
    h._handle_event(_Event("/tmp/c.wav"))
    assert seen == [Path("/tmp/a.M4A"), Path("/tmp/b.mp3")]


def test_recording_handler_prefers_dest_path_for_moved_events() -> None:
    seen: list[Path] = []

    def cb(p: Path) -> None:
        seen.append(p)

    h = w.RecordingHandler(cb, extensions=(".m4a",))
    h._handle_event(_Event("/tmp/src.tmp", dest_path="/tmp/dest.m4a"))
    assert seen == [Path("/tmp/dest.m4a")]


def test_start_watcher_schedules_handler(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeObserver:
        def schedule(self, handler, directory, recursive=False):
            captured["handler"] = handler
            captured["directory"] = directory
            captured["recursive"] = recursive

        def start(self):
            captured["started"] = True

    monkeypatch.setattr(w, "Observer", FakeObserver)

    def cb(_p: Path) -> None:
        return

    obs = w.start_watcher(Path("/tmp"), cb, extensions=(".m4a", ".mp3"))
    assert isinstance(obs, FakeObserver)
    assert captured["directory"] == "/tmp"
    assert captured["recursive"] is False
    assert captured.get("started") is True
    assert isinstance(captured["handler"], w.RecordingHandler)

