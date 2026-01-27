from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import voicememowhisper.metadata as md
from voicememowhisper.metadata import VoiceMemo


def test_resolve_created_at_prefers_metadata_created_at(settings) -> None:
    p = settings.recordings_dir / "foo.m4a"
    p.write_text("a")
    os_dt = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    memo = VoiceMemo(guid="foo", path=p, created_at=os_dt)
    resolved = md.resolve_created_at(memo)
    assert resolved is not None
    assert resolved.tzinfo is not None
    assert resolved.astimezone(timezone.utc) == os_dt


def test_list_voice_memos_includes_metadata_only_entries(settings) -> None:
    (settings.recordings_dir / "foo.m4a").write_text("a")
    (settings.recordings_dir / "bar.m4a").write_text("b")

    memos = {
        "foo": VoiceMemo(
            guid="foo",
            path=settings.recordings_dir / "foo.m4a",
            title="Foo",
            created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
            is_trashed=False,
        ),
        # metadata-only: not present on disk, but should be included
        "baz": VoiceMemo(
            guid="baz",
            path=settings.recordings_dir / "baz.m4a",
            title="Baz",
            created_at=datetime(2026, 1, 3, tzinfo=timezone.utc),
            is_trashed=False,
        ),
        # trashed should be excluded
        "trash": VoiceMemo(
            guid="trash",
            path=settings.recordings_dir / "trash.m4a",
            title="Trash",
            created_at=datetime(2026, 1, 4, tzinfo=timezone.utc),
            is_trashed=True,
        ),
    }

    with patch.object(md, "load_voice_memos", lambda _settings: memos):
        results = md.list_voice_memos(settings)

    titles = {m.title for m in results}
    assert "Foo" in titles
    assert "Baz" in titles
    assert "Trash" not in titles

