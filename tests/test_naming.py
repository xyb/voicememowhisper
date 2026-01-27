from __future__ import annotations

from datetime import datetime, timezone

from voicememowhisper.naming import (
    dedup_key,
    normalize_title,
    parse_timestamped_stem,
    sanitize_filename,
    title_from_stem,
    to_naive,
)


def test_sanitize_filename() -> None:
    assert sanitize_filename("a/b") == "a_b"
    # Use synthetic titles in tests; never include real memo titles/names.
    assert sanitize_filename("面试：示例候选人，数据工程师") == "面试：示例候选人，数据工程师"
    assert sanitize_filename("  ") == "untitled"


def test_normalize_title() -> None:
    assert normalize_title("Hello, World!") == "helloworld"
    expected = "面试示例候选人数据工程师"
    assert normalize_title("面试_示例候选人_数据工程师") == expected
    assert normalize_title("面试 示例候选人 数据工程师") == expected
    assert normalize_title("面试：示例候选人，数据工程师") == expected
    assert normalize_title(None) == ""


def test_parse_timestamped_stem() -> None:
    ts, title = parse_timestamped_stem("2026-01-27_09-30-03_meeting")
    assert ts == "2026-01-27_09-30-03"
    assert title == "meeting"

    ts2, title2 = parse_timestamped_stem("plain")
    assert ts2 is None
    assert title2 is None


def test_title_from_stem() -> None:
    assert title_from_stem("2026-01-27_09-30-03_meeting") == "meeting"
    assert title_from_stem("foo") == "foo"


def test_to_naive() -> None:
    aware = datetime(2026, 1, 1, tzinfo=timezone.utc)
    naive = to_naive(aware)
    assert naive is not None
    assert naive.tzinfo is None


def test_dedup_key() -> None:
    dt = datetime(2026, 1, 1, 1, 2, 3, tzinfo=timezone.utc)
    when, title = dedup_key(dt, "Hello!")
    assert when == "2026-01-01 01:02:03"
    assert title == "hello"

