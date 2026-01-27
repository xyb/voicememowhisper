from __future__ import annotations

import re
from datetime import datetime


def sanitize_filename(value: str) -> str:
    """
    Sanitize a string for use as a filename.

    Policy: keep Unicode punctuation (e.g. Chinese '：，') intact; only replace
    characters that are invalid or commonly problematic in POSIX paths:
    - path separator '/' and NUL
    - ASCII control chars
    """
    if not value:
        return "untitled"

    out: list[str] = []
    for ch in value:
        code = ord(ch)
        if ch == "/" or code == 0:
            out.append("_")
            continue
        if code < 32 or code == 127:
            out.append("_")
            continue
        out.append(ch)

    sanitized = "".join(out).strip()
    return sanitized or "untitled"


def normalize_title(value: str | None) -> str:
    if not value:
        return ""
    # Treat underscores as separators (they may come from previous sanitization),
    # so remove them too.
    return re.sub(r"[\W_]+", "", value).lower()


_TIMESTAMPED_STEM_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(.*)$")


def parse_timestamped_stem(stem: str) -> tuple[str | None, str | None]:
    """
    Parse stems like: YYYY-MM-DD_HH-MM-SS_Title...
    Returns (timestamp_str, title_part) or (None, None) if not matched.
    """
    match = _TIMESTAMPED_STEM_RE.match(stem)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def title_from_stem(stem: str) -> str:
    """
    Derive a human-ish title from the filename stem.

    Rules:
    - If stem is already timestamped (YYYY-MM-DD_HH-MM-SS_...), use the title part after the timestamp.
    - Otherwise, use the stem as-is.
    """
    _ts, title_part = parse_timestamped_stem(stem)
    if title_part:
        return title_part
    return stem


def to_naive(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt
    return dt.replace(tzinfo=None)


def dedup_key(created_at: datetime | None, title: str | None) -> tuple[str, str]:
    when = ""
    if created_at:
        naive = to_naive(created_at)
        if naive:
            when = naive.strftime("%Y-%m-%d %H:%M:%S")
    return when, normalize_title(title)
