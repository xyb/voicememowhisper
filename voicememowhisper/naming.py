from __future__ import annotations

import re
from datetime import datetime


def sanitize_filename(value: str) -> str:
    safe_chars = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_", " "):
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    return "".join(safe_chars).strip() or "untitled"


def normalize_title(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"[^\w]+", "", value).lower()


_TIMESTAMPED_STEM_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(.*)$")
_APPLE_DEFAULT_STEM_RE = re.compile(r"^\d{8}\s\d{6}$")


def parse_timestamped_stem(stem: str) -> tuple[str | None, str | None]:
    """
    Parse stems like: YYYY-MM-DD_HH-MM-SS_Title...
    Returns (timestamp_str, title_part) or (None, None) if not matched.
    """
    match = _TIMESTAMPED_STEM_RE.match(stem)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def title_from_stem(stem: str, fallback_title: str | None = None) -> str:
    """
    Derive a human-ish title from the filename stem.

    Rules:
    - If stem is already timestamped (YYYY-MM-DD_HH-MM-SS_...), use the title part after the timestamp.
    - If stem looks like Apple's default (YYYYMMDD HHMMSS), prefer fallback_title when provided.
    - Otherwise, use the stem as-is.
    """
    _ts, title_part = parse_timestamped_stem(stem)
    if title_part:
        return title_part
    if _APPLE_DEFAULT_STEM_RE.match(stem):
        return fallback_title or stem
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
