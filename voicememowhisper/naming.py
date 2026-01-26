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
