#!/usr/bin/env python3
"""PoC 07: Render a MergedTranscript into a human-readable Markdown file.

Format: frontmatter with metadata, then one block per speaker turn.
Consecutive segments by the same speaker are combined into a single
block. Low-confidence / low-overlap segments are flagged with ⚠️.

Usage
-----
    python 07_render.py \\
        --merged runs/sample-recording/merged.json \\
        --output outputs

Outputs:
    outputs/<recording_id>/transcript.md
    outputs/<recording_id>/merged.json     (copied from runs for archival)

If --merged is omitted, looks for runs/<recording_id>/merged.json
under the provided --output's sibling runs/ directory.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import contracts


def fmt_timestamp(seconds: float, total_duration: float) -> str:
    """mm:ss by default, hh:mm:ss when total duration >= 1h."""
    s = int(round(seconds))
    if total_duration >= 3600:
        h = s // 3600
        m = (s % 3600) // 60
        sec = s % 60
        return f"{h:02d}:{m:02d}:{sec:02d}"
    m = s // 60
    sec = s % 60
    return f"{m:02d}:{sec:02d}"


def fmt_duration_human(seconds: float) -> str:
    s = int(round(seconds))
    if s >= 3600:
        h = s // 3600
        m = (s % 3600) // 60
        sec = s % 60
        return f"{h:02d}:{m:02d}:{sec:02d}"
    m = s // 60
    sec = s % 60
    return f"{m:02d}:{sec:02d}"


def group_consecutive(segments: list[contracts.MergedSegment]) -> list[list[contracts.MergedSegment]]:
    """Group segments by consecutive same speaker_label.

    Identification maps label → name, but consecutive detection happens at
    the label level: a label change is a real turn boundary. Within a
    group we use the first non-None speaker_name if any.
    """
    groups: list[list[contracts.MergedSegment]] = []
    current: list[contracts.MergedSegment] = []
    last_label: str | None = None
    for seg in segments:
        if seg.speaker_label != last_label and current:
            groups.append(current)
            current = []
        current.append(seg)
        last_label = seg.speaker_label
    if current:
        groups.append(current)
    return groups


def render_block(group: list[contracts.MergedSegment], total_duration: float) -> str:
    start = group[0].start
    label = group[0].speaker_label
    # Prefer resolved name if available, fall back to anonymous label
    name = next((s.speaker_name for s in group if s.speaker_name), None) or label
    # Show ⚠️ if any segment in the group needs review
    flag = " ⚠️" if any(s.needs_review for s in group) else ""
    ts = fmt_timestamp(start, total_duration)
    # Combine texts. Drop empty segments, trim whitespace.
    texts = [s.text.strip() for s in group if s.text.strip()]
    body = "".join(texts) if any("。" in t or "," in t or "?" in t for t in texts) else " ".join(texts)
    return f"**[{ts}] {flag.strip()}{' ' if flag else ''}{name}**\n{body}"


def render(merged: contracts.MergedTranscript, generated_at: str) -> str:
    # Frontmatter
    duration_human = fmt_duration_human(merged.duration_sec)
    lines: list[str] = ["---"]
    lines.append(f"recording: {merged.recording_id}")
    lines.append(f"duration: {duration_human}")
    lines.append(f"pipeline: {merged.pipeline}")
    lines.append(f"generated: {generated_at}")
    lines.append(f"language: {merged.language}")
    labels = sorted({s.speaker_label for s in merged.segments})
    lines.append(f"speakers: {labels}")
    if merged.unresolved_labels:
        lines.append(f"unresolved_speakers: {merged.unresolved_labels}")
    review_count = sum(1 for s in merged.segments if s.needs_review)
    if review_count:
        lines.append(f"needs_review_segments: {review_count}")
    lines.append("---")
    lines.append("")

    # Body: speaker blocks separated by blank lines
    groups = group_consecutive(merged.segments)
    for group in groups:
        lines.append(render_block(group, merged.duration_sec))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--merged", required=True, type=Path, help="merged.json input")
    ap.add_argument(
        "--output",
        default="outputs",
        type=Path,
        help="output root directory (will write to <output>/<recording_id>/)",
    )
    ap.add_argument(
        "--copy-merged",
        action="store_true",
        default=True,
        help="also copy merged.json into outputs/ for self-contained archival",
    )
    args = ap.parse_args()

    if not args.merged.exists():
        print(f"error: not found: {args.merged}", file=sys.stderr)
        return 2

    merged = contracts.MergedTranscript.from_json(args.merged)
    out_dir = args.output / merged.recording_id
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / "transcript.md"
    copy_path = out_dir / "merged.json"

    generated_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    md = render(merged, generated_at)
    md_path.write_text(md, encoding="utf-8")

    if args.copy_merged:
        shutil.copy2(args.merged, copy_path)

    print(f"[07_render] merged:      {args.merged}")
    print(f"[07_render] output_dir:  {out_dir}")
    print(f"[07_render] transcript → {md_path}")
    if args.copy_merged:
        print(f"[07_render] merged.json → {copy_path}")
    print()
    print(f"[07_render] {len(merged.segments)} segments → "
          f"{len(group_consecutive(merged.segments))} speaker blocks")
    print(f"[07_render] file size: {md_path.stat().st_size / 1024:.1f} KB")

    # Preview first 20 lines
    print()
    print("[07_render] first 20 lines:")
    for i, line in enumerate(md.splitlines()[:20]):
        print(f"  {line}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
