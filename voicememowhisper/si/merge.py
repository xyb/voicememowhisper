#!/usr/bin/env python3
"""PoC 06: Merge transcript and diarization into a MergedTranscript.

Pure function: given a Transcript and a Diarization (both as JSON files
produced by stages 01 and 02), assign each transcript segment to the
speaker label with the highest time overlap (`max-overlap` strategy).

Identification (mapping anonymous labels to real names) is NOT done
here — that's stage 05. If `--identification <path>` is provided, the
resolved names are applied; otherwise segments keep anonymous labels
like SPEAKER_00, and every segment's speaker_name stays None.

Usage
-----
    python 06_merge.py \\
        --transcript runs/sample-recording/transcript_faster_whisper.json \\
        --diarization runs/sample-recording/diarization_pyannote.json \\
        --output runs

Optionally:
    --identification runs/sample-recording/identification.json \\
    --pipeline-name "faster_whisper + pyannote"

Outputs to runs/<recording_id>/:
    merged.json            (contracts.MergedTranscript)
    merged.meta.json       (contracts.StageMeta)
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import contracts


def peak_rss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Return the length of overlap between two time intervals, or 0."""
    lo = max(a_start, b_start)
    hi = min(a_end, b_end)
    return max(0.0, hi - lo)


def assign_speaker(
    seg_start: float,
    seg_end: float,
    diar_segments: list[contracts.SpeakerSegment],
) -> tuple[str, float, float, float]:
    """Return (label, top_share, margin_share, voiced_ratio).

    label:        diarization speaker with highest overlap, or 'UNKNOWN'.
    top_share:    top speaker's share of *voiced* time (overlap / sum of all overlaps).
                  1.0 = uncontested; 0.5 = tied with another speaker.
    margin_share: (top - second) / voiced.  Large = decisive win.
    voiced_ratio: voiced / segment_length.  Low = mostly silence in segment.

    Why not the old "overlap / segment_length"? Whisper segments often pad
    with silence/breath the diarizer didn't label as anyone. That made the
    ratio drop below 0.5 even when only one speaker was present. We now
    measure dominance over voiced time, which only flags real ambiguity.
    """
    seg_len = max(1e-6, seg_end - seg_start)
    best_label = "UNKNOWN"
    label_overlaps: dict[str, float] = {}
    for ds in diar_segments:
        if ds.end <= seg_start:
            continue
        if ds.start >= seg_end:
            break
        o = overlap(seg_start, seg_end, ds.start, ds.end)
        if o > 0:
            label_overlaps[ds.label] = label_overlaps.get(ds.label, 0.0) + o
    if not label_overlaps:
        return best_label, 0.0, 0.0, 0.0

    sorted_overlaps = sorted(label_overlaps.values(), reverse=True)
    voiced = sum(sorted_overlaps)
    top_o = sorted_overlaps[0]
    second_o = sorted_overlaps[1] if len(sorted_overlaps) > 1 else 0.0
    best_label = max(label_overlaps.items(), key=lambda kv: kv[1])[0]

    top_share = top_o / voiced if voiced > 0 else 0.0
    margin_share = (top_o - second_o) / voiced if voiced > 0 else 0.0
    voiced_ratio = voiced / seg_len
    return best_label, round(top_share, 3), round(margin_share, 3), round(voiced_ratio, 3)


def merge(
    transcript: contracts.Transcript,
    diarization: contracts.Diarization,
    identification: contracts.Identification | None,
    pipeline_name: str,
) -> contracts.MergedTranscript:
    # Sort diarization by start time once so we can exit early in the loop.
    diar_sorted = sorted(diarization.segments, key=lambda s: s.start)

    labels_used: set[str] = set()
    merged_segments: list[contracts.MergedSegment] = []

    for seg in transcript.segments:
        label, top_share, margin_share, voiced_ratio = assign_speaker(
            seg.start, seg.end, diar_sorted
        )
        labels_used.add(label)

        name: str | None = None
        confidence: float | None = None
        # Real ambiguity only: top speaker holds < 70% of voiced time AND
        # margin over the runner-up is < 20% of voiced time. Pure-silence
        # padding no longer triggers ⚠️ (silence_ratio handled separately).
        # UNKNOWN (no diarization overlap at all) always needs review.
        if label == "UNKNOWN":
            needs_review = True
        else:
            needs_review = top_share < 0.7 and margin_share < 0.2

        if identification is not None and label != "UNKNOWN":
            resolved_name, conf = identification.name_for(label)
            name = resolved_name
            confidence = conf
            if name is None:
                needs_review = True

        merged_segments.append(
            contracts.MergedSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                speaker_label=label,
                speaker_name=name,
                confidence=confidence,
                needs_review=needs_review,
            )
        )

    # Labels with no resolved name → unresolved list
    unresolved: list[str] = []
    if identification is not None:
        for label in sorted(labels_used):
            if label == "UNKNOWN":
                continue
            name, _ = identification.name_for(label)
            if name is None:
                unresolved.append(label)
    else:
        unresolved = sorted(labels_used - {"UNKNOWN"})

    return contracts.MergedTranscript(
        recording_id=transcript.recording_id,
        duration_sec=transcript.duration_sec,
        language=transcript.language,
        pipeline=pipeline_name,
        segments=merged_segments,
        unresolved_labels=unresolved,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--transcript", required=True, type=Path)
    ap.add_argument("--diarization", required=True, type=Path)
    ap.add_argument(
        "--identification",
        type=Path,
        default=None,
        help="optional Identification JSON; if omitted, anonymous labels are kept",
    )
    ap.add_argument(
        "--pipeline-name",
        default=None,
        help="human-readable pipeline description; auto-composed if omitted",
    )
    ap.add_argument(
        "--output",
        default="runs",
        type=Path,
        help="output root directory (will write to <output>/<recording_id>/)",
    )
    args = ap.parse_args()

    for p in (args.transcript, args.diarization):
        if not p.exists():
            print(f"error: not found: {p}", file=sys.stderr)
            return 2

    transcript = contracts.Transcript.from_json(args.transcript)
    diarization = contracts.Diarization.from_json(args.diarization)
    identification: contracts.Identification | None = None
    if args.identification:
        if not args.identification.exists():
            print(f"error: not found: {args.identification}", file=sys.stderr)
            return 2
        identification = contracts.Identification.from_json(args.identification)

    if transcript.recording_id != diarization.recording_id:
        print(
            f"error: recording_id mismatch: "
            f"transcript={transcript.recording_id} diarization={diarization.recording_id}",
            file=sys.stderr,
        )
        return 2

    pipeline_name = args.pipeline_name or (
        f"{transcript.backend}/{transcript.model} + {diarization.backend}"
        + (f" + {identification.backend}" if identification else " (anonymous)")
    )

    out_dir = args.output / transcript.recording_id
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_path = out_dir / "merged.json"
    meta_path = out_dir / "merged.meta.json"

    print(f"[06_merge] transcript:   {args.transcript}")
    print(f"[06_merge] diarization:  {args.diarization}")
    print(
        f"[06_merge] identification:"
        f" {args.identification if args.identification else '(none, anonymous labels)'}"
    )
    print(f"[06_merge] output:       {out_dir}")

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    merged = merge(transcript, diarization, identification, pipeline_name)
    elapsed = time.perf_counter() - t0

    merged.to_json(merged_path)

    meta = contracts.StageMeta(
        stage_id="06_merge",
        backend="max_overlap",
        model="n/a",
        started_at=started_at,
        elapsed_sec=round(elapsed, 4),
        peak_rss_bytes=peak_rss_bytes(),
        config={
            "transcript_backend": transcript.backend,
            "transcript_model": transcript.model,
            "diarization_backend": diarization.backend,
            "diarization_model": diarization.model,
            "identification_backend": identification.backend if identification else None,
            "pipeline_name": pipeline_name,
        },
    )
    meta.to_json(meta_path)

    print()
    print(f"[06_merge] done in {elapsed * 1000:.1f} ms")
    print(f"[06_merge] segments merged:  {len(merged.segments)}")
    print(f"[06_merge] labels used:      {sorted({s.speaker_label for s in merged.segments})}")
    print(f"[06_merge] unresolved:       {merged.unresolved_labels}")
    review_count = sum(1 for s in merged.segments if s.needs_review)
    print(f"[06_merge] needs_review:     {review_count} / {len(merged.segments)}")
    print(f"[06_merge] merged →  {merged_path}")
    print(f"[06_merge] meta    →  {meta_path}")

    if merged.segments:
        print()
        print("[06_merge] first 5 merged segments:")
        for s in merged.segments[:5]:
            name = s.speaker_name or s.speaker_label
            flag = " ⚠️" if s.needs_review else ""
            print(f"  [{s.start:7.2f} {s.end:7.2f}] {name}{flag}  {s.text}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
