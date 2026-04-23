#!/usr/bin/env python3
"""PoC 04: Build / update the speaker-library centroid cache.

Given a speaker-library directory with per-speaker clips, compute a
centroid embedding for each speaker and dump it as an .npy cache next
to the clips. Also writes/updates speaker-library/index.json.

Directory layout expected:

    speaker-library/
      <speaker_id>/
        profile.json                 (display_name, aliases, notes)
        clips/
          001.wav                    (audio clip)
          001.json                   (optional: source recording, time range)
          002.wav
          ...
        embedding.npy                (OUTPUT: centroid 1-D vector)
        embedding.meta.json          (OUTPUT: backend, model, dimension)

Usage
-----
    python 04_library.py --library speaker-library
    python 04_library.py --library speaker-library --speaker some-speaker

`--speaker` limits processing to one speaker; default is all speakers
with at least one clip.

Notes
-----
- The embedder is the one from pyannote speaker-diarization-community-1.
- Loading cost is ~30s; amortized across all clips in one run.
- Clips shorter than embedder.min_duration are skipped with a warning.
- `embedding.npy` is a re-runnable cache: delete it to force recompute.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .speaker_embed import CommunityOneEmbedder


# Top-level subdirectories under speaker-library/ that are NOT speakers and
# must be skipped by all enrollment / identify / index logic. Currently empty:
# badcases live inside each speaker dir as `<speaker>/badcases/<id>/`, so the
# top level only contains real speakers. Kept as a hook for future reserved
# names (templates, tests, …) — adding a name here uniformly excludes it from
# every iteration site that calls `is_speaker_dir`.
RESERVED_SUBDIRS: frozenset[str] = frozenset()


def is_speaker_dir(p: Path) -> bool:
    """True if `p` is a speaker directory (i.e. not a reserved subdir)."""
    return p.is_dir() and not p.name.startswith(".") and p.name not in RESERVED_SUBDIRS


def load_or_init_profile(speaker_dir: Path) -> dict:
    profile_path = speaker_dir / "profile.json"
    if profile_path.exists():
        return json.loads(profile_path.read_text(encoding="utf-8"))
    return {
        "speaker_id": speaker_dir.name,
        "display_name": speaker_dir.name,
        "aliases": [],
        "notes": "",
    }


def list_clip_wavs(speaker_dir: Path) -> list[Path]:
    clips_dir = speaker_dir / "clips"
    if not clips_dir.exists():
        return []
    return sorted(
        p
        for p in clips_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".wav", ".m4a", ".flac", ".mp3", ".ogg"}
    )


def compute_centroid(
    embedder: CommunityOneEmbedder,
    clips: list[Path],
    speaker_label: str,
) -> np.ndarray | None:
    """Compute the centroid (mean) of embeddings from multiple clips.

    Skip clips shorter than the embedder's minimum duration.
    Returns None if no usable clip.
    """
    vecs: list[np.ndarray] = []
    for clip in clips:
        try:
            v = embedder.embed_audio(clip)
        except ValueError as e:
            print(
                f"[04_library]   skip {speaker_label}/{clip.name}: {e}",
                file=sys.stderr,
            )
            continue
        vecs.append(v)
    if not vecs:
        return None
    stacked = np.stack(vecs, axis=0)
    centroid = stacked.mean(axis=0)
    # L2 normalize (optional but matches pyannote convention for cosine)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    return centroid


def process_speaker(
    embedder: CommunityOneEmbedder,
    speaker_dir: Path,
) -> dict | None:
    """Compute and dump centroid for one speaker. Returns index entry dict."""
    clips = list_clip_wavs(speaker_dir)
    if not clips:
        print(f"[04_library] skip {speaker_dir.name}: no clips")
        return None

    profile = load_or_init_profile(speaker_dir)
    print(
        f"[04_library] {profile.get('display_name', speaker_dir.name)} "
        f"({len(clips)} clips)"
    )
    centroid = compute_centroid(embedder, clips, speaker_dir.name)
    if centroid is None:
        print(f"[04_library]   no usable clip, skip")
        return None

    emb_path = speaker_dir / "embedding.npy"
    meta_path = speaker_dir / "embedding.meta.json"
    profile_path = speaker_dir / "profile.json"

    np.save(str(emb_path), centroid)
    meta = {
        "speaker_id": speaker_dir.name,
        "backend": "pyannote_community_one_embedding",
        "model": "pyannote/speaker-diarization-community-1#_embedding",
        "dimension": int(centroid.shape[0]),
        "num_clips": len(clips),
        "l2_normalized": True,
        "metric": "cosine",
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return {
        "speaker_id": speaker_dir.name,
        "display_name": profile.get("display_name", speaker_dir.name),
        "clip_count": len(clips),
        "dimension": int(centroid.shape[0]),
    }


def run_build(library_dir: Path, speaker_filter: str | None = None) -> int:
    """Programmatic entry: rebuild centroids for one or all speakers.

    Returns process-style exit code (0 success, non-zero error).
    """
    library_dir = Path(library_dir)
    if not library_dir.exists():
        print(f"error: library not found: {library_dir}", file=sys.stderr)
        return 2

    speaker_dirs: list[Path] = []
    if speaker_filter:
        d = library_dir / speaker_filter
        if not d.is_dir():
            print(f"error: speaker dir not found: {d}", file=sys.stderr)
            return 2
        speaker_dirs.append(d)
    else:
        speaker_dirs = sorted(d for d in library_dir.iterdir() if is_speaker_dir(d))

    if not speaker_dirs:
        print("[04_library] no speaker directories found")
        return 0

    print(f"[04_library] library:  {library_dir}")
    print(f"[04_library] speakers: {[d.name for d in speaker_dirs]}")
    print(f"[04_library] loading embedder (this takes ~30s)...")
    t0 = time.perf_counter()
    embedder = CommunityOneEmbedder()
    # Force load so the cost is attributed to the loading phase only.
    _ = embedder.dimension
    print(f"[04_library] embedder loaded in {time.perf_counter() - t0:.1f}s")
    print(
        f"[04_library] dim={embedder.dimension}, sr={embedder.sample_rate}, "
        f"min_dur={embedder.min_duration_sec:.2f}s"
    )
    print()

    index_entries: list[dict] = []
    for d in speaker_dirs:
        entry = process_speaker(embedder, d)
        if entry is not None:
            index_entries.append(entry)

    # Merge into library-level index.json (preserve other speakers if any)
    index_path = library_dir / "index.json"
    existing = {"version": 1, "speakers": []}
    if index_path.exists():
        try:
            existing = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    by_id = {s["speaker_id"]: s for s in existing.get("speakers", [])}
    for e in index_entries:
        by_id[e["speaker_id"]] = e
    existing["speakers"] = sorted(by_id.values(), key=lambda s: s["speaker_id"])
    index_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print()
    print(f"[04_library] index → {index_path}")
    print(f"[04_library] {len(index_entries)} speaker(s) updated")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", type=Path, default=Path("speaker-library"))
    ap.add_argument(
        "--speaker",
        default=None,
        help="only process this speaker_id (directory name)",
    )
    args = ap.parse_args()
    return run_build(args.library, args.speaker)


if __name__ == "__main__":
    sys.exit(main())
