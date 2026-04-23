#!/usr/bin/env python3
"""PoC 05: Map anonymous diarization labels to real speaker names.

For each anonymous label in a Diarization (e.g. SPEAKER_00), compute or
reuse a centroid embedding, then match against all centroids in the
speaker-library via cosine similarity. Assign the best match whose
cosine is above `--threshold`; otherwise leave as unknown.

Usage
-----
    python 05_identify.py \\
        --diarization runs/sample-recording/diarization_pyannote.json \\
        --library speaker-library \\
        --threshold 0.5 \\
        --output runs

Options
-------
    --audio <path>         Re-compute centroids from audio instead of
                           reusing runs/<id>/diarization_pyannote.embeddings.npz
    --threshold <float>    Cosine threshold for a match (default 0.5)

Output:
    runs/<recording_id>/identification.json      (contracts.Identification)
    runs/<recording_id>/identification.meta.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import contracts
from .library import is_speaker_dir
from .speaker_embed import CommunityOneEmbedder, cosine_similarity


def load_library(library_dir: Path) -> dict[str, tuple[np.ndarray, str]]:
    """Return {speaker_id: (centroid_1d, display_name)}."""
    library: dict[str, tuple[np.ndarray, str]] = {}
    for speaker_dir in sorted(library_dir.iterdir()):
        if not is_speaker_dir(speaker_dir):
            continue
        emb_path = speaker_dir / "embedding.npy"
        if not emb_path.exists():
            continue
        centroid = np.load(str(emb_path))
        profile_path = speaker_dir / "profile.json"
        display_name = speaker_dir.name
        if profile_path.exists():
            try:
                profile = json.loads(profile_path.read_text(encoding="utf-8"))
                display_name = profile.get("display_name", speaker_dir.name)
            except Exception:
                pass
        library[speaker_dir.name] = (centroid, display_name)
    return library


def get_label_centroids_from_npz(
    diar: contracts.Diarization,
    npz_path: Path,
) -> dict[str, np.ndarray]:
    """Read per-speaker centroids from 02_diarize_pyannote's .embeddings.npz."""
    if not npz_path.exists():
        return {}
    data = np.load(str(npz_path))
    return {k: np.asarray(data[k]) for k in data.files}


def compute_label_centroids_from_audio(
    diar: contracts.Diarization,
    audio_path: Path,
    embedder: CommunityOneEmbedder,
    min_segment_sec: float = 0.5,
) -> dict[str, np.ndarray]:
    """Re-compute per-label centroids directly from audio.

    For each label, pick segments above min_segment_sec, embed each,
    average into a centroid, L2 normalize.
    """
    segs_by_label: dict[str, list[tuple[float, float]]] = {}
    for s in diar.segments:
        if s.end - s.start >= min_segment_sec:
            segs_by_label.setdefault(s.label, []).append((s.start, s.end))

    centroids: dict[str, np.ndarray] = {}
    for label, spans in segs_by_label.items():
        # Cap to a reasonable number to keep cost bounded
        spans = spans[:10]
        vecs = []
        for start, end in spans:
            try:
                v = embedder.embed_audio(audio_path, start=start, end=end)
                vecs.append(v)
            except ValueError:
                continue
        if not vecs:
            continue
        centroid = np.stack(vecs, axis=0).mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids[label] = centroid
    return centroids


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def match_labels(
    label_centroids: dict[str, np.ndarray],
    library: dict[str, tuple[np.ndarray, str]],
    threshold: float,
) -> list[contracts.Assignment]:
    """For each label, find best library match. Assign only if above threshold."""
    assignments: list[contracts.Assignment] = []
    for label in sorted(label_centroids.keys()):
        lvec = normalize(label_centroids[label])
        best_name: str | None = None
        best_score: float = -1.0
        for speaker_id, (centroid, display_name) in library.items():
            score = cosine_similarity(lvec, normalize(centroid))
            if score > best_score:
                best_score = score
                best_name = display_name
        if best_score >= threshold:
            assignments.append(
                contracts.Assignment(
                    label=label,
                    name=best_name,
                    confidence=round(best_score, 4),
                    source="library_match",
                )
            )
        else:
            assignments.append(
                contracts.Assignment(
                    label=label,
                    name=None,
                    confidence=round(best_score, 4) if best_score > -1 else None,
                    source="unknown",
                )
            )
    return assignments


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--diarization", required=True, type=Path)
    ap.add_argument("--library", type=Path, default=Path("speaker-library"))
    ap.add_argument("--audio", type=Path, default=None,
                    help="if given, re-compute label centroids from audio (slower but bit-independent of 02)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--output", default="runs", type=Path)
    args = ap.parse_args()

    if not args.diarization.exists():
        print(f"error: not found: {args.diarization}", file=sys.stderr)
        return 2

    diar = contracts.Diarization.from_json(args.diarization)
    library = load_library(args.library)
    if not library:
        print(f"error: empty library at {args.library}", file=sys.stderr)
        return 2

    print(f"[05_identify] diarization: {args.diarization}")
    print(f"[05_identify] library:     {args.library} ({len(library)} speakers)")
    print(f"[05_identify] threshold:   {args.threshold}")

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()

    if args.audio:
        print(f"[05_identify] recomputing centroids from {args.audio}")
        embedder = CommunityOneEmbedder()
        _ = embedder.dimension
        label_centroids = compute_label_centroids_from_audio(diar, args.audio, embedder)
    else:
        # Reuse the per-speaker centroids saved by 02_diarize_pyannote
        npz_path = args.diarization.parent / "diarization_pyannote.embeddings.npz"
        label_centroids = get_label_centroids_from_npz(diar, npz_path)
        if not label_centroids:
            print(
                f"error: no centroids at {npz_path}. "
                f"Pass --audio to recompute from audio.",
                file=sys.stderr,
            )
            return 2
        print(f"[05_identify] reusing centroids from {npz_path}")

    assignments = match_labels(label_centroids, library, args.threshold)
    elapsed = time.perf_counter() - t0

    ident = contracts.Identification(
        recording_id=diar.recording_id,
        backend="community_one_embedding_cosine",
        model="pyannote/speaker-diarization-community-1#_embedding",
        threshold=args.threshold,
        assignments=assignments,
    )

    out_dir = args.output / diar.recording_id
    out_dir.mkdir(parents=True, exist_ok=True)
    ident_path = out_dir / "identification.json"
    meta_path = out_dir / "identification.meta.json"

    ident.to_json(ident_path)
    meta = contracts.StageMeta(
        stage_id="05_identify",
        backend="community_one_embedding_cosine",
        model="pyannote/speaker-diarization-community-1#_embedding",
        started_at=started_at,
        elapsed_sec=round(elapsed, 2),
        peak_rss_bytes=None,
        config={
            "threshold": args.threshold,
            "num_labels": len(label_centroids),
            "num_library_speakers": len(library),
            "centroid_source": "audio" if args.audio else "npz_cache",
        },
    )
    meta.to_json(meta_path)

    print()
    print(f"[05_identify] done in {elapsed:.1f}s")
    print(f"[05_identify] assignments: {len(assignments)}")
    for a in assignments:
        name = a.name or "(unknown)"
        conf = f"{a.confidence:.3f}" if a.confidence is not None else "n/a"
        print(f"  {a.label:12s} → {name:20s}  cosine={conf}  [{a.source}]")
    print(f"[05_identify] identification → {ident_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
