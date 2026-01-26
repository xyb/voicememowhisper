"""
One-off helper to normalize transcript filenames to match audio stems.

Usage:
    python -m scripts.rename_transcripts --dry-run
    python -m scripts.rename_transcripts --apply

Behavior:
- Finds transcript files with leading timestamp prefixes that do not exactly
  match an audio stem in the Audio directory.
- Renames to <audio_stem>.txt when content differs but no conflict exists.
- If a target filename already exists:
    - If content is identical, deletes the misnamed file.
    - If content differs, leaves both in place and reports the conflict.
- Updates the SQLite state DB (processed.transcript_path) to the kept path.
"""

from __future__ import annotations

import argparse
import filecmp
import sqlite3
from pathlib import Path
import re


def hashable_path(p: Path) -> str:
    return str(p.expanduser())


def load_audio_stems(audio_dir: Path) -> set[str]:
    return {p.stem for p in audio_dir.glob("*.m4a")}


def normalize(
    audio_dir: Path,
    transcript_dir: Path,
    state_db: Path,
    apply: bool,
):
    pattern = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(.+)$")
    audio_stems = load_audio_stems(audio_dir)
    renames: list[tuple[Path, Path]] = []
    deletes: list[tuple[Path, Path]] = []
    conflicts: list[tuple[Path, Path]] = []

    for txt in sorted(transcript_dir.glob("*.txt")):
        stem = txt.stem
        candidate = stem
        while candidate not in audio_stems:
            m = pattern.match(candidate)
            if not m:
                candidate = None
                break
            candidate = m.group(2)
        if not candidate or candidate == stem or candidate not in audio_stems:
            continue

        target = txt.with_name(f"{candidate}.txt")
        if target.exists():
            try:
                same = filecmp.cmp(txt, target, shallow=False)
            except OSError:
                same = False
            if same:
                deletes.append((txt, target))
            else:
                conflicts.append((txt, target))
        else:
            renames.append((txt, target))

    if not apply:
        print(f"[dry-run] audio stems: {len(audio_stems)}")
        print(f"[dry-run] renames: {len(renames)}")
        for old, new in renames:
            print(f"  RENAME {old.name} -> {new.name}")
        print(f"[dry-run] deletes (duplicate content): {len(deletes)}")
        for old, keep in deletes:
            print(f"  DELETE {old.name} (keep {keep.name})")
        print(f"[dry-run] conflicts (different content, skipped): {len(conflicts)}")
        for old, tgt in conflicts:
            print(f"  CONFLICT {old.name} vs {tgt.name}")
        return

    # apply
    for old, new in renames:
        new.parent.mkdir(parents=True, exist_ok=True)
        old.rename(new)
    for old, _ in deletes:
        old.unlink(missing_ok=True)

    state_updates = 0
    if state_db.exists():
        conn = sqlite3.connect(state_db)
        for old, new in renames:
            cur = conn.execute(
                "UPDATE processed SET transcript_path=? WHERE transcript_path=?",
                (hashable_path(new), hashable_path(old)),
            )
            state_updates += cur.rowcount
        for old, keep in deletes:
            cur = conn.execute(
                "UPDATE processed SET transcript_path=? WHERE transcript_path=?",
                (hashable_path(keep), hashable_path(old)),
            )
            state_updates += cur.rowcount
        conn.commit()
        conn.close()

    print(f"[apply] renames applied: {len(renames)}")
    print(f"[apply] deletes applied: {len(deletes)}")
    print(f"[apply] conflicts skipped: {len(conflicts)}")
    print(f"[apply] state updates: {state_updates}")
    if conflicts:
        print("Conflicts (different content, left untouched):")
        for old, tgt in conflicts:
            print(f"  {old.name} vs {tgt.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", default="~/Documents/VoiceMemoWhisper/Audio")
    parser.add_argument("--transcript-dir", default="~/Documents/VoiceMemoWhisper/Transcripts")
    parser.add_argument("--state-db", default="~/.local/state/voicememowhisper/state.sqlite")
    parser.add_argument("--apply", action="store_true", help="Apply changes (default dry-run)")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir).expanduser()
    transcript_dir = Path(args.transcript_dir).expanduser()
    state_db = Path(args.state_db).expanduser()

    normalize(audio_dir, transcript_dir, state_db, apply=args.apply)


if __name__ == "__main__":
    main()
