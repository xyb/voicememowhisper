"""
Validate that transcript/archived paths in the state DB exist on disk.

Usage:
    python -m scripts.check_state_paths
    python -m scripts.check_state_paths --state-db ~/.local/state/voicememowhisper/state.sqlite
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-db", default="~/.local/state/voicememowhisper/state.sqlite")
    args = parser.parse_args()

    state_db = Path(args.state_db).expanduser()
    if not state_db.exists():
        print(f"State DB not found: {state_db}")
        return

    conn = sqlite3.connect(state_db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT guid, transcript_path, archived_path FROM processed").fetchall()
    conn.close()

    missing_transcripts = []
    missing_archives = []

    for row in rows:
        guid = row["guid"]
        tpath = Path(row["transcript_path"]) if row["transcript_path"] else None
        apath = Path(row["archived_path"]) if row["archived_path"] else None

        if tpath and not tpath.exists():
            missing_transcripts.append((guid, tpath))
        if apath and not apath.exists():
            missing_archives.append((guid, apath))

    print(f"Total entries: {len(rows)}")
    print(f"Missing transcripts: {len(missing_transcripts)}")
    for guid, path in missing_transcripts:
        print(f"  {guid}: {path}")
    print(f"Missing archives: {len(missing_archives)}")
    for guid, path in missing_archives:
        print(f"  {guid}: {path}")


if __name__ == "__main__":
    main()
