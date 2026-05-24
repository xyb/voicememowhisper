"""`voicememo-whisper si` subcommand group.

User-facing entry into the speaker-ID pipeline. Each pipeline stage
gets its own subcommand; the description in `--help` shows the step
number, so the processing order is visible at the command line without
baking position numbers into the subcommand name itself.

Subcommands:
    transcribe   Step 1/5
    diarize      Step 2/5
    identify     Step 3/5
    merge        Step 4/5
    render       Step 5/5
    run          Run a contiguous range of steps (--from / --to / --force)
    library      Manage the speaker-library (list / add / rebuild)
    inspect      Show cached intermediate state for a recording
    steps        Print the ordered list of pipeline steps and exit

The heavy ML imports (faster-whisper, pyannote.audio, torch) are
deferred to the actual step invocation, so `si --help`, `si steps`,
and `si library list` work even before the optional `[speaker-id]`
extra is installed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Lightweight imports only at module top.
from .pipeline import STEPS, STEP_BY_NAME, STEP_BY_NUM, STEP_NAMES, resolve_step

DEFAULT_RUNS = Path.home() / ".local" / "share" / "voicememowhisper" / "speaker-id" / "runs"
DEFAULT_OUTPUT = Path.home() / ".local" / "share" / "voicememowhisper" / "speaker-id" / "outputs"
# Speaker library lives with the other durable assets (Audio/ / Transcripts/)
# so a single backup of ~/Documents/VoiceMemoWhisper/ covers everything that
# can't be regenerated. runs/outputs stay under ~/.local/share/ — they're
# cache: rebuildable from speaker-library + the source audio.
DEFAULT_LIBRARY = Path.home() / "Documents" / "VoiceMemoWhisper" / "speaker-library"


def _step_help(num: int) -> str:
    name = STEP_BY_NUM[num]
    desc = next(d for n, _, d in STEPS if n == num)
    return f"Step {num}/{len(STEPS)} — {desc}"


# ───────── Subcommand handlers ────────────────────────────────────────


def _build_asr_backend_kwargs(args: argparse.Namespace) -> dict:
    """Turn the --asr-* CLI flags (plus an optional TOML config file)
    into kwargs for pipeline.run.

    Precedence: CLI flags > config file > built-in defaults. A CLI
    flag that the user didn't pass stays ``None`` and doesn't override
    the config.

    Returns a dict with ``asr_backend`` and optionally
    ``asr_backend_config``. Default backend is ``faster_whisper`` —
    ``asr_backend_config`` is only populated when the effective backend
    is an HTTP one.
    """
    from .asr_config import load_asr_config, build_pipeline_kwargs

    # Base layer: config file (may be empty).
    merged: dict = load_asr_config(getattr(args, "asr_config", None))

    # Overlay: every --asr-* / --diarize-* flag the user actually passed.
    for flag in (
        "asr_backend",
        "asr_url",
        "asr_model",
        "asr_api_key",
        "asr_host_header",
        "asr_response_format",
        "asr_timeout_sec",
        "asr_ws_url",
        "asr_ws_idle_timeout_sec",
        "asr_ws_connect_timeout_sec",
        "diarize_backend",
        "diarize_url",
        "diarize_api_key",
        "diarize_host_header",
        "diarize_timeout_sec",
        "diarize_include_embeddings",
        "diarize_num_speakers",
        "diarize_min_speakers",
        "diarize_max_speakers",
    ):
        val = getattr(args, flag, None)
        if val is not None:
            merged[flag] = val

    return build_pipeline_kwargs(merged)


def _cmd_single_step(step_num: int, args: argparse.Namespace) -> int:
    """Run exactly one stage. Always forces re-run (single-step is the
    explicit way to redo a stage; use `run` to leverage caching)."""
    audio = _resolve_audio(args)
    from .pipeline import run
    run(
        audio,
        model=getattr(args, "model", "medium") or "medium",
        language=getattr(args, "language", "zh") or "zh",
        compute_type=getattr(args, "compute_type", "int8") or "int8",
        library_dir=_path_or_default(getattr(args, "library", None), DEFAULT_LIBRARY),
        threshold=getattr(args, "threshold", 0.5),
        runs_dir=_path_or_default(getattr(args, "runs", None), DEFAULT_RUNS),
        output_dir=_path_or_default(getattr(args, "output", None), DEFAULT_OUTPUT),
        output_transcript_dir=_path_or_none(getattr(args, "output_transcript", None)),
        skip_identify=getattr(args, "skip_identify", False),
        from_step=step_num,
        to_step=step_num,
        force=True,
        recording_id=getattr(args, "recording_id", None),
        archive=False,  # archive is owned by the main flow, never by `si`
        **_build_asr_backend_kwargs(args),
    )
    return 0


def _cmd_viewer(args: argparse.Namespace) -> int:
    """Build an interactive word-aligned viewer for the given audio."""
    import subprocess
    from .viewer import build_viewer, default_viewers_root

    audio = Path(args.audio).expanduser().resolve()
    if not audio.exists():
        print(f"audio file not found: {audio}", file=sys.stderr)
        return 2

    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = default_viewers_root() / audio.stem

    try:
        index_html = build_viewer(
            audio,
            out_dir,
            model_name=args.model,
            language=args.language,
            force=args.force,
        )
    except Exception as err:
        print(f"viewer: {err}", file=sys.stderr)
        return 1

    print(f"viewer: {index_html}")
    if not args.no_open:
        # macOS `open` — harmless on other OS since we're Mac-only in practice.
        try:
            subprocess.run(["open", str(index_html)], check=False)
        except FileNotFoundError:
            pass
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    audio = _resolve_audio(args)
    from .pipeline import run
    from .._lock import single_instance_lock

    # `si run` is the diagnostic / re-run entry. End-to-end "first time"
    # processing of an audio file goes through the main flow
    # (`voicememowhisper <audio>`), which owns the state DB and the
    # ArchiveManager. Refuse to run on an audio that the main flow
    # hasn't seen yet — otherwise the two paths produce inconsistent
    # state (different archive naming, no state-DB record, etc.).
    if not getattr(args, "allow_unregistered", False):
        if not _audio_known_to_main_flow(audio, getattr(args, "recording_id", None)):
            print(
                f"si run: audio not registered with the main flow: {audio}",
                file=sys.stderr,
            )
            print(
                "  This entry is for diagnostic re-runs of already-processed audio.",
                file=sys.stderr,
            )
            print(
                "  To process a fresh file end-to-end, use:",
                file=sys.stderr,
            )
            print(
                f"      voicememo-whisper '{audio}'",
                file=sys.stderr,
            )
            print(
                "  To override and run anyway (advanced), pass --allow-unregistered.",
                file=sys.stderr,
            )
            return 2

    # Share the same single-instance lockfile as the main flow so a user
    # can't accidentally run `voicememowhisper` in one terminal and
    # `voicememowhisper si run ...` in another — both would race on the
    # same state DB / archive / outputs dirs.
    with single_instance_lock():
        run(
            audio,
            model=args.model or "medium",
            language=args.language or "zh",
            compute_type=args.compute_type or "int8",
            library_dir=_path_or_default(args.library, DEFAULT_LIBRARY),
            threshold=args.threshold,
            runs_dir=_path_or_default(args.runs, DEFAULT_RUNS),
            output_dir=_path_or_default(args.output, DEFAULT_OUTPUT),
            output_transcript_dir=_path_or_none(args.output_transcript),
            skip_identify=args.skip_identify,
            from_step=args.from_step,
            to_step=args.to_step,
            force=args.force,
            recording_id=args.recording_id,
            # Archive is owned by the main-flow ArchiveManager. `si run`
            # never touches the archive dir.
            archive=False,
            **_build_asr_backend_kwargs(args),
        )
    return 0


def _audio_known_to_main_flow(audio: Path, recording_id: str | None) -> bool:
    """Return True if the audio (or its corresponding recording_id cache)
    has been seen by the main flow — i.e. either:

    - the file lives under the configured archive dir (so it must have
      gone through the main flow's ArchiveManager), OR
    - there's a state-DB row whose archived_path or guid matches it, OR
    - a runs/ cache directory matching ``recording_id`` already exists
      (so it was previously processed under that id, regardless of where
      the source audio currently lives — covers re-runs against an
      archived/renamed file).
    """
    import sqlite3
    from ..config import Settings

    audio = audio.resolve()

    # 1. Already in the archive dir → trust the main flow put it there.
    try:
        settings = Settings()
        if settings.archive_dir:
            archive_dir = settings.archive_dir.resolve()
            if audio.is_relative_to(archive_dir):
                return True
            # 2. State DB has a row for this archived_path.
            state_db = settings.state_db
            if state_db and state_db.exists():
                conn = sqlite3.connect(str(state_db))
                try:
                    cur = conn.execute(
                        "SELECT 1 FROM processed WHERE archived_path = ? LIMIT 1",
                        (str(audio),),
                    )
                    if cur.fetchone():
                        return True
                    # 3. State DB has a row whose guid equals the audio stem
                    #    (Voice Memos source files use their guid as stem).
                    cur = conn.execute(
                        "SELECT 1 FROM processed WHERE guid = ? LIMIT 1",
                        (audio.stem,),
                    )
                    if cur.fetchone():
                        return True
                finally:
                    conn.close()
    except Exception:
        # Defensive: if we can't tell, fall through and let recording_id
        # check decide. Better to occasionally miss than to wrongly block.
        pass

    # 4. A cache dir already exists under the runs/ root for this id.
    rid = recording_id or audio.stem
    runs_dir = _path_or_default(None, DEFAULT_RUNS)
    if (runs_dir / rid).is_dir():
        return True

    return False


def _cmd_steps(_args: argparse.Namespace) -> int:
    print("Speaker-ID pipeline stages:\n")
    for num, name, desc in STEPS:
        print(f"  {num}. {name:<10s}  {desc}")
    print(
        "\nRun a single stage:   voicememo-whisper si <name> <audio>"
        "\nRun a range:          voicememo-whisper si run <audio> --from <name> --to <name>"
        "\nForce re-run:         add --force"
    )
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    audio = _resolve_audio(args)
    runs = _path_or_default(args.runs, DEFAULT_RUNS)
    output = _path_or_default(args.output, DEFAULT_OUTPUT)
    rid = getattr(args, "recording_id", None) or audio.stem
    run_dir = runs / rid
    out_dir = output / rid

    artifacts = [
        (1, "transcribe", run_dir / "transcript_faster_whisper.json"),
        (2, "diarize",    run_dir / "diarization_pyannote.json"),
        (2, "diarize",    run_dir / "diarization_pyannote.embeddings.npz"),
        (3, "identify",   run_dir / "identification.json"),
        (4, "merge",      run_dir / "merged.json"),
        (5, "render",     out_dir / "transcript.md"),
        (5, "render",     out_dir / "transcript.txt"),
    ]
    print(f"recording: {rid}")
    print(f"audio:     {audio}")
    print(f"runs dir:  {run_dir}")
    print(f"output:    {out_dir}")
    print()
    print(f"{'step':<6}{'stage':<12}{'state':<10}{'size':>10}  path")
    print("-" * 78)
    for num, name, p in artifacts:
        if p.exists():
            size = f"{p.stat().st_size / 1024:>9.1f}K"
            state = "cached"
        else:
            size = "       —"
            state = "MISSING"
        print(f"{num:<6}{name:<12}{state:<10}{size}  {p}")
    return 0


def _cmd_library(args: argparse.Namespace) -> int:
    library = _path_or_default(args.library, DEFAULT_LIBRARY)

    if args.lib_action == "list":
        if not library.exists():
            print(f"speaker library does not exist: {library}", file=sys.stderr)
            return 1
        # Inlined clip discovery (matches `library.list_clip_wavs` behaviour:
        # scan `<speaker>/clips/` with the same extensions). Kept stdlib-only
        # so `library list` works without the [speaker-id] extra — which is
        # useful for spotting an unenrolled speaker on a fresh checkout.
        print(f"library: {library}\n")
        rows = []
        clip_exts = {".wav", ".m4a", ".flac", ".mp3", ".ogg"}
        from .library import is_speaker_dir
        for d in sorted(p for p in library.iterdir() if is_speaker_dir(p)):
            clips_dir = d / "clips"
            clips = (
                sorted(p for p in clips_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in clip_exts)
                if clips_dir.exists() else []
            )
            embedding = d / "embedding.npy"
            rows.append((d.name, len(clips), "yes" if embedding.exists() else "NO"))
        if not rows:
            print("(empty)")
            return 0
        name_w = max(len(r[0]) for r in rows)
        print(f"{'speaker':<{name_w}}  {'clips':>5}  embedding")
        for name, n_clips, has_emb in rows:
            print(f"{name:<{name_w}}  {n_clips:>5}  {has_emb}")
        return 0

    if args.lib_action == "add":
        speaker = args.speaker
        clip = Path(args.clip).expanduser().resolve()
        if not clip.exists():
            print(f"clip not found: {clip}", file=sys.stderr)
            return 2
        # Drop clips under <speaker>/clips/ so `library.list_clip_wavs` and
        # the embedding rebuild find them. Writing to the speaker dir root
        # (the previous behaviour) silently excluded the clip from the next
        # centroid compute.
        clips_dir = library / speaker / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(clips_dir.glob("[0-9][0-9][0-9].*"))
        next_num = (int(existing[-1].stem) + 1) if existing else 1
        target = clips_dir / f"{next_num:03d}{clip.suffix.lower()}"
        shutil.copy2(clip, target)
        print(f"copied {clip} → {target}")

        # Populate profile.json with the human-readable display_name up front
        # when the speaker dir is new, so the user doesn't have to hand-edit the
        # JSON every time. Skipped if profile.json already exists (preserves
        # prior manual edits on re-enrollment of an existing speaker).
        display_name = getattr(args, "display_name", None)
        aliases = getattr(args, "alias", None) or []
        notes = getattr(args, "notes", None)
        profile_path = library / speaker / "profile.json"
        if not profile_path.exists() and (display_name or aliases or notes):
            import json as _json
            profile_path.write_text(
                _json.dumps(
                    {
                        "speaker_id": speaker,
                        "display_name": display_name or speaker,
                        "aliases": aliases,
                        "notes": notes or "",
                    },
                    ensure_ascii=False, indent=2,
                ) + "\n",
                encoding="utf-8",
            )
            print(f"wrote {profile_path}")

        if not args.no_rebuild:
            return _rebuild_speaker(library, speaker)
        print("(skipping embedding rebuild; rerun with `library rebuild` to update)")
        return 0

    if args.lib_action == "rebuild":
        return _rebuild_speaker(library, args.speaker)

    if args.lib_action == "score":
        return _cmd_library_score(args, library)

    if args.lib_action == "badcase-add":
        return _cmd_library_badcase_add(args, library)

    if args.lib_action == "find-candidates":
        return _cmd_library_find_candidates(args, library)

    return 0


def _cmd_library_find_candidates(args: argparse.Namespace, library: Path) -> int:
    """List longest independent blocks per speaker in a cached recording."""
    import json as _json
    import subprocess as _sp

    runs_dir = _path_or_default(getattr(args, "runs", None), DEFAULT_RUNS)
    rid = getattr(args, "recording_id", None)
    if not rid and getattr(args, "audio", None):
        rid = Path(args.audio).expanduser().stem
    if not rid:
        candidates = sorted(
            (d for d in runs_dir.iterdir() if d.is_dir()),
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            print(f"no cached runs found under {runs_dir}", file=sys.stderr)
            return 2
        rid = candidates[0].name
        print(f"# using most recently-modified run: {rid}", file=sys.stderr)

    merged_path = runs_dir / rid / "merged.json"
    if not merged_path.exists():
        print(f"merged.json not found: {merged_path}", file=sys.stderr)
        print("(run `si run` through stage 4 first)", file=sys.stderr)
        return 2

    data = _json.loads(merged_path.read_text(encoding="utf-8"))
    segs = data.get("segments", [])

    # Group consecutive same-speaker segments into blocks
    groups: list[dict] = []
    cur: dict | None = None
    for s in segs:
        spk = s.get("speaker_label") or s.get("speaker")
        name = s.get("speaker_name") or ""
        if cur and cur["spk"] == spk:
            cur["end"] = s["end"]
            cur["text"] += s.get("text", "")
        else:
            if cur:
                groups.append(cur)
            cur = {
                "spk": spk,
                "name": name,
                "start": s["start"],
                "end": s["end"],
                "text": s.get("text", ""),
            }
    if cur:
        groups.append(cur)

    # Index by speaker
    by_spk: dict[str, list[dict]] = {}
    for g in groups:
        by_spk.setdefault(g["spk"], []).append(g)

    # Filter to one speaker if requested (match by label OR display name)
    target_filter = getattr(args, "speaker", None)
    if target_filter:
        match_spks = [
            spk
            for spk, blocks in by_spk.items()
            if spk == target_filter
            or (blocks and blocks[0]["name"] == target_filter)
        ]
        if not match_spks:
            available = sorted(
                f"{spk} ({blocks[0]['name'] or '?'})"
                for spk, blocks in by_spk.items()
            )
            print(f"no speaker matches '{target_filter}'", file=sys.stderr)
            print(f"available: {', '.join(available)}", file=sys.stderr)
            return 2
        by_spk = {spk: by_spk[spk] for spk in match_spks}

    audio_path = (
        Path(args.audio).expanduser().resolve()
        if getattr(args, "audio", None)
        else None
    )
    cut_middle = getattr(args, "cut_middle_sec", None)
    out_dir = Path(getattr(args, "out_dir", "/tmp")).expanduser()
    if cut_middle and not audio_path:
        print("--cut-middle-sec requires --audio", file=sys.stderr)
        return 2
    if cut_middle:
        out_dir.mkdir(parents=True, exist_ok=True)

    top_n = max(1, int(getattr(args, "top", 5)))
    min_dur = float(getattr(args, "min_duration", 10.0))

    for spk in sorted(by_spk):
        blocks = by_spk[spk]
        total_dur = sum(g["end"] - g["start"] for g in blocks)
        long_blocks = sorted(
            (g for g in blocks if (g["end"] - g["start"]) >= min_dur),
            key=lambda g: -(g["end"] - g["start"]),
        )[:top_n]
        name = blocks[0]["name"] or "(unknown)"
        print(f"=== {spk} → {name}  total {total_dur:.1f}s, "
              f"{len(blocks)} blocks, top {len(long_blocks)} ≥ {min_dur:.0f}s ===")
        for i, g in enumerate(long_blocks, 1):
            dur = g["end"] - g["start"]
            preview = g["text"][:60].replace("\n", " ")
            print(f"  [{i}] {g['start']:7.1f}-{g['end']:7.1f}  {dur:5.1f}s  {preview}")
            if cut_middle and audio_path:
                cut_dur = min(cut_middle, dur)
                ss = g["start"] + max(0.0, (dur - cut_dur) / 2)
                clip_name = f"{spk}-{i:02d}-{int(ss)}-{int(ss + cut_dur)}.m4a"
                clip_path = out_dir / clip_name
                cmd = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-ss", f"{ss:.2f}", "-i", str(audio_path),
                    "-t", f"{cut_dur:.2f}", "-c", "copy", str(clip_path),
                ]
                try:
                    _sp.run(cmd, check=True)
                    print(f"      cut → {clip_path}")
                except (FileNotFoundError, _sp.CalledProcessError) as e:
                    print(f"      ffmpeg failed: {e}", file=sys.stderr)
        print()
    return 0


def _cmd_library_badcase_add(args: argparse.Namespace, library: Path) -> int:
    """Save a non-speaker clip + embedding + baseline cos snapshot under the
    speaker directory it was misattributed to."""
    import json as _json
    import shutil as _shutil
    import numpy as _np

    from .library import is_speaker_dir
    from .speaker_embed import CommunityOneEmbedder

    clip = Path(args.clip).expanduser().resolve()
    if not clip.exists():
        print(f"clip not found: {clip}", file=sys.stderr)
        return 2

    speaker_dir = library / args.speaker
    if not speaker_dir.is_dir():
        print(f"speaker directory does not exist: {speaker_dir}", file=sys.stderr)
        print("(badcases attach to an existing speaker — enroll the speaker first, "
              "then add the badcase)", file=sys.stderr)
        return 2

    bc_dir = speaker_dir / "badcases" / args.badcase_id
    (bc_dir / "clips").mkdir(parents=True, exist_ok=True)
    target = bc_dir / "clips" / f"clip{clip.suffix.lower()}"
    _shutil.copy2(clip, target)
    print(f"copied {clip} → {target}")

    print("loading embedder ...")
    emb = CommunityOneEmbedder()
    v = emb.embed_audio(target)
    norm = _np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    _np.save(str(bc_dir / "embedding.npy"), v)

    baselines: dict[str, float] = {}
    for d in sorted(p for p in library.iterdir() if is_speaker_dir(p)):
        e_path = d / "embedding.npy"
        if not e_path.exists():
            continue
        e = _np.load(str(e_path))
        e_norm = _np.linalg.norm(e)
        if e_norm > 0:
            e = e / e_norm
        baselines[d.name] = round(float(_np.dot(v, e)), 4)

    max_cos = max(baselines.values()) if baselines else 0.0
    max_speaker = max(baselines, key=baselines.get) if baselines else None
    from datetime import date as _date
    today = _date.today().isoformat()

    meta = {
        "badcase_id": args.badcase_id,
        "kind": args.kind,
        "false_match_speaker": args.speaker,
        "source_recording": args.source_recording,
        "source_time_range": args.source_time_range,
        "description": args.description or "",
        "max_cos_allowed": args.max_cos_allowed,
        f"baseline_cos_against_speakers_{today}": baselines,
        "max_cos": max_cos,
        "max_cos_speaker": max_speaker,
    }
    (bc_dir / "meta.json").write_text(
        _json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {bc_dir / 'meta.json'}")

    # Append a summary entry to the speaker's profile.json so the existence of
    # this badcase is visible without opening the meta.json. Keeps profile.json
    # as the single-stop overview of a speaker (clips + notes + badcases).
    profile_path = speaker_dir / "profile.json"
    profile = (
        _json.loads(profile_path.read_text(encoding="utf-8"))
        if profile_path.exists()
        else {"speaker_id": args.speaker, "display_name": args.speaker, "aliases": [], "notes": ""}
    )
    badcase_summary = {
        "badcase_id": args.badcase_id,
        "kind": args.kind,
        "max_cos_at_enrollment": baselines.get(args.speaker, 0.0),
        "source_recording": args.source_recording,
        "source_time_range": args.source_time_range,
    }
    if args.description:
        badcase_summary["note"] = args.description
    existing_badcases = profile.get("badcases", [])
    # Replace by id if already present (idempotent re-add).
    existing_badcases = [b for b in existing_badcases if b.get("badcase_id") != args.badcase_id]
    existing_badcases.append(badcase_summary)
    profile["badcases"] = existing_badcases
    profile_path.write_text(
        _json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"updated {profile_path} (badcases entry)")

    print(f"max cos against any real speaker: {max_cos:.4f} → {max_speaker} "
          f"(allowed ≤ {args.max_cos_allowed})")
    if max_cos > args.max_cos_allowed:
        print("WARNING: baseline already exceeds max_cos_allowed", file=sys.stderr)
        return 1
    return 0


def _cmd_library_score(args: argparse.Namespace, library: Path) -> int:
    """Cosine cross-match: every SPEAKER centroid (from a cached diarize
    run) vs every library centroid. Pure numpy — no ML deps reload, so
    it's safe to run while another pyannote pipeline is going."""
    import json as _json
    import numpy as np  # part of [speaker-id] extra; required for this cmd

    # Resolve recording_id: explicit → audio stem → most recently-modified
    # runs subdir. Lets the caller run `si library score --recording-id X` after
    # archiving an audio without needing the original path back.
    runs_dir = _path_or_default(getattr(args, "runs", None), DEFAULT_RUNS)
    rid = getattr(args, "recording_id", None)
    if not rid and getattr(args, "audio", None):
        rid = Path(args.audio).expanduser().stem
    if not rid:
        # Fall back to most-recently-used cache dir.
        candidates = sorted(
            (d for d in runs_dir.iterdir() if d.is_dir()),
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        ) if runs_dir.exists() else []
        if not candidates:
            print(f"no cached runs under {runs_dir}", file=sys.stderr)
            return 2
        rid = candidates[0].name
        print(f"(no --recording-id / --audio given; using most recent cache: {rid})")

    emb_path = runs_dir / rid / "diarization_pyannote.embeddings.npz"
    if not emb_path.exists():
        print(f"embeddings cache missing: {emb_path}", file=sys.stderr)
        print(f"run stage 2 (diarize) on this recording first.", file=sys.stderr)
        return 2

    if not library.exists():
        print(f"library does not exist: {library}", file=sys.stderr)
        return 2

    npz = np.load(emb_path)
    speakers = sorted(npz.files)

    # Gather library centroids + display names.
    lib = {}
    for d in sorted(library.iterdir()):
        if not d.is_dir():
            continue
        emb = d / "embedding.npy"
        if not emb.exists():
            continue
        prof = d / "profile.json"
        name = d.name
        if prof.exists():
            try:
                name = _json.loads(prof.read_text(encoding="utf-8")).get("display_name", d.name)
            except Exception:
                pass
        lib[d.name] = (name, np.load(emb))

    if not lib:
        print(f"library has no enrolled embeddings: {library}", file=sys.stderr)
        return 2

    def _cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    threshold = float(getattr(args, "threshold", 0.5))
    print(f"recording: {rid}")
    print(f"library:   {library}  ({len(lib)} speakers)")
    print(f"threshold: {threshold}")
    print()
    print(f"{'SPEAKER':<11} {'best':<18} {'cos':>7}  {'2nd':<18} {'cos':>7}  {'margin':>7}  flag")
    print("-" * 86)
    for s in speakers:
        vec = npz[s]
        scores = sorted(
            [(_cos(vec, v), sid, name) for sid, (name, v) in lib.items()],
            reverse=True,
        )
        best = scores[0]
        snd = scores[1] if len(scores) > 1 else (0.0, "", "—")
        margin = best[0] - snd[0]
        flags = []
        if best[0] < threshold:
            flags.append("UNMATCHED")
        elif margin < 0.1:
            flags.append("AMBIGUOUS")
        print(
            f"{s:<11} {best[2]:<18} {best[0]:>7.4f}  "
            f"{snd[2]:<18} {snd[0]:>7.4f}  {margin:>7.4f}  "
            f"{' '.join(flags)}"
        )
    return 0


def _rebuild_speaker(library: Path, speaker: str | None) -> int:
    """Recompute embedding.npy for one speaker (or all if speaker is None)."""
    from . import library as mod_library  # lazy: needs numpy + pyannote
    code = mod_library.run_build(library_dir=library, speaker_filter=speaker)
    return code


# ───────── Helpers ────────────────────────────────────────────────────


def _resolve_audio(args: argparse.Namespace) -> Path:
    audio = Path(args.audio).expanduser().resolve()
    if not audio.exists():
        raise SystemExit(f"error: audio file not found: {audio}")
    return audio


def _path_or_default(value, default: Path) -> Path:
    if value is None:
        default.mkdir(parents=True, exist_ok=True)
        return default
    return Path(value).expanduser().resolve()


def _path_or_none(value):
    return Path(value).expanduser().resolve() if value else None


# ───────── Argparse plumbing ──────────────────────────────────────────


def _add_common_audio_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("audio", help="Path to audio file (.m4a, .wav, .mp3)")
    p.add_argument("--runs", help=f"Runs directory for intermediates (default: {DEFAULT_RUNS})")
    p.add_argument("--output", help=f"Final outputs directory (default: {DEFAULT_OUTPUT})")
    p.add_argument("--library", help=f"Speaker library directory (default: {DEFAULT_LIBRARY})")
    p.add_argument("--output-transcript",
                   help="Copy final transcript.md/.txt to this directory (e.g. ~/Documents/VoiceMemoWhisper/Transcripts)")
    p.add_argument("--recording-id", dest="recording_id", default=None,
                   help="Override cache-dir key (default: audio.stem). Use when the "
                        "audio has been archived/renamed but the original runs/<id>/ "
                        "cache still exists.")


def _add_asr_backend_args(p: argparse.ArgumentParser) -> None:
    """Flags that select + configure the stage-1 ASR backend.

    Added to both `run` and the `transcribe` single-step subcommand so
    either entry point can switch backends. Defaults preserve the
    existing behaviour (``faster_whisper``) — passing nothing changes
    nothing.
    """
    p.add_argument(
        "--asr-backend", dest="asr_backend", default=None,
        choices=["faster_whisper", "openai-audio", "ws-funasr", "ws_funasr"],
        help="Stage-1 transcription backend. 'openai-audio' routes to "
             "any OpenAI Audio API compatible server (OpenAI Whisper, "
             "a self-hosted FunASR slim, groq, ...) — see --asr-url / "
             "--asr-model. 'ws-funasr' streams PCM to a funasr-aipod "
             "WebSocket endpoint (aliyun SpeechTranscriber protocol) "
             "with idle-timer monitoring, no hard wall-clock timeout — "
             "see --asr-ws-url. Reads from "
             "~/.config/voicememowhisper/config.toml if present; falls "
             "back to 'faster_whisper' if neither flag nor config sets it.",
    )
    p.add_argument("--asr-config", dest="asr_config", default=None,
                   help="Path to a TOML config file with ASR backend settings. "
                        "Default search: $VMW_CONFIG, "
                        "~/.config/voicememowhisper/config.toml, "
                        "~/.voicememowhisper.toml. See docs/asr-backends.md.")
    p.add_argument("--asr-url", dest="asr_url", default=None,
                   help="openai-audio endpoint URL "
                        "(e.g. http://asr.internal:8000/v1/audio/transcriptions)")
    p.add_argument("--asr-model", dest="asr_model", default=None,
                   help="Model name the openai-audio server accepts "
                        "(e.g. paraformer-large, whisper-1, sensevoice-small)")
    p.add_argument("--asr-api-key", dest="asr_api_key", default=None,
                   help="Bearer token for openai-audio (omit for no-auth self-hosted)")
    p.add_argument("--asr-host-header", dest="asr_host_header", default=None,
                   help="Host header override (e.g. 'asr.internal' when routing via a reverse proxy)")
    p.add_argument("--asr-response-format", dest="asr_response_format", default=None,
                   choices=["json", "verbose_json", "text", "srt", "vtt"],
                   help="openai-audio response format (default: verbose_json)")
    p.add_argument("--asr-timeout-sec", dest="asr_timeout_sec", type=float, default=None,
                   help="openai-audio per-request timeout in seconds (default: 600)")
    # ws-funasr knobs. Kept under their own --asr-ws-* namespace so they
    # don't collide with the HTTP-backend flags above — a user with both
    # [asr.http] and [asr.ws] in config can switch backends without the
    # flag set conflicting.
    p.add_argument("--asr-ws-url", dest="asr_ws_url", default=None,
                   help="ws-funasr endpoint URL (e.g. wss://your-host/ws/v1/asr)")
    p.add_argument("--asr-ws-idle-timeout-sec", dest="asr_ws_idle_timeout_sec",
                   type=float, default=None,
                   help="ws-funasr idle timeout in seconds. There is NO "
                        "wall-clock cap — only this idle timer. Server "
                        "sends no event for this long → assume dead. "
                        "(default: 60)")
    p.add_argument("--asr-ws-connect-timeout-sec", dest="asr_ws_connect_timeout_sec",
                   type=float, default=None,
                   help="ws-funasr WebSocket connect timeout in seconds (default: 15)")


def _add_diarize_backend_args(p: argparse.ArgumentParser) -> None:
    """Flags that select + configure the stage-2 diarization backend.

    Defaults preserve existing behaviour (``local_pyannote`` on CPU/MPS);
    switching to ``http`` routes to the self-hosted pyannote GPU service
    and cuts wall-clock from ~20 min to ~90 sec on a 23-min recording.
    """
    p.add_argument(
        "--diarize-backend", dest="diarize_backend", default=None,
        choices=["local_pyannote", "http"],
        help="Stage-2 diarization backend. 'http' calls the self-hosted "
             "pyannote service (~12× faster on GPU). Reads from config "
             "file if present; falls back to 'local_pyannote' otherwise.",
    )
    p.add_argument("--diarize-url", dest="diarize_url", default=None,
                   help="diarize service endpoint URL "
                        "(e.g. http://diarize.internal:8000/diarize)")
    p.add_argument("--diarize-host-header", dest="diarize_host_header", default=None,
                   help="Host header override for reverse-proxy routing")
    p.add_argument("--diarize-api-key", dest="diarize_api_key", default=None,
                   help="Bearer token (omit for no-auth self-hosted)")
    p.add_argument("--diarize-timeout-sec", dest="diarize_timeout_sec", type=float,
                   default=None,
                   help="per-request timeout in seconds (default: 900)")
    p.add_argument("--diarize-num-speakers", dest="diarize_num_speakers", type=int,
                   default=None, help="hint: exact number of speakers")
    p.add_argument("--diarize-min-speakers", dest="diarize_min_speakers", type=int,
                   default=None, help="hint: minimum number of speakers")
    p.add_argument("--diarize-max-speakers", dest="diarize_max_speakers", type=int,
                   default=None, help="hint: maximum number of speakers")


def _add_stage_specific_args(name: str, p: argparse.ArgumentParser) -> None:
    if name == "transcribe":
        p.add_argument("--model", default="medium",
                       help="Whisper model: tiny / base / small / medium / large-v3 (default: medium)")
        p.add_argument("--language", default="zh",
                       help="Language hint, or 'auto' (default: zh)")
        p.add_argument("--compute-type", default="int8",
                       help="int8 / int8_float16 / float16 / float32 (default: int8)")
        _add_asr_backend_args(p)
    if name == "diarize":
        _add_diarize_backend_args(p)
    if name == "identify":
        p.add_argument("--threshold", type=float, default=0.5,
                       help="Cosine match threshold (default: 0.5)")
    if name == "merge":
        # merge needs identify-related model name only for the pipeline tag string
        p.add_argument("--model", default="medium", help="(only used in pipeline label)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="voicememo-whisper si",
        description=(
            "Speaker-ID pipeline. Runs in 5 ordered stages "
            "(transcribe → diarize → identify → merge → render). "
            "Each stage caches its output; subcommands let you replay one stage, "
            "or `run` a contiguous range. Use `si steps` for the full ordered list."
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    # Per-stage subcommands.
    for num, name, _desc in STEPS:
        sp = sub.add_parser(name, help=_step_help(num),
                            description=_step_help(num))
        _add_common_audio_args(sp)
        _add_stage_specific_args(name, sp)
        if name == "identify":
            sp.add_argument("--skip-identify", action="store_true",
                            help="Run as no-op (skip identification)")
        sp.set_defaults(_handler=lambda a, n=num: _cmd_single_step(n, a))

    # `run` — contiguous range.
    sr = sub.add_parser("run", help="Run a range of steps (with stage caching)",
                        description="Run a contiguous range of steps. Cached intermediates "
                                    "are reused unless --force is given.")
    _add_common_audio_args(sr)
    sr.add_argument("--model", default="medium",
                    help="Whisper model (default: medium)")
    sr.add_argument("--language", default="zh", help="Language hint (default: zh)")
    sr.add_argument("--compute-type", default="int8",
                    help="Whisper compute type (default: int8)")
    sr.add_argument("--threshold", type=float, default=0.5,
                    help="Identify cosine threshold (default: 0.5)")
    sr.add_argument("--from", dest="from_step", default="1",
                    help=f"Start step. Name (one of {', '.join(STEP_NAMES)}) or number 1-{len(STEPS)}. Default: 1.")
    sr.add_argument("--to", dest="to_step", default=str(len(STEPS)),
                    help=f"End step (inclusive). Same format as --from. Default: {len(STEPS)}.")
    sr.add_argument("--force", action="store_true",
                    help="Re-run in-window steps even if cached output exists.")
    sr.add_argument("--skip-identify", action="store_true",
                    help="Skip the identify stage (treats it as no-op)")
    sr.add_argument("--allow-unregistered", action="store_true",
                    help="Run on an audio file that the main flow hasn't "
                         "seen yet. Default behaviour is to refuse and point "
                         "the user at `voicememo-whisper <audio>`. Use this "
                         "flag only for ad-hoc/development experiments where "
                         "you don't want a state DB row or archive copy.")
    _add_asr_backend_args(sr)
    _add_diarize_backend_args(sr)
    sr.set_defaults(_handler=_cmd_run)

    # `library`.
    sl = sub.add_parser("library", help="Manage the speaker-library",
                        description="List, add to, or rebuild the speaker-library used for identification.")
    sl.add_argument("--library", help=f"Speaker library directory (default: {DEFAULT_LIBRARY})")
    lsub = sl.add_subparsers(dest="lib_action", required=True, metavar="ACTION")
    lsub.add_parser("list", help="List speakers and their clip / embedding state")
    sla = lsub.add_parser("add", help="Add a clip to a speaker, then rebuild that speaker's embedding")
    sla.add_argument("speaker", help="Speaker id (directory name under the library)")
    sla.add_argument("clip", help="Path to a .wav (or .m4a) clip to enroll")
    sla.add_argument("--display-name", dest="display_name",
                     help="Human-readable name written into profile.json (e.g. '徐子悠'). "
                          "Only applied on first enrollment — existing profile.json is preserved.")
    sla.add_argument("--alias", action="append", default=None,
                     help="Alias for the speaker; repeat to add multiple (e.g. --alias 'Ziyou Xu' --alias ziyou).")
    sla.add_argument("--notes", default=None,
                     help="Notes written into profile.json on first enrollment "
                          "(evidence chain, clip sources, etc).")
    sla.add_argument("--no-rebuild", action="store_true",
                     help="Don't rebuild the embedding after copying")
    slr = lsub.add_parser("rebuild", help="Recompute centroid embeddings")
    slr.add_argument("--speaker", help="Restrict to a single speaker id (default: all speakers)")
    # `badcase-add` — store a known-bad clip (env noise / wrong-attribution / overlap)
    # under <library>/<false-match-speaker>/badcases/<id>/ together with its embedding
    # and a baseline cos snapshot against every current real speaker. Owning the
    # badcase by the speaker it was misattributed to keeps the directory layout flat
    # at the top (only real speakers) and makes the false-match relationship explicit.
    slba = lsub.add_parser(
        "badcase-add",
        help="Add a non-speaker clip (noise / overlap / mis-diarized) as a regression badcase",
        description="Saves the clip + its embedding + cos-vs-every-real-speaker baseline "
                    "under <library>/<speaker>/badcases/<id>/. <speaker> is the speaker the "
                    "clip was MISATTRIBUTED to during identify (the false match), not the "
                    "speaker the clip actually contains.",
    )
    slba.add_argument("speaker",
                      help="Speaker id this badcase was falsely attributed to during identify "
                           "(directory under the library)")
    slba.add_argument("badcase_id",
                      help="Folder name under <speaker>/badcases/ "
                           "(suggested format: <date>-<source-cluster>-<kind>-<duration>)")
    slba.add_argument("clip", help="Path to clip file")
    slba.add_argument("--kind", default="env_noise",
                      help="Tag describing the badcase type (env_noise, overlap, cross_talk, ...). "
                           "Default: env_noise")
    slba.add_argument("--source-recording", default=None,
                      help="Source recording filename")
    slba.add_argument("--source-time-range", default=None,
                      help="Time range in source (e.g. '632.1-641.6 (9.5s)')")
    slba.add_argument("--description", default=None,
                      help="Free-form notes about why this is a badcase")
    slba.add_argument("--max-cos-allowed", type=float, default=0.5,
                      help="Acceptable upper bound for cos vs any real speaker (default: 0.5).")
    # `find-candidates` — for a given cached recording, list each speaker's longest
    # independent monologue blocks. Used during library cleanup: pick a clean clip
    # to enroll an under-matched speaker. Reads merged.json (no ML reload).
    slfc = lsub.add_parser(
        "find-candidates",
        help="List longest independent blocks per speaker in a cached recording (for picking enrollment clips)",
        description="For each speaker in a cached recording's merged.json, group "
                    "consecutive same-speaker segments into blocks and print the "
                    "longest N blocks. Use to pick a candidate enrollment clip without "
                    "manually parsing JSON. Optionally cuts clips with ffmpeg.",
    )
    slfc.add_argument("--audio",
                      help="Audio path — uses its stem as recording_id by default.")
    slfc.add_argument("--recording-id", dest="recording_id",
                      help="Recording id (runs/<id>/ subdir name). If neither --audio "
                           "nor --recording-id is given, picks the most recently-updated "
                           "runs/ subdir.")
    slfc.add_argument("--runs", help=f"Runs directory (default: {DEFAULT_RUNS})")
    slfc.add_argument("--speaker", default=None,
                      help="Filter to one speaker (cluster label like 'SPEAKER_02' or "
                           "display name from the speaker library). Default: list every speaker.")
    slfc.add_argument("--top", type=int, default=5,
                      help="Top N longest blocks per speaker (default: 5).")
    slfc.add_argument("--min-duration", type=float, default=10.0,
                      help="Skip blocks shorter than this many seconds (default: 10).")
    slfc.add_argument("--cut-middle-sec", type=float, default=None,
                      help="If set, also cut the middle N seconds of each top block as "
                           "a candidate clip (requires --audio). Files written to /tmp/ "
                           "by default, override with --out-dir.")
    slfc.add_argument("--out-dir", default="/tmp",
                      help="Where to write cut clips (default: /tmp).")
    # `score` — cross-match a cached recording's speakers against the library.
    # Pure numpy (reads cached .npz), so safe to run in parallel with a live
    # pyannote pipeline and fast enough for interactive iteration.
    sls = lsub.add_parser(
        "score",
        help="Show cosine match table for a cached recording's SPEAKERs against the library",
        description="Reads the cached diarize-stage embeddings and prints a per-speaker "
                    "best/2nd/margin cosine table against the current library. Flags rows "
                    "below threshold as UNMATCHED and rows with margin<0.1 as AMBIGUOUS.",
    )
    sls.add_argument("--audio",
                     help="Audio path — uses its stem as recording_id by default.")
    sls.add_argument("--recording-id", dest="recording_id",
                     help="Recording id (runs/<id>/ subdir name). If neither --audio "
                          "nor --recording-id is given, picks the most recently-updated "
                          "runs/ subdir.")
    sls.add_argument("--runs", help=f"Runs directory (default: {DEFAULT_RUNS})")
    sls.add_argument("--threshold", type=float, default=0.5,
                     help="Cosine threshold for UNMATCHED flag (default: 0.5).")
    sl.set_defaults(_handler=_cmd_library)

    # `viewer`.
    sv = sub.add_parser(
        "viewer",
        help="Generate an interactive word-aligned HTML viewer for an audio clip",
        description="Runs faster-whisper with word-level timestamps on the "
                    "given audio, renders a waveform + spectrogram PNG with "
                    "per-word color bands, and emits an HTML page with "
                    "click-to-seek character labels and a 60fps playback "
                    "cursor. Built for auditioning speaker-library candidate "
                    "clips and sanity-checking word-level alignment.",
    )
    sv.add_argument("audio", help="Path to audio file")
    sv.add_argument("--model", default="small",
                    help="faster-whisper model size "
                         "(tiny/base/small/medium/large-v3). Default: small.")
    sv.add_argument("--language", default="zh",
                    help="ASR language hint. Default: zh.")
    sv.add_argument("--out-dir", default=None,
                    help="Where to write the viewer dir. Default: "
                         "~/.local/share/voicememowhisper/viewers/<audio_stem>/")
    sv.add_argument("--force", action="store_true",
                    help="Re-run transcription even if a cached "
                         "transcript.json exists for this audio.")
    sv.add_argument("--no-open", action="store_true",
                    help="Don't auto-open the generated HTML in a browser.")
    sv.set_defaults(_handler=_cmd_viewer)

    # `inspect`.
    si = sub.add_parser("inspect",
                        help="Show cached intermediate state for a recording",
                        description="List which pipeline artifacts exist for a given audio file.")
    _add_common_audio_args(si)
    si.set_defaults(_handler=_cmd_inspect)

    # `steps`.
    sst = sub.add_parser("steps", help="Print the ordered list of pipeline steps",
                         description="Print the ordered list of pipeline steps and exit.")
    sst.set_defaults(_handler=_cmd_steps)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args._handler(args)
    except KeyboardInterrupt:
        print("\nsi: interrupted", file=sys.stderr)
        return 130
    except SystemExit:
        raise
    except Exception as e:
        print(f"\nsi: error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
