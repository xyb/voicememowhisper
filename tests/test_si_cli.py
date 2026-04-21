"""Smoke tests for the `voicememo-whisper si` CLI.

Intentionally stdlib-only: these guard against import-level regressions
that would break the CLI when the `[speaker-id]` extra is NOT installed.
All ML deps (numpy / torch / pyannote) must stay lazy so `si --help`,
`si steps`, `si inspect`, and `si library list` keep working on a fresh
checkout without the heavy stack.
"""

from __future__ import annotations

import io
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from voicememowhisper.si import cli as si_cli
from voicememowhisper.si.pipeline import STEPS, STEP_NAMES, resolve_step


# ---------- step resolution ----------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        (1, 1), ("1", 1), ("transcribe", 1),
        (5, 5), ("5", 5), ("render", 5),
        ("DIARIZE", 2),  # case-insensitive
    ],
)
def test_resolve_step_accepts_name_and_number(raw, expected):
    assert resolve_step(raw) == expected


def test_resolve_step_rejects_unknown():
    with pytest.raises(ValueError):
        resolve_step("bogus")
    with pytest.raises(ValueError):
        resolve_step(99)


def test_steps_constant_matches_names():
    assert [n for _, n, _ in STEPS] == STEP_NAMES
    assert [num for num, _, _ in STEPS] == list(range(1, len(STEPS) + 1))


# ---------- parser & --help surface the pipeline order --------------------


def test_parser_lists_all_stages():
    parser = si_cli.build_parser()
    help_text = parser.format_help()
    for name in STEP_NAMES:
        assert name in help_text, f"{name} missing from `si --help`"
    # Run/library/inspect/steps are also discoverable.
    for cmd in ("run", "library", "inspect", "steps"):
        assert cmd in help_text


def test_steps_subcommand_prints_ordered_list():
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = si_cli.main(["steps"])
    assert rc == 0
    out = buf.getvalue()
    for num, name, _ in STEPS:
        assert f"{num}. {name}" in out


def test_each_stage_subcommand_help_shows_step_number():
    parser = si_cli.build_parser()
    for num, name, _desc in STEPS:
        # argparse reaches subparsers via choices dict on _SubParsersAction.
        subparsers_action = next(
            a for a in parser._actions
            if isinstance(a, type(parser._subparsers._group_actions[0]))
            and getattr(a, "choices", None) and name in a.choices
        )
        sub = subparsers_action.choices[name]
        help_text = sub.format_help()
        assert f"Step {num}/{len(STEPS)}" in help_text, (
            f"Step number not surfaced in `si {name} --help`"
        )


# ---------- import-level guards for missing extras ------------------------


def test_si_module_init_does_not_import_ml_deps():
    """`import voicememowhisper.si` must not pull numpy/torch/pyannote.

    We check by looking at sys.modules after a fresh sub-process import.
    This is the invariant that keeps `si --help` working on machines
    without the [speaker-id] extra.
    """
    code = (
        "import sys, voicememowhisper.si  # noqa\n"
        "heavy = [m for m in ('numpy','torch','pyannote','faster_whisper')\n"
        "         if m in sys.modules]\n"
        "print('HEAVY:' + ','.join(heavy))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, check=True,
    )
    assert "HEAVY:" in proc.stdout
    leaked = proc.stdout.strip().split("HEAVY:", 1)[1]
    assert leaked == "", f"heavy modules leaked into `import si`: {leaked!r}"


def test_si_cli_help_does_not_import_ml_deps():
    """`voicememo-whisper si --help` must exit 0 without ML deps."""
    code = (
        "import sys\n"
        "from voicememowhisper.si import cli\n"
        "try:\n"
        "    cli.main(['--help'])\n"
        "except SystemExit as e:\n"
        "    print('EXIT:', e.code)\n"
        "heavy = [m for m in ('numpy','torch','pyannote','faster_whisper')\n"
        "         if m in sys.modules]\n"
        "print('HEAVY:' + ','.join(heavy))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, check=True,
    )
    assert "EXIT: 0" in proc.stdout
    leaked = proc.stdout.strip().split("HEAVY:", 1)[1]
    assert leaked == "", f"heavy modules leaked on --help: {leaked!r}"


# ---------- library list tolerates empty / missing structure --------------


def test_library_list_reports_zero_clips_without_clips_dir(tmp_path, capsys):
    lib = tmp_path / "library"
    (lib / "alice").mkdir(parents=True)
    (lib / "bob" / "clips").mkdir(parents=True)  # empty clips dir
    (lib / "alice" / "embedding.npy").write_bytes(b"\x00")  # stand-in

    rc = si_cli.main(["library", "--library", str(lib), "list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "alice" in out and "bob" in out
    # Both show 0 clips (no actual audio files); only alice has embedding.
    for line in out.splitlines():
        if line.startswith("alice"):
            assert "0" in line and "yes" in line
        if line.startswith("bob"):
            assert "0" in line and "NO" in line


def test_library_add_writes_under_clips_subdir(tmp_path, capsys):
    lib = tmp_path / "library"
    clip = tmp_path / "sample.wav"
    clip.write_bytes(b"fake-wav")

    # Skip embedding rebuild (that would need ML deps + real audio).
    rc = si_cli.main([
        "library", "--library", str(lib),
        "add", "charlie", str(clip),
        "--no-rebuild",
    ])
    assert rc == 0
    dest = lib / "charlie" / "clips" / "001.wav"
    assert dest.exists(), f"clip should land under clips/: tree={list(lib.rglob('*'))}"
    assert dest.read_bytes() == b"fake-wav"


# ---------- library add: profile.json auto-population ---------------------


def test_library_add_writes_profile_on_first_enroll(tmp_path):
    """--display-name / --alias / --notes populate profile.json on first add
    so xyb doesn't have to hand-edit JSON every time."""
    import json
    lib = tmp_path / "library"
    clip = tmp_path / "sample.wav"
    clip.write_bytes(b"fake-wav")

    rc = si_cli.main([
        "library", "--library", str(lib),
        "add", "dave", str(clip),
        "--no-rebuild",
        "--display-name", "戴夫",
        "--alias", "Dave",
        "--alias", "dave-alt",
        "--notes", "test notes with 中文",
    ])
    assert rc == 0
    prof = json.loads((lib / "dave" / "profile.json").read_text(encoding="utf-8"))
    assert prof["speaker_id"] == "dave"
    assert prof["display_name"] == "戴夫"
    assert prof["aliases"] == ["Dave", "dave-alt"]
    assert "中文" in prof["notes"]


def test_library_add_preserves_existing_profile(tmp_path):
    """Second `add` to an existing speaker must NOT overwrite a manually-
    edited profile.json (re-enrollment with a new clip should be safe)."""
    import json
    lib = tmp_path / "library"
    (lib / "eve").mkdir(parents=True)
    orig = {
        "speaker_id": "eve",
        "display_name": "夏娃 (hand-edited)",
        "aliases": ["Eve"],
        "notes": "do not overwrite me",
    }
    (lib / "eve" / "profile.json").write_text(
        json.dumps(orig, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    clip = tmp_path / "sample.wav"
    clip.write_bytes(b"fake-wav")

    rc = si_cli.main([
        "library", "--library", str(lib),
        "add", "eve", str(clip),
        "--no-rebuild",
        "--display-name", "SHOULD-NOT-APPLY",  # ignored because profile exists
    ])
    assert rc == 0
    prof = json.loads((lib / "eve" / "profile.json").read_text(encoding="utf-8"))
    assert prof["display_name"] == "夏娃 (hand-edited)"
    assert prof["notes"] == "do not overwrite me"


def test_library_add_without_profile_flags_leaves_no_profile(tmp_path):
    """If no display-name/alias/notes provided and no existing profile,
    library.py's embedder step will create a default one later. The add
    step itself should NOT create an empty profile that might mask a
    richer one from rebuild."""
    lib = tmp_path / "library"
    clip = tmp_path / "sample.wav"
    clip.write_bytes(b"fake-wav")
    rc = si_cli.main([
        "library", "--library", str(lib),
        "add", "frank", str(clip),
        "--no-rebuild",
    ])
    assert rc == 0
    assert not (lib / "frank" / "profile.json").exists(), (
        "no flags given → no profile.json should be created by `add`"
    )


# ---------- library score: cosine cross-match against cached embeddings ---


@pytest.fixture
def mini_library(tmp_path):
    """Build a tiny fake library with 2 enrolled speakers + a recording
    cache containing 3 SPEAKERs whose embeddings approximate 2 of the
    library centroids plus one unknown-ish vector."""
    np = pytest.importorskip("numpy")
    lib = tmp_path / "library"
    runs = tmp_path / "runs"
    rid = "test-rec"
    (runs / rid).mkdir(parents=True)

    # 256-d embeddings (matches pyannote dim) — use sparse one-hot-ish
    # vectors so cosines are predictable.
    dim = 256
    def _vec(idx):
        v = np.zeros(dim, dtype=np.float32)
        v[idx] = 1.0
        return v

    alice_emb = _vec(0)
    bob_emb = _vec(1)

    import json
    for sid, emb, disp in (
        ("alice", alice_emb, "爱丽丝"),
        ("bob", bob_emb, "鲍勃"),
    ):
        d = lib / sid
        d.mkdir(parents=True)
        np.save(d / "embedding.npy", emb)
        (d / "profile.json").write_text(
            json.dumps({"speaker_id": sid, "display_name": disp}, ensure_ascii=False),
            encoding="utf-8",
        )

    # Cached diar embeddings:
    # SPEAKER_00 very close to alice (0.99 cos), SPEAKER_01 to bob (0.99),
    # SPEAKER_02 off-axis (below 0.5 cos to either).
    np.savez(
        runs / rid / "diarization_pyannote.embeddings.npz",
        SPEAKER_00=alice_emb + 0.1 * _vec(2),  # tiny perturbation
        SPEAKER_01=bob_emb + 0.1 * _vec(2),
        SPEAKER_02=_vec(5),  # totally different — no alice/bob signal
    )
    return lib, runs, rid


def test_library_score_matches_close_speakers(mini_library, capsys):
    lib, runs, rid = mini_library
    rc = si_cli.main([
        "library", "--library", str(lib),
        "score", "--runs", str(runs), "--recording-id", rid,
    ])
    out = capsys.readouterr().out
    assert rc == 0
    # Header present.
    assert "SPEAKER" in out and "best" in out
    # Close matches labelled with Chinese display names.
    assert "爱丽丝" in out  # alice's display_name
    assert "鲍勃" in out    # bob's display_name
    # SPEAKER_02 has no real match → flagged UNMATCHED (below 0.5 threshold).
    assert "UNMATCHED" in out


def test_library_score_requires_embeddings_cache(tmp_path, capsys):
    """Missing cache produces a clear error, not a stack trace."""
    lib = tmp_path / "library"
    lib.mkdir()
    runs = tmp_path / "runs"
    runs.mkdir()
    rc = si_cli.main([
        "library", "--library", str(lib),
        "score", "--runs", str(runs), "--recording-id", "nonexistent",
    ])
    assert rc != 0
    err = capsys.readouterr().err
    assert "embeddings cache missing" in err


def test_library_score_picks_latest_when_no_id(mini_library, capsys):
    """Without --recording-id / --audio, score uses the most-recently-used
    cache subdir under --runs. Convenient for iteration."""
    lib, runs, rid = mini_library
    rc = si_cli.main([
        "library", "--library", str(lib),
        "score", "--runs", str(runs),
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert f"most recent cache: {rid}" in out


# ---------- --recording-id wiring for run / inspect -----------------------


def test_run_accepts_recording_id_override_in_parser():
    """The `--recording-id` flag must be exposed on `si run` so users can
    re-execute downstream stages after the original audio has been
    renamed or archived."""
    parser = si_cli.build_parser()
    text = parser.format_help()
    # Args description mentions the flag at the top level (subparsers are
    # hidden from the top-level help but the choice is there).
    run_parser = parser._subparsers._group_actions[0].choices["run"]  # noqa
    run_help = run_parser.format_help()
    assert "--recording-id" in run_help


def test_render_copy_uses_audio_stem_not_recording_id(tmp_path, capsys, monkeypatch):
    """When ``--recording-id`` aims at a stale cache (e.g. after archive
    rename), the ``--output-transcript`` copy must land as ``<audio.stem>.md``
    — the canonical vault name — not as ``<recording_id>.md``. Previously
    xyb had to `mv` the file by hand after every re-render, which this
    test guards against."""
    import json as _json
    from voicememowhisper.si import contracts
    from voicememowhisper.si.pipeline import run

    # Set up: a stale cache keyed by the ORIGINAL stem, and an archived
    # audio file whose name differs (matches the vault canonical form).
    runs = tmp_path / "runs"
    output = tmp_path / "output"
    transcript_dir = tmp_path / "transcripts"
    cache_rid = "20260420 110030"            # stale cache key
    audio_stem = "2026-04-20_11-00-30_my-meeting"   # canonical vault stem

    cache = runs / cache_rid
    cache.mkdir(parents=True)
    # Upstream fake caches — current run() loads them on every --from 5
    # invocation even though stage 5 only reads merged.json. That extra
    # load is not what this test is probing; just satisfy it.
    transcript = contracts.Transcript(
        recording_id=cache_rid, backend="test", model="m", language="zh",
        duration_sec=60.0, segments=[],
    )
    transcript.to_json(cache / "transcript_faster_whisper.json")
    diar = contracts.Diarization(
        recording_id=cache_rid, backend="test", model="m",
        num_speakers=1, segments=[],
    )
    diar.to_json(cache / "diarization_pyannote.json")
    merged = contracts.MergedTranscript(
        recording_id=cache_rid,
        duration_sec=60.0,
        language="zh",
        pipeline="test",
        segments=[
            contracts.MergedSegment(
                start=0.0, end=10.0, text="hello",
                speaker_label="SPEAKER_00", speaker_name="alice",
                confidence=0.9, needs_review=False,
            )
        ],
    )
    merged.to_json(cache / "merged.json")

    audio = tmp_path / f"{audio_stem}.m4a"
    audio.write_bytes(b"fake")

    run(
        audio, runs_dir=runs, output_dir=output,
        output_transcript_dir=transcript_dir,
        recording_id=cache_rid,
        from_step=5, to_step=5, force=True,
    )

    # The vault copy must use audio.stem (canonical), NOT cache_rid (stale).
    assert (transcript_dir / f"{audio_stem}.md").exists()
    assert (transcript_dir / f"{audio_stem}.txt").exists()
    assert not (transcript_dir / f"{cache_rid}.md").exists(), (
        "regression: output-transcript copy used the cache key "
        "instead of the audio's canonical stem"
    )
    # And the durable archive under output/<cache_rid>/ stays keyed by
    # the recording id (that's the single cache identity).
    assert (output / cache_rid / "transcript.md").exists()


def test_default_library_lives_under_documents_not_local_share():
    """Regression guard for the 2026-04-21 relocation: speaker library
    belongs next to Audio/ and Transcripts/ under ~/Documents/, so a
    single backup of that folder covers every hand-curated asset. The
    runs/ and outputs/ caches stay under ~/.local/share/ (rebuildable).
    """
    from voicememowhisper.si.cli import DEFAULT_LIBRARY, DEFAULT_RUNS, DEFAULT_OUTPUT
    from voicememowhisper.config import Settings

    expected_lib = Path.home() / "Documents" / "VoiceMemoWhisper" / "speaker-library"
    expected_runs = Path.home() / ".local" / "share" / "voicememowhisper" / "speaker-id" / "runs"
    expected_output = Path.home() / ".local" / "share" / "voicememowhisper" / "speaker-id" / "outputs"

    assert DEFAULT_LIBRARY == expected_lib
    assert DEFAULT_RUNS == expected_runs
    assert DEFAULT_OUTPUT == expected_output

    # Settings must match — these are the defaults the watcher's
    # SpeakerPipeline reads. If these diverge from cli.py's defaults,
    # `voicememo-whisper` and `voicememo-whisper si` disagree about
    # where the library lives.
    s = Settings()
    assert s.speaker_library_dir == expected_lib
    assert s.speaker_runs_dir == expected_runs
    assert s.speaker_output_dir == expected_output


def test_inspect_uses_recording_id_override(tmp_path, capsys):
    """When audio stem != cache dir (e.g. after archive rename), an
    explicit --recording-id should point `inspect` at the right cache."""
    runs = tmp_path / "runs"
    rid = "original-stem"
    (runs / rid).mkdir(parents=True)
    (runs / rid / "transcript_faster_whisper.json").write_text("{}")
    output = tmp_path / "output"
    (output / rid).mkdir(parents=True)

    # Audio file with a DIFFERENT stem than rid.
    audio = tmp_path / "archived-different-name.m4a"
    audio.write_bytes(b"fake")

    rc = si_cli.main([
        "inspect", str(audio),
        "--runs", str(runs), "--output", str(output),
        "--recording-id", rid,
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert rid in out  # used rid, not audio.stem
    assert "archived-different-name" not in out.split("recording:")[1].split("\n")[0]
