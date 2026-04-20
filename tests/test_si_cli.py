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
