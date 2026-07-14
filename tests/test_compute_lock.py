"""Heavy ML stages must be serialized machine-wide.

Every whisper/pyannote stage saturates the CPU and holds 3-6 GB RSS. Two at
once is slower than either alone and can swap the machine to a halt. The old
lock only guarded the entry points that wrote the state DB, so the read-only
ones (`si library find-candidates --vet`, `si closeup`) took no lock and ran a
second pyannote right alongside the first (2026-07-14: pinned the CPU until the
user killed both by hand).

The fix gates the *model load* rather than the subcommand, so a path that
doesn't exist yet is covered too. These tests pin that down:

1. it actually excludes a second process,
2. it does NOT self-deadlock when the process already holds the outer lock,
3. no model-load point escapes it — the one that catches the next new stage.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from voicememowhisper import _lock


SRC = Path(__file__).resolve().parent.parent / "voicememowhisper"


@pytest.fixture(autouse=True)
def _reset_held():
    """The held-fd is module-global; don't leak it between tests."""
    before = _lock._HELD_FD
    _lock._HELD_FD = None
    yield
    _lock._HELD_FD = before


def test_second_acquire_in_another_process_is_refused(tmp_path):
    lock = tmp_path / "main.lock"
    _lock.acquire_compute_lock(lock, what="a fake model")

    # A child inherits no flock, so it genuinely contends for the lock.
    child = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(f"""
            from pathlib import Path
            from voicememowhisper import _lock
            _lock.acquire_compute_lock(Path({str(lock)!r}), what="a fake model")
            print("ACQUIRED")
        """)],
        capture_output=True, text=True,
        cwd=str(SRC.parent),
    )
    assert child.returncode == 1, f"second process should be refused, got:\n{child.stdout}"
    assert "ACQUIRED" not in child.stdout
    assert "already running" in child.stderr


def test_reacquire_in_same_process_is_a_noop(tmp_path):
    """A stage loading two models (transcribe then diarize) must not block on
    the lock its own process is already holding. flock is keyed by open-file-
    description, so a naive second open()+flock() would deadlock against self."""
    lock = tmp_path / "main.lock"
    _lock.acquire_compute_lock(lock, what="model A")
    _lock.acquire_compute_lock(lock, what="model B")   # would hang if broken
    assert _lock._HELD_FD is not None


def test_model_load_under_single_instance_lock_does_not_deadlock(tmp_path):
    """The main flow / `si run` take single_instance_lock, then load models
    underneath it. That inner acquire must see the lock as already ours."""
    lock = tmp_path / "main.lock"
    with _lock.single_instance_lock(lock):
        _lock.acquire_compute_lock(lock, what="a model")   # would hang if broken
    # released on block exit
    assert _lock._HELD_FD is None


def test_lock_is_released_when_process_dies(tmp_path):
    """flock dies with the process — no stale-lock recovery needed, even on
    kill -9. Guards the choice of flock over a pidfile."""
    lock = tmp_path / "main.lock"
    child = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(f"""
            from pathlib import Path
            from voicememowhisper import _lock
            _lock.acquire_compute_lock(Path({str(lock)!r}), what="a model")
        """)],
        capture_output=True, text=True, cwd=str(SRC.parent),
    )
    assert child.returncode == 0
    # The child exited without ever releasing explicitly; we must still get it.
    _lock.acquire_compute_lock(lock, what="a model")
    assert _lock._HELD_FD is not None


# Constructors that pull a multi-GB model into RSS. Extend only alongside an
# acquire_compute_lock() at the new call site.
_HEAVY = re.compile(r"\bWhisperModel\(|\bPipeline\.from_pretrained\(")


def test_every_model_load_point_takes_the_compute_lock():
    """The guard that outlives us: gating subcommands would mean whoever adds
    the next one has to remember. Gating the model load means they don't — but
    only if every load point is actually gated. This fails the moment someone
    adds a WhisperModel(...) or Pipeline.from_pretrained(...) without the lock
    above it, which is exactly when a human would otherwise not notice."""
    offenders = []
    for py in SRC.rglob("*.py"):
        lines = py.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if not _HEAVY.search(line):
                continue
            # The lock must be taken somewhere earlier in the same file. Cheap
            # and slightly loose (it doesn't prove the same code path), but it
            # reliably catches a whole new load point added with no lock at all.
            preceding = "\n".join(lines[:i])
            if "acquire_compute_lock(" not in preceding:
                offenders.append(f"{py.relative_to(SRC)}:{i + 1}: {line.strip()}")

    assert not offenders, (
        "these load a heavy model without first taking the compute lock — two of "
        "them can then run at once and thrash the machine:\n  "
        + "\n  ".join(offenders)
    )
