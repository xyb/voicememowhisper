"""Single-instance lock for voicememowhisper processing entry points.

One pain point we kept running into: the user (or an AI assistant
driving the user's shell) starts the main flow once, then — not
realizing it's still running in another window or backgrounded —
starts a second one. Both processes then:

- hit the same Voice Memos source files,
- race against the same state DB,
- race against the same archive destination.

The symptoms are horrible and non-deterministic: half-archived
files, duplicate transcripts, mangled state.

This module provides a small advisory-lock helper around a single
lockfile under ``~/.local/state/voicememowhisper/main.lock``. On
acquire we use ``fcntl.flock(LOCK_EX | LOCK_NB)`` so the lock is
held for the lifetime of the process and released automatically
when it exits — even on ``kill -9``. That avoids stale-lock issues
that PID-only schemes suffer from.

We write the PID, start time and the argv joined with spaces into
the lockfile at acquire time, so when a second instance fails to
acquire, we can print an actionable message pointing the user at
the exact competing process.

Usage::

    from ._lock import single_instance_lock

    with single_instance_lock():
        run_the_long_job()

If the lock is already held, the context manager exits by
``sys.exit(1)`` with a message on stderr. Call sites that need to
handle that themselves can catch ``LockHeldError`` instead of
letting it turn into ``SystemExit``.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Iterator


DEFAULT_LOCK_PATH = Path.home() / ".local/state/voicememowhisper/main.lock"

# Persistent per-run history. The lock file above is zeroed on graceful exit,
# so it only ever describes the *currently running* instance — useless for
# after-the-fact debugging. This append-only JSONL keeps a durable record:
# one "start" line when a run acquires the lock and one "end" line when it
# releases (with wall time + outcome). A run that crashed/was killed leaves a
# "start" with no matching "end" — exactly the signal you want when something
# hung. ponytail: unbounded append (~200 B/run); add rotation if it ever
# crosses a few MB — that's thousands of runs away.
RUN_LOG_PATH = Path.home() / ".local/state/voicememowhisper/runs.jsonl"


def _append_run_log(record: dict) -> None:
    """Append one JSON line to RUN_LOG_PATH. Best-effort: logging must never
    break an actual run, so every failure here is swallowed."""
    try:
        RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RUN_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _run_log_base(info: dict) -> dict:
    """Fields shared by a run's start and end records, pulled from the holder
    info so start/end lines correlate by pid + started_epoch."""
    return {
        "ts": info.get("started_at"),
        "started_epoch": info.get("started_epoch"),
        "pid": info.get("pid"),
        "audio_name": info.get("audio_name"),
        "audio_duration_sec": info.get("audio_duration_sec"),
        "est_total_sec": info.get("est_total_sec"),
        "argv": info.get("argv"),
    }

# Empirical wall-clock cost of the full pipeline (transcribe + diarize +
# identify + merge + render), as a fraction of the audio's own duration, with
# the default openai-audio HTTP backend: a ~30-min segment took ~11 min of wall
# time ⇒ ~0.38. Rounded up a touch so a waiting caller isn't told to retry *too*
# early. Retune if the backend or hardware changes.
_PIPELINE_WALL_PER_AUDIO = 0.4
_PIPELINE_FIXED_OVERHEAD_SEC = 20.0


def job_info_for_audio(audio_path) -> dict | None:
    """Build the lock ``job_info`` for a single-file run: the audio's duration and
    name, so a second instance can estimate when this run finishes. Best-effort —
    returns None if the duration can't be probed. Import listing lazily to keep this
    low-level lock module free of a hard dependency on it.

    Pass this (or a lambda calling it) so the probe runs only *after* the lock is
    acquired — a second instance that loses the race never spends time probing."""
    try:
        from .listing import probe_audio_duration_seconds
        dur = probe_audio_duration_seconds(Path(audio_path))
        if dur:
            return {"audio_duration_sec": dur, "audio_name": Path(audio_path).name}
    except Exception:
        pass
    return None


def _humanize_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s" if secs else f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _holder_elapsed_seconds(holder_info: dict) -> float | None:
    """Seconds since the holder acquired the lock, or None if unknown."""
    epoch = holder_info.get("started_epoch")
    if isinstance(epoch, (int, float)):
        return max(0.0, time.time() - float(epoch))
    started = holder_info.get("started_at")
    if started:
        try:
            t0 = time.mktime(time.strptime(started, "%Y-%m-%d %H:%M:%S"))
            return max(0.0, time.time() - t0)
        except (ValueError, OverflowError):
            return None
    return None


def _estimate_wait(holder_info: dict) -> str | None:
    """Best-effort human string estimating how long until the current
    holder finishes and it's worth retrying. None if we can't tell.
    Never raises — a wait estimate must not break the real error path."""
    try:
        elapsed = _holder_elapsed_seconds(holder_info)
        if elapsed is None:
            return None
        dur = holder_info.get("audio_duration_sec")
        if isinstance(dur, (int, float)) and dur > 0:
            # Prefer the estimate the holder already stored (single source of truth);
            # fall back to computing it for older-format locks that predate est_total_sec.
            stored = holder_info.get("est_total_sec")
            est_total = stored if isinstance(stored, (int, float)) and stored > 0 else \
                dur * _PIPELINE_WALL_PER_AUDIO + _PIPELINE_FIXED_OVERHEAD_SEC
            remaining = est_total - elapsed
            audio_h = _humanize_seconds(dur)
            if remaining <= 0:
                return (
                    f"running {_humanize_seconds(elapsed)}, already past the "
                    f"~{_humanize_seconds(est_total)} estimate for {audio_h} of "
                    f"audio — should finish any moment; retry in ~30s"
                )
            return (
                f"running {_humanize_seconds(elapsed)} of an estimated "
                f"~{_humanize_seconds(est_total)} ({audio_h} of audio) — "
                f"retry in ~{_humanize_seconds(remaining)}"
            )
        # Duration unknown (backlog/watch mode, or an older-format lock):
        # report elapsed and give a generic retry hint.
        return (
            f"running {_humanize_seconds(elapsed)} (audio length unknown) — "
            f"retry in ~2m"
        )
    except Exception:
        return None


class LockHeldError(RuntimeError):
    """Raised when another instance already holds the lock."""

    def __init__(self, holder_info: dict):
        self.holder_info = holder_info
        pid = holder_info.get("pid", "?")
        argv = holder_info.get("argv", "?")
        started = holder_info.get("started_at", "?")
        wait = _estimate_wait(holder_info)
        wait_line = f" {wait}." if wait else ""
        super().__init__(
            f"Another voicememowhisper instance is already running "
            f"(PID {pid}, started {started}).{wait_line} "
            f"argv: {argv}. "
            f"If it's stuck, kill it first: `kill {pid}`."
        )


# --- compute lock -----------------------------------------------------------
#
# The lock above exists to stop two runs racing on the *state DB / archive*, so
# it was only ever taken by the entry points that write those. But every ML
# stage also saturates CPU and holds 3-6 GB RSS, and that constraint is
# orthogonal: `si library find-candidates --vet`, `si closeup` and friends touch
# no shared state, took no lock — and happily ran a second pyannote alongside a
# first one, thrashing the machine (2026-07-14: two concurrent find-candidates
# pinned the CPU until the user killed them by hand).
#
# Gating the *subcommands* would just move the footgun: whoever adds the next
# subcommand has to remember. So the gate lives at the point where a heavy model
# is actually loaded. Any code path that loads whisper/pyannote — existing or
# not yet written — passes through here and is serialized for free.
#
# Held for the life of the process, not a block: the memory stays resident and
# the compute keeps running long after the model object is constructed, so
# releasing at the end of a `with` would let a second process pile in mid-run.
# The fd is deliberately never closed — the OS drops the flock on exit, kill -9
# included.

_HELD_FD: int | None = None


def _mark_held(fd: int) -> None:
    """Record that this process holds the lock, so a later
    acquire_compute_lock() inside the same process is a no-op instead of
    self-deadlocking. flock is per open-file-description, not per process: a
    second open() of the same path in the same process would block on our own
    lock."""
    global _HELD_FD
    _HELD_FD = fd


def _clear_held(fd: int) -> None:
    global _HELD_FD
    if _HELD_FD == fd:
        _HELD_FD = None


def acquire_compute_lock(lock_path: Path | None = None, *, what: str = "a model") -> None:
    """Serialize heavy ML work machine-wide. Call immediately before loading a
    whisper / pyannote model.

    No-op if this process already holds the lock (e.g. the main flow or `si run`
    took it via single_instance_lock, and now a stage underneath loads a model).

    If another *process* holds it, print who and exit — refusing to start is the
    whole point. Bailing out early is also the kind thing to do: two of these
    running at once is slower than either alone, so there is nothing to gain by
    waiting in-process.
    """
    global _HELD_FD
    if _HELD_FD is not None:
        return

    path = Path(lock_path) if lock_path is not None else DEFAULT_LOCK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        holder = _read_holder_info(path)
        os.close(fd)
        err = LockHeldError(holder)
        print(
            f"voicememo-whisper: refusing to load {what} — {err}\n"
            f"  Heavy stages (whisper / pyannote) are serialized on purpose: each "
            f"saturates the CPU and holds several GB of RSS, so two at once is "
            f"slower than one after the other and can swap the machine to a halt.",
            file=sys.stderr,
        )
        sys.exit(1)

    info = {
        "pid": os.getpid(),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "started_epoch": time.time(),
        "argv": " ".join(sys.argv),
    }
    try:
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, (json.dumps(info) + "\n").encode("utf-8"))
    except OSError:
        pass   # holder info is a debugging nicety; the flock is what matters

    _mark_held(fd)   # fd stays open for the life of the process — see above


class _RunHandle:
    """Yielded by single_instance_lock so a body that exits the block normally
    (via ``return N`` or by swallowing its own KeyboardInterrupt) can still record
    the real run outcome for the run-log. Untouched → the run is logged "ok"."""

    __slots__ = ("outcome", "error")

    def __init__(self):
        self.outcome = None
        self.error = None

    def mark(self, outcome: str, error: str | None = None) -> None:
        self.outcome = outcome
        self.error = error


@contextlib.contextmanager
def single_instance_lock(
    lock_path: Path | None = None,
    *,
    exit_on_conflict: bool = True,
    job_info: dict | Callable[[], dict | None] | None = None,
) -> Iterator["_RunHandle"]:
    """Hold an exclusive advisory lock for the duration of the block.

    Parameters
    ----------
    lock_path
        File to lock. Defaults to ``~/.local/state/voicememowhisper/main.lock``.
    exit_on_conflict
        If True (default), print a message to stderr and call ``sys.exit(1)``
        when the lock is already held. If False, raise ``LockHeldError``.
    job_info
        Optional extra fields merged into the stored holder info (e.g.
        ``{"audio_duration_sec": 2280.0}``). A second instance uses these to
        estimate how long until this run finishes so it can tell the caller
        when to retry, instead of just "already running". May be a callable
        returning that dict — it is invoked only *after* the lock is acquired,
        so an instance that loses the race never pays for building it (e.g. an
        audio-duration probe subprocess).
    """
    path = Path(lock_path) if lock_path is not None else DEFAULT_LOCK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
    try:   # outermost: closes fd exactly once on every exit path (see final finally)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            holder = _read_holder_info(path)
            err = LockHeldError(holder)
            if exit_on_conflict:
                print(f"voicememowhisper: {err}", file=sys.stderr)
                sys.exit(1)
            raise err

        info = {
            "pid": os.getpid(),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "started_epoch": time.time(),
            "argv": " ".join(sys.argv),
        }
        # Resolve job_info only now that we hold the lock (a callable lets the
        # caller defer an expensive probe past the conflict check). Best-effort.
        if callable(job_info):
            try:
                job_info = job_info()
            except Exception:
                job_info = None
        if job_info:
            info.update(job_info)
        # Store the estimate at the source so a waiting second instance uses an
        # authoritative eta instead of re-deriving it with a hardcoded copy of the
        # constants below. Only when we know the audio length.
        _dur = info.get("audio_duration_sec")
        if _dur:
            _est = _dur * _PIPELINE_WALL_PER_AUDIO + _PIPELINE_FIXED_OVERHEAD_SEC
            info["est_total_sec"] = round(_est, 1)
            info["eta_epoch"] = round(info["started_epoch"] + _est, 3)
        # Truncate + rewrite. This open→write window is microseconds; a conflicting
        # reader that catches it mid-write sees an empty/partial file and retries a
        # few times (see _read_holder_info) rather than blocking on our lock.
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, (json.dumps(info) + "\n").encode("utf-8"))

        # Publish the fd so a model load underneath us (acquire_compute_lock)
        # sees the lock is already ours and skips it. Without this it would
        # open() the same path again and flock() would block on us — flock is
        # keyed by open-file-description, so a process can deadlock against
        # itself.
        _mark_held(fd)

        _append_run_log({**_run_log_base(info), "event": "start"})
        # Default "interrupted" covers any abnormal teardown that skips the excepts
        # below — GeneratorExit, other BaseException — so an aborted run never records
        # a false "ok". Only a clean completion (or an explicit exit 0) sets "ok".
        outcome, err_msg = "interrupted", None
        # A body that signals failure via `return N` or swallows its own
        # KeyboardInterrupt exits the block normally, so exceptions never reach the
        # classifier below — such a body calls run.mark(...) to record the real
        # outcome. Bodies that just raise/return-clean need not touch it.
        run = _RunHandle()
        try:
            yield run
            outcome = run.outcome or "ok"
            err_msg = run.error
        except SystemExit as exc:
            code = getattr(exc, "code", None)
            if code in (0, None):
                outcome = "ok"                       # sys.exit(0)/sys.exit() is a success exit
            else:
                outcome, err_msg = "error", f"SystemExit: {code}"
            raise
        except KeyboardInterrupt:
            outcome = "interrupted"
            raise
        except Exception as exc:   # noqa: BLE001 — record the outcome, then re-raise
            outcome = "error"
            err_msg = f"{type(exc).__name__}: {exc}"[:500]
            raise
        finally:
            _append_run_log({
                **_run_log_base(info), "event": "end", "outcome": outcome,
                "error": err_msg, "wall_sec": round(time.time() - info["started_epoch"], 1),
            })
            # flock releases automatically on close / process death, but
            # zero the file on graceful exit so stale holder info doesn't
            # confuse a reader after we're gone.
            try:
                os.ftruncate(fd, 0)
            except OSError:
                pass
    finally:
        # Single close for every path (lock acquired or not, clean or raised) —
        # closing twice risks closing an fd another thread has since reopened.
        _clear_held(fd)
        try:
            os.close(fd)
        except OSError:
            pass


def _read_holder_info(path: Path, _retries: int = 5) -> dict:
    """Best-effort read of the current holder's metadata. Returns {} if the file
    is empty or unreadable.

    A conflicting reader can land in the holder's ``ftruncate(0)``→``write`` window
    (microseconds) and see an empty or half-written file; retry a few times with a
    tiny sleep to ride that out. We deliberately do NOT take a shared flock: the
    holder keeps the exclusive lock for the *whole run*, so a blocking shared read
    would hang until the run finished — the opposite of the fail-fast retry hint
    this feeds."""
    for attempt in range(_retries):
        try:
            raw = path.read_text(encoding="utf-8").strip()
            if raw:
                return json.loads(raw)
        except (OSError, json.JSONDecodeError):
            pass
        if attempt < _retries - 1:
            time.sleep(0.02)
    return {}
