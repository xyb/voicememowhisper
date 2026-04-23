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
from typing import Iterator


DEFAULT_LOCK_PATH = Path.home() / ".local/state/voicememowhisper/main.lock"


class LockHeldError(RuntimeError):
    """Raised when another instance already holds the lock."""

    def __init__(self, holder_info: dict):
        self.holder_info = holder_info
        pid = holder_info.get("pid", "?")
        argv = holder_info.get("argv", "?")
        started = holder_info.get("started_at", "?")
        super().__init__(
            f"Another voicememowhisper instance is already running "
            f"(PID {pid}, started {started}, argv: {argv}). "
            f"If it's stuck, kill it first: `kill {pid}`."
        )


@contextlib.contextmanager
def single_instance_lock(
    lock_path: Path | None = None,
    *,
    exit_on_conflict: bool = True,
) -> Iterator[None]:
    """Hold an exclusive advisory lock for the duration of the block.

    Parameters
    ----------
    lock_path
        File to lock. Defaults to ``~/.local/state/voicememowhisper/main.lock``.
    exit_on_conflict
        If True (default), print a message to stderr and call ``sys.exit(1)``
        when the lock is already held. If False, raise ``LockHeldError``.
    """
    path = Path(lock_path) if lock_path is not None else DEFAULT_LOCK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            holder = _read_holder_info(path)
            err = LockHeldError(holder)
            if exit_on_conflict:
                print(f"voicememowhisper: {err}", file=sys.stderr)
                os.close(fd)
                sys.exit(1)
            else:
                os.close(fd)
                raise err

        info = {
            "pid": os.getpid(),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "argv": " ".join(sys.argv),
        }
        # Truncate + rewrite. We hold the exclusive lock so no reader
        # can get a half-written view; read_holder_info also opens with
        # a shared lock so it waits for us if we're mid-write.
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, (json.dumps(info) + "\n").encode("utf-8"))
        try:
            yield
        finally:
            # flock releases automatically on close / process death, but
            # zero the file on graceful exit so stale holder info doesn't
            # confuse a reader after we're gone.
            try:
                os.ftruncate(fd, 0)
            except OSError:
                pass
            os.close(fd)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def _read_holder_info(path: Path) -> dict:
    """Best-effort read of the current holder's metadata. Returns {}
    if the file is empty or unreadable (which can happen transiently
    between ``truncate`` and ``write`` in the holder)."""
    try:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return {}
        return json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return {}
