"""Progress reporting that adapts to TTY vs non-TTY output.

Why two modes?

- **TTY / interactive**: a human is watching. We delegate to `tqdm`
  per stage — standard bar, overwrites itself, shows elapsed + ETA.
  The internal library handles it well; we don't try to be clever.
- **Non-TTY**: a log file, an AI agent tailing output, a pipe. ANSI
  animation is useless there. Instead we emit **one compact line per
  event**, at most every ``min_interval_s`` (default 60s) so the reader
  can tell "still alive" without drowning in lines.

Non-TTY line format fuses both layers of context on a single line:

    [<i>/<N> <stage>] <sub>  <pct>% (<cur>/<total>) · elapsed <t> · eta <t>

Examples::

    [1/5 transcribe]  28% (12m30s/45m02s) · elapsed 5m10s · eta 13m02s
    [2/5 diarize] speaker_segmentation  45% (900/2000) · elapsed 2m15s
    [2/5 diarize] embeddings · elapsed 8m22s
    [5/5 render] done · elapsed 0.3s

The format is chosen to be readable to humans grepping logs but short
enough to be cheap for an AI reader — a 45-min run emits roughly 1 line
per minute (transcribe) plus 3–6 sub-step announcements (diarize),
totalling tens of lines, not hundreds.

Environment escape hatches:

- ``VOICE_MEMO_PROGRESS_MODE=plain`` forces non-TTY mode (useful in tests)
- ``VOICE_MEMO_PROGRESS_MODE=tty`` forces TTY mode
- ``VOICE_MEMO_PROGRESS_INTERVAL=<seconds>`` overrides the throttle (float)
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional


# ───────────────────────── helpers ────────────────────────────────────


def _fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "?"
    s = int(max(0, seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{sec:02d}s"
    return f"{m}m{sec:02d}s"


def _is_tty() -> bool:
    forced = os.environ.get("VOICE_MEMO_PROGRESS_MODE")
    if forced == "plain":
        return False
    if forced == "tty":
        return True
    return sys.stderr.isatty()


def _default_interval() -> float:
    raw = os.environ.get("VOICE_MEMO_PROGRESS_INTERVAL")
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return 60.0


# ───────────────────────── StageProgress ──────────────────────────────


class StageProgress:
    """Per-stage progress tracker with TTY-adaptive output.

    Parameters
    ----------
    label
        Short stage name, e.g. ``"transcribe"``. Shown in every emitted
        line.
    stage_num, total_stages
        Position within the full pipeline. When set, the non-TTY header
        reads ``[1/5 transcribe]`` so the reader sees pipeline-level
        context without a second line. When ``None`` the prefix omits
        the x/N.
    total
        Known total of the inner progress axis. For ``transcribe`` this
        is audio seconds; for stages without a meaningful total (e.g.
        ``diarize``), pass ``None`` and the line drops the percent and
        ETA.
    unit
        Display unit for ``total``/``current``. ``"s"`` renders as
        ``m:ss``; anything else is printed as a raw integer (used for
        sub-step counters like ``900/2000``).
    min_interval_s
        Non-TTY only. Minimum seconds between emitted lines. Default
        from ``VOICE_MEMO_PROGRESS_INTERVAL`` or 60s.

    Example
    -------
        with StageProgress("transcribe", 1, 5, total=duration) as prog:
            for seg in segments_iter:
                prog.update(seg.end)
    """

    def __init__(
        self,
        label: str,
        stage_num: Optional[int] = None,
        total_stages: Optional[int] = None,
        total: Optional[float] = None,
        unit: str = "s",
        min_interval_s: Optional[float] = None,
    ) -> None:
        self.label = label
        self.stage_num = stage_num
        self.total_stages = total_stages
        self.total = total
        self.unit = unit
        self.min_interval_s = (
            min_interval_s if min_interval_s is not None else _default_interval()
        )
        self.current: float = 0.0
        self._substep: str = ""
        self._substep_total: Optional[float] = None
        self._substep_current: float = 0.0
        self._start: float = 0.0
        self._last_emit_time: float = 0.0
        self._bar = None  # tqdm instance when in TTY mode
        self._tty = _is_tty()

    # public -----------------------------------------------------------

    def __enter__(self) -> "StageProgress":
        self._start = time.monotonic()
        self._last_emit_time = self._start
        if self._tty:
            self._bar = self._make_tqdm()
            if self._bar is None:
                # tqdm missing — fall back to non-TTY behaviour.
                self._tty = False
                self._emit_start_line()
        else:
            self._emit_start_line()
        return self

    def update(self, current: float) -> None:
        """Set cumulative progress (not a delta).

        If a sub-step was announced with its own total, ``current`` is
        interpreted against that sub-step total. Otherwise it advances
        the outer stage progress.
        """
        current = max(0.0, float(current))
        if self._substep_total is not None:
            self._substep_current = current
        else:
            self.current = current
        if self._bar is not None:
            # Stage bar only advances on outer progress; sub-step counts
            # go into the postfix so the bar frame stays meaningful.
            if self._substep_total is not None:
                try:
                    self._bar.set_postfix_str(
                        f"{self._substep} {int(current)}/{int(self._substep_total)}",
                        refresh=False,
                    )
                except Exception:
                    pass
            else:
                delta = current - self.current
                if delta > 0:
                    self._bar.update(delta)
            return
        now = time.monotonic()
        if now - self._last_emit_time >= self.min_interval_s:
            self._emit_status_line(now)

    def set_substep(self, name: str, total: Optional[float] = None) -> None:
        """Announce a new within-stage sub-step.

        ``total`` (optional) is the sub-step's own progress scale; when
        set, subsequent ``update(n)`` calls report ``n/total`` for this
        sub-step rather than advancing the outer stage counter.

        Non-TTY: emits a status line *immediately* (resetting the throttle
        clock) so readers see the change. TTY: updates the tqdm bar's
        postfix so the sub-step name follows the bar.
        """
        if name == self._substep and total == self._substep_total:
            return
        self._substep = name
        self._substep_total = total
        self._substep_current = 0.0
        if self._bar is not None:
            try:
                self._bar.set_postfix_str(name, refresh=True)
            except Exception:
                pass
            return
        self._emit_status_line(time.monotonic())

    def note(self, message: str) -> None:
        """Emit an informational line without changing progress counters."""
        if self._bar is not None:
            try:
                self._bar.write(f"{self._prefix()} {message}", file=sys.stderr)
            except Exception:
                print(f"{self._prefix()} {message}", file=sys.stderr, flush=True)
        else:
            print(f"{self._prefix()} {message}", file=sys.stderr, flush=True)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None
            return
        if exc_type is None:
            elapsed = time.monotonic() - self._start
            print(
                f"{self._prefix()} done · elapsed {_fmt_duration(elapsed)}",
                file=sys.stderr, flush=True,
            )

    # internals --------------------------------------------------------

    def _prefix(self) -> str:
        if self.stage_num is not None and self.total_stages is not None:
            return f"[{self.stage_num}/{self.total_stages} {self.label}]"
        return f"[{self.label}]"

    def _fmt_value(self, v: float) -> str:
        if self.unit == "s":
            return _fmt_duration(v)
        return str(int(v))

    def _emit_start_line(self) -> None:
        total_str = self._fmt_value(self.total) if self.total is not None else "?"
        print(
            f"{self._prefix()} start · total {total_str}",
            file=sys.stderr, flush=True,
        )

    def _emit_status_line(self, now: float) -> None:
        elapsed = now - self._start
        sub = f" {self._substep}" if self._substep else ""

        # Substep-scoped progress wins when a substep has its own total.
        if self._substep_total and self._substep_total > 0:
            sub_pct = self._substep_current / self._substep_total * 100
            print(
                f"{self._prefix()}{sub} {sub_pct:3.0f}% "
                f"({int(self._substep_current)}/{int(self._substep_total)}) "
                f"· elapsed {_fmt_duration(elapsed)}",
                file=sys.stderr, flush=True,
            )
        elif self.total and self.total > 0 and self.current > 0:
            pct = self.current / self.total * 100
            rate = self.current / elapsed if elapsed > 0 else 0
            eta = (self.total - self.current) / rate if rate > 0 else 0
            print(
                f"{self._prefix()}{sub} {pct:3.0f}% "
                f"({self._fmt_value(self.current)}/{self._fmt_value(self.total)}) "
                f"· elapsed {_fmt_duration(elapsed)} · eta {_fmt_duration(eta)}",
                file=sys.stderr, flush=True,
            )
        else:
            print(
                f"{self._prefix()}{sub} · elapsed {_fmt_duration(elapsed)}",
                file=sys.stderr, flush=True,
            )
        self._last_emit_time = now

    def _make_tqdm(self):
        try:
            from tqdm import tqdm  # type: ignore[import-not-found]
        except ImportError:
            return None
        desc = self._prefix()
        if self.total is None:
            # Indeterminate bar (spinner + elapsed) — for diarize.
            return tqdm(
                total=None,
                desc=desc,
                bar_format="{desc} {elapsed}{postfix}",
                file=sys.stderr,
                mininterval=0.5,
            )
        return tqdm(
            total=self.total,
            unit=self.unit,
            desc=desc,
            bar_format=(
                "{desc} {percentage:3.0f}%|{bar}| "
                "{n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}"
            ),
            file=sys.stderr,
            mininterval=0.5,
        )


# ───────────────────────── DiarizeProgressHook ────────────────────────


class DiarizeProgressHook:
    """Bridge pyannote.audio 4.0's hook protocol onto ``StageProgress``.

    pyannote invokes the hook across internal sub-steps
    (``speaker_segmentation``, ``embeddings``, ``discrete_diarization``
    …). Each sub-step optionally reports ``total``/``completed`` counts.

    This adapter:
    - calls ``prog.set_substep(name)`` on every sub-step transition
      (which emits one non-TTY line immediately / updates the tqdm
      postfix in TTY mode),
    - forwards the ``completed`` counter to ``prog.update()`` so the
      outer throttle fires a normal progress line (non-TTY) / the bar
      advances (TTY) at most once per ``min_interval_s``.
    """

    def __init__(self, progress: StageProgress) -> None:
        self._prog = progress
        # Each sub-step has its own counter starting from 0; we don't
        # want diarize's "total" to be the sum. Instead we let the bar
        # stay indeterminate (total=None passed to StageProgress) and
        # report each sub-step's completed/total via `note()`-style
        # wording in the line prefix.
        self._current_step: Optional[str] = None

    def __enter__(self) -> "DiarizeProgressHook":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return None

    def __call__(
        self,
        step_name: str,
        step_artifact=None,
        file=None,
        total: Optional[int] = None,
        completed: Optional[int] = None,
    ) -> None:
        if step_name != self._current_step:
            self._current_step = step_name
            # Pass the sub-step's own total so the emitted line reads
            # ``<step> NN% (cur/total)`` instead of just elapsed.
            self._prog.set_substep(step_name, total=total)
        if total is not None and completed is not None and total > 0:
            # Let the outer throttle gate decide when to emit a line.
            self._prog.update(float(completed))


__all__ = ["StageProgress", "DiarizeProgressHook"]
