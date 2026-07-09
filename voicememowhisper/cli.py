from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import shutil
from datetime import datetime
from dataclasses import replace
from pathlib import Path

from .config import Settings, load_settings, DEFAULT_ARCHIVE_PATH, DEFAULT_TRANSCRIPT_PATH, DEFAULT_STATE_DB_PATH
from .listing import collect_recordings, format_list_output
from . import listing as listing_mod
from .state import StateStore

LOGGER = logging.getLogger("cli")


class _VerbosityFilter(logging.Filter):
    """
    Allow INFO/DEBUG logs based on -v/-vv.

    - WARNING/ERROR/CRITICAL always shown
    - INFO shown when verbosity >= record.verbosity (default 1)
    - DEBUG shown when verbosity >= record.verbosity (default 2)
    """

    def __init__(self, verbosity: int) -> None:
        super().__init__()
        self._verbosity = verbosity

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        required = getattr(record, "verbosity", None)
        if required is None:
            required = 2 if record.levelno <= logging.DEBUG else 1
        return self._verbosity >= int(required)


def _default_log_dir() -> Path:
    """Where rotated log files live. Override with ``VOICE_MEMO_LOG_DIR``."""
    override = os.environ.get("VOICE_MEMO_LOG_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".local" / "state" / "voicememowhisper" / "logs"


def _setup_file_logging(log_dir: Path) -> Path | None:
    """Attach a daily-rotating file handler to the root logger that captures
    everything at DEBUG level — independent of CLI ``-v`` verbosity. Returns
    the active log file path (or None on failure). Goal: "looks fine on
    stdout but actually crashed" issues (e.g. backend HTTP failures that
    fall back to a local model and silently emit a degraded transcript)
    stay debuggable after the fact without rerunning.
    """
    import logging.handlers as _lh

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None

    log_path = log_dir / "voicememowhisper.log"
    try:
        handler = _lh.TimedRotatingFileHandler(
            str(log_path),
            when="midnight",
            interval=1,
            backupCount=14,
            encoding="utf-8",
            utc=False,
            delay=False,
        )
    except OSError:
        return None

    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
            # More detail than stdout: full timestamp + ms + logger name +
            # source location (file:line) for fast jump-to-source.
            "%(asctime)s.%(msecs)03d %(levelname)-7s [%(name)s] "
            "%(filename)s:%(lineno)d %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(handler)
    return log_path


def _configure_logging(level: str, verbosity: int) -> None:
    # Force line-buffered stdout/stderr so log lines flush immediately even
    # when redirected to a file (`> run.log 2>&1`). Without this, Python uses
    # 4 KB block buffering on non-tty streams and progress is invisible for
    # minutes during long-running stages.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass

    logging.basicConfig(
        # Keep root level permissive; filter controls what is printed.
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    root = logging.getLogger()
    for handler in root.handlers:
        handler.addFilter(_VerbosityFilter(verbosity))
        # Wrap emit() to flush after every record, in case the underlying
        # stream falls back to block buffering.
        if isinstance(handler, logging.StreamHandler):
            _orig_emit = handler.emit
            def _flushing_emit(record, _orig=_orig_emit, _h=handler):
                _orig(record)
                try:
                    _h.flush()
                except Exception:
                    pass
            handler.emit = _flushing_emit  # type: ignore[assignment]

    log_path = _setup_file_logging(_default_log_dir())
    if log_path is not None:
        # Make the file path visible up front so users (and AI assistants
        # debugging an issue) know where to look without grepping the source.
        logging.info("Detailed log file: %s", log_path, extra={"verbosity": 1})


def build_settings(args: argparse.Namespace) -> Settings:
    settings = load_settings()
    overrides = {}
    if args.model:
        overrides["whisperkit_model"] = args.model
    if args.language:
        overrides["language"] = args.language
    if args.newest_first is not None:
        overrides["processing_order"] = "newest-first" if args.newest_first else "oldest-first"
    
    if args.transcript_dir:
        overrides["transcript_dir"] = Path(args.transcript_dir).expanduser()

    # Archiving configuration (default enabled; use --no-archive to disable)
    overrides["archive_enabled"] = bool(getattr(args, "archive", True))
    if args.archive_dir:
        overrides["archive_dir"] = Path(args.archive_dir).expanduser()
        overrides["archive_enabled"] = True

    if overrides:
        settings = replace(settings, **overrides)
    return settings


def _format_duration(seconds: float | None) -> str:
    # Backward-compatible wrapper; formatting lives in listing module.
    if seconds is None:
        return "-"
    minutes, rem = divmod(int(seconds), 60)
    if minutes:
        return f"{minutes}m{rem:02d}s"
    return f"{rem}s"


def _parse_filename(path: Path) -> tuple[str | None, str | None]:
    # Backward-compatible wrapper kept for tests; actual parsing lives in listing module.
    from .listing import _parse_filename as _parse  # local import to avoid cycles

    return _parse(path)


def _list_recordings(settings: Settings, *, limit: int = 10) -> int:
    try:
        items = collect_recordings(settings)
    except Exception:
        return 1

    visible_all = [i for i in items if (i.has_source or i.has_transcript or i.has_archive)]
    total = len(visible_all)
    visible = visible_all
    if limit > 0:
        visible = visible[:limit]

    LOGGER.debug(
        "List summary: total=%d shown=%d limit=%d",
        total,
        len(visible),
        limit,
        extra={"verbosity": 2},
    )

    # Fill in duration for archive-only / inbox-imported files (best-effort).
    # Probe only for the items we are actually going to show to keep --list fast.
    probed = 0
    updated = 0
    store: StateStore | None = None
    for item in visible:
        if item.duration is not None:
            continue
        if not item.has_archive or item.archive_path is None:
            continue
        try:
            seconds = listing_mod.probe_audio_duration_seconds(item.archive_path)
        except Exception as err:
            LOGGER.debug(
                "Duration probe failed for %s: %s",
                item.archive_path.name,
                err,
                extra={"verbosity": 2},
            )
            seconds = None
        if seconds is not None:
            item.duration = seconds
            probed += 1
            # Persist best-effort duration back to our state DB to avoid re-probing next time.
            try:
                if store is None:
                    store = StateStore(settings.state_db)
                updated += store.set_duration_for_archived_path(item.archive_path, seconds)
            except Exception as err:
                LOGGER.debug(
                    "Failed to persist duration for %s: %s",
                    item.archive_path.name,
                    err,
                    extra={"verbosity": 2},
                )

    if probed:
        LOGGER.debug(
            "Filled duration from audio probe for %d item(s) (persisted=%d)",
            probed,
            updated,
            extra={"verbosity": 2},
        )
    if store is not None:
        store.close()

    shown = len(visible)
    output = format_list_output(visible, shown=shown, total=total)
    print(output, end="")
    return 0


def _pending_for_processing(settings: Settings) -> list:
    """The recordings a real run would process: have a source/archive file but
    no transcript yet, ordered the same way the worker scan enqueues them.

    Mirrors ``VoiceMemoService.enqueue_existing`` selection (pending = not yet
    transcribed) + sort (``processing_order``) so ``--dry-run`` previews the
    exact backlog without touching it.
    """
    items = collect_recordings(settings)
    pending = [i for i in items if (i.has_source or i.has_archive) and not i.has_transcript]
    newest_first = settings.processing_order == "newest-first"
    pending.sort(
        key=lambda i: i.created_at.replace(tzinfo=None) if i.created_at else datetime.min,
        reverse=newest_first,
    )
    return pending


def _dry_run_recordings(settings: Settings) -> int:
    """Print what a real run would transcribe, then exit. No lock, no work."""
    try:
        pending = _pending_for_processing(settings)
    except Exception as err:
        LOGGER.error("%s", err)
        return 1

    order = settings.processing_order
    if not pending:
        print("Dry run — no pending recordings to process (all transcribed).")
        return 0

    print(f"Dry run — would process {len(pending)} pending recording(s), order: {order}:")
    print(f"  {'#':>2}  {'When':19}  {'Duration':8}  Title")
    for idx, item in enumerate(pending, 1):
        when = item.created_at.strftime("%Y-%m-%d %H:%M:%S") if item.created_at else "unknown"
        duration_str = listing_mod._format_duration(item.duration)
        title = item.title or item.key
        print(f"  {idx:>2}  {when:19}  {duration_str:<8}  {title}")
    print("(dry run — nothing transcribed; drop --dry-run to process)")
    return 0


def _unset_proxy_env() -> None:
    """Force every backend HTTP request to bypass any system / shell proxy.

    The ASR + diarize HTTP backends are typically reachable directly (LAN /
    VPN / private hostname) and must not be routed through an outbound
    HTTP proxy. A leaked ``http_proxy`` from the user's shell session,
    **or** the macOS system proxy (which ``urllib.getproxies()`` picks up
    via SystemConfiguration even when env vars are clean), causes the
    proxy to CONNECT-tunnel the request and the inner TLS handshake gets
    torn down (LibreSSL ``SSL_ERROR_SYSCALL`` / ``Broken pipe``) — the
    pipeline then silently falls back to the local model and ships a
    ``.txt`` with no ``.md``.

    Two-step fix:
      1. Drop ``*_proxy`` env vars (covers shell-leaked proxy).
      2. Set ``no_proxy=*`` / ``NO_PROXY=*`` so urllib's
         ``getproxies_macosx_sysconf()`` lookup returns no proxy
         regardless of what System Settings → Network → Proxies says.
    Done unconditionally on every invocation so the user doesn't need to
    remember a prefix.
    """
    for var in (
        "http_proxy", "https_proxy", "all_proxy",
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    ):
        os.environ.pop(var, None)
    # urllib + requests both honour `no_proxy=*` as "bypass for everything".
    os.environ["no_proxy"] = "*"
    os.environ["NO_PROXY"] = "*"


def main(argv: list[str] | None = None) -> int:
    _unset_proxy_env()

    # Route the `si` subcommand group to the speaker-id CLI before argparse
    # touches the rest of the args. Keeping it as a pre-dispatch (rather than
    # a subparser) means existing flat flags (`--watch`, `-l`, ...) keep
    # working unchanged — this is the user-facing entry, no migration needed.
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    if raw_argv and raw_argv[0] == "si":
        from .si.cli import main as si_main
        return si_main(raw_argv[1:])

    # Common typo: user types `ls` / `list` expecting a list subcommand.
    # Map to the existing `-l` flag so it shows the listing instead of
    # falling through to the positional `audio` path — that path enters
    # the single-instance lock and looks "stuck" whenever a real
    # transcribe is running in another shell.
    if raw_argv and raw_argv[0] in {"ls", "list"}:
        raw_argv = ["-l"] + raw_argv[1:]

    parser = argparse.ArgumentParser(
        description="Transcribe Apple Voice Memos with WhisperKit (default) or the speaker-id pipeline.",
        epilog=(
            "Subcommands:\n"
            "  si <subcommand>   Speaker-ID pipeline (transcribe → diarize → identify → merge → render).\n"
            "                    Run `voicememo-whisper si --help` or `voicememo-whisper si steps` for details."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v for info, -vv for debug).",
    )
    parser.add_argument(
        "audio",
        nargs="?",
        help="Optional path to a single audio file to process end-to-end. "
             "Bypasses the Voice Memos / Inbox / archive scan and runs only "
             "this file through the same state-DB + archive + speaker-id flow.",
    )
    parser.add_argument("--watch", action="store_true", help="Keep running and watch for new recordings.")
    parser.add_argument(
        "--model", help="WhisperKit model identifier (default from env or 'large-v3-v20240930_turbo')."
    )
    parser.add_argument("--language", help="Language hint for Whisper (e.g. 'en', 'zh').")
    parser.add_argument("-l", "--list", action="store_true", help="List available recordings and exit.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which pending recordings a real run would transcribe (newest-first), then exit. "
             "Nothing is transcribed and the single-instance lock is not taken.",
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=10,
        help="For --list: number of items to show (default: 10; 0 for all).",
    )
    parser.add_argument(
        "--archive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable archiving of processed recordings (default: true). Use --no-archive to disable.",
    )
    parser.add_argument(
        "--archive-dir", 
        help=f"Directory to archive audio files (implies --archive). Defaults to '{DEFAULT_ARCHIVE_PATH}' or VOICE_MEMO_ARCHIVE_DIR env var."
    )
    parser.add_argument(
        "--transcript-dir",
        help=f"Directory to save transcripts. Defaults to '{DEFAULT_TRANSCRIPT_PATH}' or VOICE_MEMO_TRANSCRIPT_DIR env var."
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING...). Default: INFO.",
    )
    parser.add_argument(
        "--newest-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Process backlog newest first (disable for oldest first). Default: true.",
    )

    args = parser.parse_args(raw_argv)
    _configure_logging(args.log_level, args.verbose)

    try:
        settings = build_settings(args)
    except Exception as err:
        LOGGER.error("%s", err)
        return 1

    if args.list:
        return _list_recordings(settings, limit=args.limit)

    if args.dry_run:
        return _dry_run_recordings(settings)

    # Fail-fast on a missing audio file BEFORE acquiring the
    # single-instance lock. A plain typo (`voicememo-whisper foo`) should
    # not appear to "hang" on the lock when another transcribe is
    # running; surface the error + point at `-l` while the user is still
    # at the prompt.
    if args.audio and not Path(args.audio).exists():
        LOGGER.error(
            "audio file not found: %s (did you mean `-l` to list recordings?)",
            args.audio,
        )
        return 1

    # Long-running processing: guard against a second instance racing on
    # the same Voice Memos source / state DB / archive dir.
    from ._lock import single_instance_lock, job_info_for_audio

    # Give a second instance enough to estimate our finish time (so it can tell the
    # user when to retry). Single-file mode knows the audio up front; backlog mode
    # doesn't, so pass nothing there. Deferred (callable) so the probe runs only
    # after we win the lock — a losing instance never pays for it.
    lock_job_info = (lambda: job_info_for_audio(args.audio)) if args.audio else None

    with single_instance_lock(job_info=lock_job_info) as run:
        try:
            from .service import VoiceMemoService
            service = VoiceMemoService(settings)
        except Exception as err:
            LOGGER.error("%s", err)
            run.mark("error", str(err))   # `return` exits the lock normally; record the failure
            return 1

        try:
            if args.audio:
                # Single-file mode: skip the broad scan, just run this one
                # through the same state-DB + ArchiveManager + speaker_pipeline
                # path. This is the canonical way to (re)process one specific
                # file — `si run` is intentionally the diagnostic-only entry.
                service.process_one(Path(args.audio))
                service.join()
            else:
                service.start(watch=args.watch)
                if args.watch:
                    logging.info(
                        "Backlog synced. Watching for new recordings. Press Ctrl+C to exit.",
                        extra={"verbosity": 1},
                    )
                    while True:
                        time.sleep(1)
                else:
                    service.join()
        except KeyboardInterrupt:
            logging.info("Interrupted by user.", extra={"verbosity": 1})
            run.mark("interrupted")   # swallowed here, so the classifier can't see it
        finally:
            service.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
