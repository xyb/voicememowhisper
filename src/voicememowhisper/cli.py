from __future__ import annotations

import argparse
import logging
import time
from dataclasses import replace

from .config import Settings, load_settings
from .metadata import list_voice_memos, resolve_created_at
from .service import VoiceMemoService


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )


def build_settings(args: argparse.Namespace) -> Settings:
    settings = load_settings()
    if args.model or args.language:
        settings = replace(
            settings,
            whisperkit_model=args.model or settings.whisperkit_model,
            language=args.language or settings.language,
        )
    return settings


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    minutes, rem = divmod(int(seconds), 60)
    if minutes:
        return f"{minutes}m{rem:02d}s"
    return f"{rem}s"


def _list_recordings(settings: Settings) -> int:
    try:
        memos = list_voice_memos(settings)
    except Exception as err:
        logging.getLogger(__name__).error("Failed to list recordings: %s", err)
        return 1

    if not memos:
        logging.info("No recordings found in %s", settings.recordings_dir)
        return 0

    print("When                 | Duration | Title                           | GUID                                 | File")
    print("-" * 140)
    for memo in memos:
        created = resolve_created_at(memo)
        when = created.strftime("%Y-%m-%d %H:%M:%S") if created else "unknown"
        title = (memo.title or "").strip() or memo.guid
        duration = _format_duration(memo.duration_seconds)
        path = memo.path
        suffix = "" if path.exists() else " (missing file)"
        print(f"{when:19} | {duration:8} | {title[:30]:30} | {memo.guid:36} | {path}{suffix}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Transcribe Apple Voice Memos with WhisperKit.")
    parser.add_argument("--watch", action="store_true", help="Keep running and watch for new recordings.")
    parser.add_argument(
        "--model", help="WhisperKit model identifier (default from env or 'large-v3-v20240930_turbo')."
    )
    parser.add_argument("--language", help="Language hint for Whisper (e.g. 'en', 'zh').")
    parser.add_argument("--list", action="store_true", help="List available recordings and exit.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING...). Default: INFO.",
    )

    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    try:
        settings = build_settings(args)
    except Exception as err:
        logging.getLogger(__name__).error("%s", err)
        return 1

    if args.list:
        return _list_recordings(settings)

    try:
        service = VoiceMemoService(settings)
    except Exception as err:
        logging.getLogger(__name__).error("%s", err)
        return 1

    try:
        service.start(watch=args.watch)
        if args.watch:
            logging.info("Backlog synced. Watching for new recordings. Press Ctrl+C to exit.")
            while True:
                time.sleep(1)
        else:
            service.join()
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    finally:
        service.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
