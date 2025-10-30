from __future__ import annotations

import argparse
import logging
import time
from dataclasses import replace

from .config import Settings, load_settings
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Transcribe Apple Voice Memos with WhisperKit.")
    parser.add_argument("--watch", action="store_true", help="Keep running and watch for new recordings.")
    parser.add_argument(
        "--model", help="WhisperKit model identifier (default from env or 'large-v3-v20240930_turbo')."
    )
    parser.add_argument("--language", help="Language hint for Whisper (e.g. 'en', 'zh').")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING...). Default: INFO.",
    )

    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    try:
        settings = build_settings(args)
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
