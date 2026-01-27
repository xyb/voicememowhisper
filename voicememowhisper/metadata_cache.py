from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Callable

from .config import Settings
from .metadata import VoiceMemo, load_voice_memos

LOGGER = logging.getLogger("metadata_cache")


class MetadataCache:
    """Cache voice memo metadata and provide memo lookup by path."""

    def __init__(self, settings: Settings, *, loader: Callable[[Settings], dict[str, VoiceMemo]] = load_voice_memos) -> None:
        self._settings = settings
        self._loader = loader
        self._memos: dict[str, VoiceMemo] = {}

    def refresh(self) -> None:
        try:
            self._memos = self._loader(self._settings)
        except PermissionError as err:
            LOGGER.warning("Metadata access denied: %s", err)
            self._memos = {}

    def get_memo(self, path: Path) -> VoiceMemo:
        guid = path.stem
        memo = self._memos.get(guid)
        if memo and memo.title:
            if memo.path != path:
                memo = replace(memo, path=path)
                self._memos[guid] = memo
            return memo

        # Lazy refresh when we don't have a good memo cached.
        self.refresh()
        memo = self._memos.get(guid)
        if memo:
            if memo.path != path:
                memo = replace(memo, path=path)
                self._memos[guid] = memo
            return memo

        memo = VoiceMemo(guid=guid, path=path)
        self._memos[guid] = memo
        return memo

    @staticmethod
    def display_name(memo: VoiceMemo) -> str:
        if memo.title:
            title = memo.title.strip()
            if title:
                return title
        stem = memo.path.stem
        return stem or memo.guid

