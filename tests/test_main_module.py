from __future__ import annotations

import runpy
from unittest.mock import patch

import pytest

def test_main_module_exits_with_cli_code() -> None:
    with patch("voicememowhisper.cli.main", lambda _argv=None: 7):
        with pytest.raises(SystemExit) as ctx:
            runpy.run_module("voicememowhisper.__main__", run_name="__main__")
    assert ctx.value.code == 7

