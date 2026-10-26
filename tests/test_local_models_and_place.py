"""Parsers and --place decisions. No model is loaded."""

from __future__ import annotations

import argparse

from voicememowhisper.si.asr_backends.moss import parse_moss_output
from voicememowhisper.si.asr_backends.whisper_cpp import parse_whisper_json
from voicememowhisper.si.asr_config import build_pipeline_kwargs, load_asr_config
from voicememowhisper.si.cli import _build_asr_backend_kwargs
from voicememowhisper.si.place import (
    MachineSnapshot,
    apply_place,
    local_diarize_blockers,
    parse_pmset,
)


def test_parse_whisper_json_uses_offsets() -> None:
    segments, duration, language = parse_whisper_json(
        {
            "result": {"language": "zh"},
            "transcription": [
                {
                    "offsets": {"from": 530, "to": 1610},
                    "text": " 重复一下 ",
                }
            ],
        }
    )
    assert language == "zh"
    assert duration == 1.61
    assert segments[0].start == 0.53
    assert segments[0].text == "重复一下"


def test_parse_moss_output_keeps_speaker_ids() -> None:
    text = """
[info] timings: load=186.40 ms
  duration:   5.000 s
  [   0.53 ->    1.61] S1: 重复一下了可能。
  [   1.68 ->    2.12] S2: 嗯。
"""
    segments, speakers, duration = parse_moss_output(text)
    assert duration == 5.0
    assert [s.text for s in segments] == ["重复一下了可能。", "嗯。"]
    assert [s.label for s in speakers] == ["S1", "S2"]


def test_whisper_cpp_toml_section(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        "\n".join(
            [
                "[asr]",
                'backend = "whisper-cpp"',
                "[asr.whisper_cpp]",
                'model_path = "/models/ggml.bin"',
                "threads = 4",
                "",
            ]
        ),
        encoding="utf-8",
    )
    kwargs = build_pipeline_kwargs(load_asr_config(path))
    assert kwargs["asr_backend"] == "whisper-cpp"
    assert kwargs["asr_backend_config"]["model_path"] == "/models/ggml.bin"
    assert kwargs["asr_backend_config"]["threads"] == 4


def test_moss_cli_selects_its_own_diarization(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("VMW_CONFIG", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    args = argparse.Namespace(
        asr_backend="moss",
        asr_config=None,
        asr_url=None,
        asr_model=None,
        asr_api_key=None,
        asr_host_header=None,
        asr_response_format=None,
        asr_timeout_sec=None,
        asr_ws_url=None,
        asr_ws_idle_timeout_sec=None,
        asr_ws_connect_timeout_sec=None,
        asr_ws_chunk_send_interval_sec=None,
        diarize_backend=None,
        place=None,
    )
    kwargs = _build_asr_backend_kwargs(args)
    assert kwargs["asr_backend"] == "moss"
    assert kwargs["diarize_backend"] == "moss"


def _ready() -> MachineSnapshot:
    return MachineSnapshot(
        power_known=True,
        on_ac=True,
        battery_percent=100,
        memory_free_gb=12.0,
        load_1min=0.4,
        cpu_count=10,
        compute_lock_held=False,
    )


def test_parse_pmset_ac_and_battery() -> None:
    known, on_ac, percent = parse_pmset(
        "Now drawing from 'AC Power'\n -InternalBattery-0 86%; charging;\n"
    )
    assert (known, on_ac, percent) == (True, True, 86)
    known, on_ac, percent = parse_pmset(
        "Now drawing from 'Battery Power'\n -InternalBattery-0 40%; discharging;\n"
    )
    assert (known, on_ac, percent) == (True, False, 40)


def test_auto_blocks_battery_and_low_charge() -> None:
    on_battery = _ready()
    on_battery = MachineSnapshot(
        power_known=True,
        on_ac=False,
        battery_percent=100,
        memory_free_gb=12,
        load_1min=0.2,
        cpu_count=10,
        compute_lock_held=False,
    )
    assert any("battery" in b for b in local_diarize_blockers(on_battery))
    charging_low = MachineSnapshot(
        power_known=True,
        on_ac=True,
        battery_percent=40,
        memory_free_gb=12,
        load_1min=0.2,
        cpu_count=10,
        compute_lock_held=False,
    )
    assert any("40%" in b for b in local_diarize_blockers(charging_low))
    merged, reason = apply_place({"asr_backend": "openai-audio"}, "auto", charging_low)
    assert merged["asr_backend"] == "openai-audio"
    assert "diarize_backend" not in merged
    assert "keep configured diarize" in reason


def test_auto_ready_moves_only_diarize() -> None:
    merged, reason = apply_place(
        {"asr_backend": "openai-audio", "diarize_backend": "http"},
        "auto",
        _ready(),
    )
    assert merged["asr_backend"] == "openai-audio"
    assert merged["diarize_backend"] == "local_pyannote"
    assert "diarize on this Mac" in reason


def test_place_local_selects_whisper_cpp() -> None:
    merged, reason = apply_place({"asr_backend": "openai-audio"}, "local", None)
    assert merged["asr_backend"] == "whisper-cpp"
    assert merged["diarize_backend"] == "local_pyannote"
    assert "whisper.cpp" in reason
