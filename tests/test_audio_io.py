from __future__ import annotations

import pytest

import numpy as np
import sounddevice as sd

from kraken_dsp.amp import AmpSettings, Version1Amp
from kraken_dsp import audio_io
from kraken_dsp.audio_io import (
    DeviceSelectionError,
    LiveAmp,
    StreamConfig,
    list_device_lines,
    resolve_device,
    select_stream_devices,
)


DEVICES = [
    {"name": "Built-in Microphone", "max_input_channels": 2, "max_output_channels": 0, "default_samplerate": 48_000},
    {"name": "Focusrite Scarlett 2i2 USB", "max_input_channels": 2, "max_output_channels": 2, "default_samplerate": 48_000},
    {"name": "Built-in Output", "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 48_000},
]


def test_auto_selects_unambiguous_focusrite_for_input_and_output() -> None:
    assert resolve_device(DEVICES, None, "input", default_index=0) == 1
    assert resolve_device(DEVICES, None, "output", default_index=2) == 1


def test_device_name_and_index_selection_validate_channel_count() -> None:
    assert resolve_device(DEVICES, "scarlett", "input", channels=2) == 1
    assert resolve_device(DEVICES, 1, "output", channels=2) == 1
    with pytest.raises(DeviceSelectionError, match="supports 3 channel"):
        resolve_device(DEVICES, "scarlett", "input", channels=3)


def test_exact_device_name_wins_over_a_substring_match() -> None:
    pipewire_devices = [
        {"name": "sysdefault", "max_input_channels": 128, "max_output_channels": 128, "default_samplerate": 48_000},
        {"name": "default", "max_input_channels": 64, "max_output_channels": 64, "default_samplerate": 44_100},
    ]

    assert resolve_device(pipewire_devices, "default", "input") == 1
    assert resolve_device(pipewire_devices, "default", "output") == 1


def test_missing_focusrite_without_default_is_actionable() -> None:
    devices = [DEVICES[0], DEVICES[2]]
    with pytest.raises(DeviceSelectionError, match="kraken-dsp devices"):
        resolve_device(devices, None, "input")


def test_device_list_displays_channels() -> None:
    lines = list_device_lines(DEVICES)
    assert "Focusrite Scarlett 2i2 USB" in lines[1]
    assert "inputs: 2" in lines[1]


def test_live_callback_uses_requested_one_based_input_and_duplicates_output() -> None:
    amp = Version1Amp(48_000, settings=AmpSettings(cabinet_bypass=True))
    live_amp = LiveAmp(amp, StreamConfig(input_channel=2))
    indata = np.column_stack((np.zeros(128, dtype=np.float32), np.full(128, 0.05, dtype=np.float32)))
    outdata = np.empty((128, 2), dtype=np.float32)

    live_amp.callback(indata, outdata, 128, None, sd.CallbackFlags())

    assert np.allclose(outdata[:, 0], outdata[:, 1])
    assert np.max(np.abs(outdata)) > 0.0
    assert live_amp.meter.input_peak == pytest.approx(0.05)


def test_live_callback_mutes_and_records_a_dsp_error() -> None:
    class BrokenAmp:
        def process_block(self, _input_samples):
            raise RuntimeError("simulated DSP failure")

    live_amp = LiveAmp(BrokenAmp(), StreamConfig())
    indata = np.full((128, 1), 0.05, dtype=np.float32)
    outdata = np.full((128, 2), 1.0, dtype=np.float32)

    with pytest.raises(sd.CallbackAbort):
        live_amp.callback(indata, outdata, 128, None, sd.CallbackFlags())

    assert isinstance(live_amp.callback_error, RuntimeError)
    assert np.all(outdata == 0.0)


def test_live_callback_crossfades_a_requested_processor_swap() -> None:
    class ConstantAmp:
        def __init__(self, value: float) -> None:
            self.value = value

        def process_block(self, input_samples):
            return np.full(len(input_samples), self.value, dtype=np.float32)

    live_amp = LiveAmp(ConstantAmp(0.0), StreamConfig(sample_rate=48_000))
    indata = np.zeros((128, 1), dtype=np.float32)
    outdata = np.empty((128, 2), dtype=np.float32)
    live_amp.request_processor(ConstantAmp(1.0))

    live_amp.callback(indata, outdata, 128, None, sd.CallbackFlags())
    assert np.all(outdata >= 0.0)
    assert np.all(outdata < 1.0)

    for _ in range(12):
        live_amp.callback(indata, outdata, 128, None, sd.CallbackFlags())
    assert np.allclose(outdata, 1.0)


def test_auto_duplex_selection_never_mixes_two_focusrites(monkeypatch) -> None:
    two_focusrites = [
        {"name": "Focusrite Scarlett A", "max_input_channels": 2, "max_output_channels": 2, "default_samplerate": 48_000},
        {"name": "Focusrite Scarlett B", "max_input_channels": 2, "max_output_channels": 2, "default_samplerate": 48_000},
    ]
    monkeypatch.setattr(audio_io, "_query_devices", lambda: two_focusrites)
    monkeypatch.setattr(audio_io, "_default_indices", lambda: (0, 1))

    with pytest.raises(DeviceSelectionError, match="More than one duplex Focusrite"):
        select_stream_devices(StreamConfig(), capture_only=False)


def test_one_explicit_duplex_device_is_used_for_both_directions(monkeypatch) -> None:
    monkeypatch.setattr(audio_io, "_query_devices", lambda: DEVICES)
    monkeypatch.setattr(audio_io, "_default_indices", lambda: (0, 2))

    input_device, output_device, output_channels = select_stream_devices(
        StreamConfig(input_device="scarlett"), capture_only=False
    )

    assert (input_device, output_device, output_channels) == (1, 1, 2)


def test_one_explicit_output_device_is_used_for_both_directions(monkeypatch) -> None:
    monkeypatch.setattr(audio_io, "_query_devices", lambda: DEVICES)
    monkeypatch.setattr(audio_io, "_default_indices", lambda: (0, 2))

    input_device, output_device, output_channels = select_stream_devices(
        StreamConfig(output_device="scarlett"), capture_only=False
    )

    assert (input_device, output_device, output_channels) == (1, 1, 2)


def test_explicit_split_devices_need_an_opt_in(monkeypatch) -> None:
    split_devices = [
        {"name": "Focusrite Input", "max_input_channels": 2, "max_output_channels": 0, "default_samplerate": 48_000},
        {"name": "Focusrite Output", "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 48_000},
    ]
    monkeypatch.setattr(audio_io, "_query_devices", lambda: split_devices)
    monkeypatch.setattr(audio_io, "_default_indices", lambda: (0, 1))

    with pytest.raises(DeviceSelectionError, match="allow-split-devices"):
        select_stream_devices(StreamConfig(input_device=0, output_device=1), capture_only=False)
    assert select_stream_devices(
        StreamConfig(input_device=0, output_device=1, allow_split_devices=True), capture_only=False
    ) == (0, 1, 2)
