"""Device selection, metering, and live Focusrite audio streams."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log10, sqrt
from typing import Any, Mapping, Sequence

import numpy as np
import sounddevice as sd

from .amp import Version1Amp


FOCUSRITE_NAME_MARKERS = ("focusrite", "scarlett", "clarett", "vocaster")
CALLBACK_STATUS_NAMES = (
    "input_underflow",
    "input_overflow",
    "output_underflow",
    "output_overflow",
    "priming_output",
)


class DeviceSelectionError(ValueError):
    """Raised when no safe, unambiguous input/output device can be selected."""


def _channel_key(direction: str) -> str:
    if direction not in {"input", "output"}:
        raise ValueError(f"Unknown device direction: {direction}")
    return f"max_{direction}_channels"


def device_supports(device: Mapping[str, Any], direction: str, channels: int = 1) -> bool:
    return int(device.get(_channel_key(direction), 0)) >= channels


def is_focusrite(device: Mapping[str, Any]) -> bool:
    name = str(device.get("name", "")).casefold()
    return any(marker in name for marker in FOCUSRITE_NAME_MARKERS)


def format_device(index: int, device: Mapping[str, Any]) -> str:
    return (
        f"{index:>2}: {device.get('name', '<unnamed>')} "
        f"(inputs: {int(device.get('max_input_channels', 0))}, "
        f"outputs: {int(device.get('max_output_channels', 0))}, "
        f"default rate: {float(device.get('default_samplerate', 0)):g} Hz)"
    )


def list_device_lines(devices: Sequence[Mapping[str, Any]]) -> list[str]:
    if not devices:
        return ["No PortAudio audio devices were found."]
    return [format_device(index, device) for index, device in enumerate(devices)]


def _coerce_device_reference(reference: str | int) -> int | None:
    if isinstance(reference, int):
        return reference
    stripped = reference.strip()
    return int(stripped) if stripped.isdecimal() else None


def _matching_indices(
    devices: Sequence[Mapping[str, Any]],
    reference: str | int,
    direction: str,
    channels: int,
) -> list[int]:
    numeric_reference = _coerce_device_reference(reference)
    if numeric_reference is not None:
        if 0 <= numeric_reference < len(devices) and device_supports(devices[numeric_reference], direction, channels):
            return [numeric_reference]
        return []

    needle = str(reference).casefold().strip()
    return [
        index
        for index, device in enumerate(devices)
        if needle in str(device.get("name", "")).casefold() and device_supports(device, direction, channels)
    ]


def resolve_device(
    devices: Sequence[Mapping[str, Any]],
    reference: str | int | None,
    direction: str,
    *,
    channels: int = 1,
    default_index: int | None = None,
) -> int:
    """Resolve an explicit name/index or one unambiguous Focusrite device."""

    if not devices:
        raise DeviceSelectionError("No audio devices are available. Connect the Focusrite and grant microphone access.")

    if reference is not None:
        matches = _matching_indices(devices, reference, direction, channels)
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise DeviceSelectionError(
                f"No {direction} device matching {reference!r} supports {channels} channel(s). "
                "Run `kraken-dsp devices` to see the available devices."
            )
        choices = "; ".join(format_device(index, devices[index]) for index in matches)
        raise DeviceSelectionError(f"{reference!r} matches multiple {direction} devices: {choices}")

    focusrite_matches = [
        index
        for index, device in enumerate(devices)
        if is_focusrite(device) and device_supports(device, direction, channels)
    ]
    if len(focusrite_matches) == 1:
        return focusrite_matches[0]
    if len(focusrite_matches) > 1:
        if default_index in focusrite_matches:
            return int(default_index)
        choices = "; ".join(format_device(index, devices[index]) for index in focusrite_matches)
        raise DeviceSelectionError(
            f"More than one Focusrite {direction} device is available. Select one with "
            f"--{direction}-device. Choices: {choices}"
        )

    if default_index is not None and 0 <= default_index < len(devices):
        if device_supports(devices[default_index], direction, channels):
            return default_index
    raise DeviceSelectionError(
        f"No Focusrite {direction} device was found and no usable default is set. "
        f"Run `kraken-dsp devices` and pass --{direction}-device NAME_OR_INDEX."
    )


def _default_indices() -> tuple[int | None, int | None]:
    defaults = sd.default.device
    try:
        input_index, output_index = int(defaults[0]), int(defaults[1])
    except (IndexError, TypeError, ValueError):
        return None, None
    return (input_index if input_index >= 0 else None, output_index if output_index >= 0 else None)


@dataclass(frozen=True)
class StreamConfig:
    input_device: str | int | None = None
    output_device: str | int | None = None
    input_channel: int = 1  # Human-facing / one-based channel number.
    output_channels: int | None = None
    sample_rate: int = 48_000
    blocksize: int = 128
    latency: str | float = "low"
    allow_split_devices: bool = False

    def __post_init__(self) -> None:
        if self.input_channel < 1:
            raise ValueError("Input channels are one-based; use 1 for the first Focusrite input")
        if self.output_channels is not None and self.output_channels < 1:
            raise ValueError("Output channel count must be positive")
        if self.sample_rate not in (44_100, 48_000):
            raise ValueError("Choose --sample-rate 44100 or 48000")
        if self.blocksize < 0:
            raise ValueError("Block size must be zero (device default) or positive")


@dataclass
class Meter:
    """Low-cost callback telemetry rendered later by the main thread."""

    input_peak: float = 0.0
    input_rms: float = 0.0
    output_peak: float = 0.0
    output_rms: float = 0.0
    status_flags: set[str] = field(default_factory=set)

    def update(self, input_samples: np.ndarray, output_samples: np.ndarray | None = None) -> None:
        self.input_peak = float(np.max(np.abs(input_samples))) if len(input_samples) else 0.0
        self.input_rms = float(sqrt(np.mean(np.square(input_samples)))) if len(input_samples) else 0.0
        if output_samples is not None:
            self.output_peak = float(np.max(np.abs(output_samples))) if len(output_samples) else 0.0
            self.output_rms = float(sqrt(np.mean(np.square(output_samples)))) if len(output_samples) else 0.0

    def add_status(self, status: sd.CallbackFlags) -> None:
        # CallbackFlags is truthy when any condition is present, but it is not
        # an integer/bitmask in sounddevice. Keep readable names for the main
        # thread to print instead.
        for name in CALLBACK_STATUS_NAMES:
            if getattr(status, name):
                self.status_flags.add(name.replace("_", " "))

    @staticmethod
    def dbfs(value: float) -> float:
        return 20.0 * log10(max(value, 1e-12))

    def format(self, include_output: bool) -> str:
        line = f"input peak {self.dbfs(self.input_peak):6.1f} dBFS | rms {self.dbfs(self.input_rms):6.1f} dBFS"
        if include_output:
            line += f" | output peak {self.dbfs(self.output_peak):6.1f} dBFS"
        if self.status_flags:
            line += " | PortAudio status: " + ", ".join(sorted(self.status_flags))
        return line


def _query_devices() -> list[Mapping[str, Any]]:
    return list(sd.query_devices())


def select_stream_devices(config: StreamConfig, *, capture_only: bool) -> tuple[int, int | None, int]:
    """Resolve device IDs and a sensible mono/stereo output channel count."""

    devices = _query_devices()
    default_input, default_output = _default_indices()
    if capture_only:
        input_device = resolve_device(
            devices,
            config.input_device,
            "input",
            channels=config.input_channel,
            default_index=default_input,
        )
        return input_device, None, 0

    # A duplex stream using independent USB/Core Audio devices has independent
    # clocks and can drift or underrun. Prefer one common Focusrite unless the
    # caller deliberately opts into a split-device setup.
    if config.input_device is None and config.output_device is None:
        common_focusrites = [
            index
            for index, device in enumerate(devices)
            if is_focusrite(device)
            and device_supports(device, "input", config.input_channel)
            and device_supports(device, "output")
        ]
        if len(common_focusrites) == 1:
            input_device = output_device = common_focusrites[0]
        elif len(common_focusrites) > 1 and default_input == default_output and default_input in common_focusrites:
            input_device = output_device = int(default_input)
        elif len(common_focusrites) > 1:
            choices = "; ".join(format_device(index, devices[index]) for index in common_focusrites)
            raise DeviceSelectionError(
                "More than one duplex Focusrite is available. Select one with --input-device "
                "and --output-device. Choices: "
                + choices
            )
        else:
            raise DeviceSelectionError(
                "No single Focusrite device has both the requested input and an output. "
                "Pass both devices and --allow-split-devices only if you intentionally use separate clocks."
            )
    elif config.input_device is not None and config.output_device is None:
        input_device = resolve_device(
            devices,
            config.input_device,
            "input",
            channels=config.input_channel,
            default_index=default_input,
        )
        if not device_supports(devices[input_device], "output"):
            raise DeviceSelectionError(
                "The selected input device has no output. Pass --output-device and --allow-split-devices "
                "only if you intentionally use separate devices."
            )
        output_device = input_device
    elif config.input_device is None and config.output_device is not None:
        output_device = resolve_device(
            devices,
            config.output_device,
            "output",
            channels=1,
            default_index=default_output,
        )
        if not device_supports(devices[output_device], "input", config.input_channel):
            raise DeviceSelectionError(
                "The selected output device cannot provide the requested input channel. Pass --input-device "
                "and --allow-split-devices only if you intentionally use separate devices."
            )
        input_device = output_device
    else:
        input_device = resolve_device(
            devices,
            config.input_device,
            "input",
            channels=config.input_channel,
            default_index=default_input,
        )
        output_device = resolve_device(
            devices,
            config.output_device,
            "output",
            channels=1,
            default_index=default_output,
        )
        if input_device != output_device and not config.allow_split_devices:
            raise DeviceSelectionError(
                "Input and output are different devices. Use one Focusrite for both, or explicitly add "
                "--allow-split-devices if you understand the clocking risk."
            )

    max_output_channels = int(devices[output_device]["max_output_channels"])
    output_channels = config.output_channels or min(2, max_output_channels)
    if output_channels > max_output_channels:
        raise DeviceSelectionError(
            f"Output device {output_device} only has {max_output_channels} channel(s), "
            f"but {output_channels} were requested."
        )
    return input_device, output_device, output_channels


def _validate_stream_settings(
    input_device: int,
    output_device: int | None,
    config: StreamConfig,
    output_channels: int,
) -> None:
    sd.check_input_settings(
        device=input_device,
        channels=config.input_channel,
        samplerate=config.sample_rate,
        dtype="float32",
    )
    if output_device is not None:
        sd.check_output_settings(
            device=output_device,
            channels=output_channels,
            samplerate=config.sample_rate,
            dtype="float32",
        )


class LiveAmp:
    """Run a Version1Amp in a mono-in, duplicated-mono-out callback."""

    def __init__(self, amp: Version1Amp, config: StreamConfig) -> None:
        self.amp = amp
        self.config = config
        self.meter = Meter()
        self._input_channel_index = config.input_channel - 1
        self.callback_error: Exception | None = None

    def callback(self, indata: np.ndarray, outdata: np.ndarray, _frames: int, _time: Any, status: sd.CallbackFlags) -> None:
        outdata.fill(0.0)
        if status:
            self.meter.add_status(status)
        input_samples = indata[:, self._input_channel_index]
        try:
            processed = self.amp.process_block(input_samples)
            outdata[:, :] = processed[:, np.newaxis]
            self.meter.update(input_samples, processed)
        except Exception as error:
            # Prevent a Python DSP failure from emitting stale/undefined audio.
            # Save it for the main thread, then stop the callback immediately.
            self.callback_error = error
            outdata.fill(0.0)
            raise sd.CallbackAbort from error

    def open_stream(self) -> sd.Stream:
        input_device, output_device, output_channels = select_stream_devices(self.config, capture_only=False)
        assert output_device is not None
        _validate_stream_settings(input_device, output_device, self.config, output_channels)
        return sd.Stream(
            device=(input_device, output_device),
            samplerate=self.config.sample_rate,
            blocksize=self.config.blocksize,
            latency=self.config.latency,
            dtype="float32",
            channels=(self.config.input_channel, output_channels),
            callback=self.callback,
        )


class InputCapture:
    """Safely meter a Focusrite input without sending audio anywhere."""

    def __init__(self, config: StreamConfig) -> None:
        self.config = config
        self.meter = Meter()
        self._input_channel_index = config.input_channel - 1

    def callback(self, indata: np.ndarray, _frames: int, _time: Any, status: sd.CallbackFlags) -> None:
        if status:
            self.meter.add_status(status)
        self.meter.update(indata[:, self._input_channel_index])

    def open_stream(self) -> sd.InputStream:
        input_device, _, _ = select_stream_devices(self.config, capture_only=True)
        _validate_stream_settings(input_device, None, self.config, 0)
        return sd.InputStream(
            device=input_device,
            samplerate=self.config.sample_rate,
            blocksize=self.config.blocksize,
            latency=self.config.latency,
            dtype="float32",
            channels=self.config.input_channel,
            callback=self.callback,
        )
