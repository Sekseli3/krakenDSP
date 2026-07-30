"""Command-line interface for checking a Focusrite input and running the amp."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

import sounddevice as sd

from .amp import AdvancedAmpSettings, Version2Amp, load_cabinet_ir
from .audio_io import DeviceSelectionError, InputCapture, LiveAmp, StreamConfig, list_device_lines


def _add_stream_options(parser: argparse.ArgumentParser, *, include_output: bool) -> None:
    parser.add_argument(
        "--input-device",
        help="Focusrite input device name fragment or index. Defaults to an unambiguous Focusrite device.",
    )
    if include_output:
        parser.add_argument(
            "--output-device",
            help="Output device name fragment or index. Defaults to an unambiguous Focusrite device.",
        )
        parser.add_argument(
            "--output-channels",
            type=int,
            help="Number of output channels to receive the processed mono signal (defaults to stereo when possible).",
        )
        parser.add_argument(
            "--allow-split-devices",
            action="store_true",
            help="Allow different input/output devices (can cause clocking dropouts; normally avoid this).",
        )
    parser.add_argument(
        "--input-channel",
        type=int,
        default=1,
        help="One-based Focusrite input channel carrying the guitar (default: 1).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        choices=(44_100, 48_000),
        default=48_000,
        help="Interface sample rate (default: 48000).",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=128,
        help="PortAudio frames per callback; 128 is a good starting point (default: 128).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kraken-dsp",
        description="Live Python prototype of the Kraken-inspired guitar amp.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("devices", help="List PortAudio devices and their channel counts.")

    capture = subparsers.add_parser("capture", help="Meter a Focusrite input without playback; use this first.")
    _add_stream_options(capture, include_output=False)

    run = subparsers.add_parser("run", help="Run the three-stage high-gain amp on a live guitar input.")
    _add_stream_options(run, include_output=True)
    run.add_argument("--channel", choices=("i", "ii"), default="ii", help="Preamp voice: i is looser; ii is tighter high gain (default: ii).")
    run.add_argument("--gain", type=float, default=6.5, help="Preamp gain from 0 to 10 (default: 6.5).")
    run.add_argument("--input-gain-db", type=float, default=24.0, help="Digital input trim in dB (default: 24).")
    run.add_argument("--output-gain-db", type=float, default=-18.0, help="Final output gain in dB (default: -18).")
    run.add_argument("--cabinet-ir", type=Path, help="Path to a mono or stereo WAV cabinet impulse response.")
    run.add_argument("--cabinet-bypass", action="store_true", help="Bypass the cabinet filter (normally harsh; for debugging only).")
    return parser


def _config_from_args(args: argparse.Namespace, *, include_output: bool) -> StreamConfig:
    return StreamConfig(
        input_device=args.input_device,
        output_device=getattr(args, "output_device", None) if include_output else None,
        input_channel=args.input_channel,
        output_channels=getattr(args, "output_channels", None) if include_output else None,
        sample_rate=args.sample_rate,
        blocksize=args.blocksize,
        allow_split_devices=getattr(args, "allow_split_devices", False) if include_output else False,
    )


def _run_with_meter(stream_owner: InputCapture | LiveAmp, stream: sd.Stream, *, include_output: bool) -> None:
    print(
        f"Streaming at {stream.samplerate:g} Hz, blocksize {stream.blocksize}, latency {stream.latency}. "
        "Press Ctrl-C to stop."
    )
    with stream:
        while stream.active:
            print("\r" + stream_owner.meter.format(include_output), end="", flush=True)
            time.sleep(0.25)
    if isinstance(stream_owner, LiveAmp) and stream_owner.callback_error is not None:
        raise RuntimeError("The audio callback stopped after a DSP processing error") from stream_owner.callback_error


def _run_capture(args: argparse.Namespace) -> None:
    capture = InputCapture(_config_from_args(args, include_output=False))
    _run_with_meter(capture, capture.open_stream(), include_output=False)


def _run_amp(args: argparse.Namespace) -> None:
    config = _config_from_args(args, include_output=True)
    settings = AdvancedAmpSettings(
        channel=args.channel,
        gain=args.gain,
        input_gain_db=args.input_gain_db,
        output_gain_db=args.output_gain_db,
        cabinet_bypass=args.cabinet_bypass,
    )
    cabinet_ir = load_cabinet_ir(args.cabinet_ir, config.sample_rate) if args.cabinet_ir else None
    amp = Version2Amp(config.sample_rate, settings=settings, cabinet_ir=cabinet_ir)
    live_amp = LiveAmp(amp, config)
    _run_with_meter(live_amp, live_amp.open_stream(), include_output=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "devices":
            for line in list_device_lines(list(sd.query_devices())):
                print(line)
        elif args.command == "capture":
            _run_capture(args)
        elif args.command == "run":
            _run_amp(args)
        else:  # Defensive: argparse already makes this unreachable.
            parser.error(f"Unknown command: {args.command}")
    except KeyboardInterrupt:
        print("\nStopped.")
    except (DeviceSelectionError, ValueError, FileNotFoundError, RuntimeError, sd.PortAudioError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
