"""Render a sound-bearing visual walkthrough of the amp DSP signal chain."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import gcd
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from scipy import signal
from scipy.io import wavfile

from .amp import AMP_STAGE_TAP_NAMES, AdvancedAmpSettings, Version2Amp, load_cabinet_ir


@dataclass(frozen=True)
class _StageDescription:
    short_label: str
    detail: str


_STAGE_DESCRIPTIONS: dict[str, _StageDescription] = {
    "Input": _StageDescription("DI input", "The clean guitar DI, before any amp processing."),
    "Input conditioning": _StageDescription(
        "30 Hz high-pass + input trim", "Removes rumble, then sets how hard the guitar hits the preamp."
    ),
    "Preamp stage 1": _StageDescription(
        "Stage 1: pre-emphasis + soft asymmetric clipping", "Adds the first harmonics while retaining low-mid body."
    ),
    "Preamp stage 2": _StageDescription(
        "Stage 2: bass tightening + stronger clipping", "The main high-gain stage; low bass is reduced before distortion."
    ),
    "Preamp stage 3": _StageDescription(
        "Stage 3: saturation + high-frequency smoothing", "Adds compression and smooths harsh upper harmonics."
    ),
    "Tone stack": _StageDescription(
        "Bass / Middle / Treble", "Three EQ filters shape the distorted preamp signal."
    ),
    "Power amp": _StageDescription(
        "Master + sag + power amp + presence", "Master level drives power saturation; sag and Presence/Bass Focus finish the voice."
    ),
    "Cabinet + output": _StageDescription(
        "Cabinet FIR + output limiter", "The cabinet filter supplies speaker-like roll-off; the limiter makes a safe final output."
    ),
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an MP4 that plays and visualises every Kraken DSP stage.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional clean mono/stereo guitar DI WAV. A synthetic plucked-guitar example is used when omitted.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/kraken_dsp_stage_walkthrough.mp4"),
        help="Destination MP4 path (default: artifacts/kraken_dsp_stage_walkthrough.mp4).",
    )
    parser.add_argument("--sample-rate", type=int, choices=(44_100, 48_000), default=48_000)
    parser.add_argument(
        "--seconds-per-stage",
        type=float,
        help="Optional excerpt length for every stage. By default, each stage plays the whole input WAV.",
    )
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--blocksize",
        type=int,
        default=128,
        help="DSP block size. Match the live command's blocksize for an exact sag response (default: 128).",
    )
    parser.add_argument("--channel", choices=("i", "ii"), default="ii")
    parser.add_argument("--gain", type=float, default=7.5)
    parser.add_argument("--input-gain-db", type=float, default=24.0)
    parser.add_argument("--bass", type=float, default=4.0)
    parser.add_argument("--middle", type=float, default=6.0)
    parser.add_argument("--treble", type=float, default=5.0)
    parser.add_argument("--master", type=float, default=6.0)
    parser.add_argument("--presence", type=float, default=5.0)
    parser.add_argument("--presence-bright", action="store_true", help="Enable the brighter presence voicing.")
    parser.add_argument("--sag", type=float, default=2.5)
    parser.add_argument("--bass-focus", choices=("tight", "loose"), default="tight")
    parser.add_argument("--output-gain-db", type=float, default=-6.0)
    parser.add_argument("--cabinet-ir", type=Path, help="Path to a mono or stereo WAV cabinet impulse response.")
    parser.add_argument("--cabinet-bypass", action="store_true", help="Bypass the cabinet FIR (normally only useful for debugging).")
    return parser.parse_args(argv)


def _decode_wav(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, raw = wavfile.read(path)
    samples = np.asarray(raw)
    if samples.ndim == 2:
        samples = np.mean(samples, axis=1)
    if samples.ndim != 1 or not len(samples):
        raise ValueError("Input WAV must contain at least one mono or stereo audio sample")

    if np.issubdtype(samples.dtype, np.integer):
        scale = float(max(abs(np.iinfo(samples.dtype).min), np.iinfo(samples.dtype).max))
        samples = samples.astype(np.float64) / scale
    else:
        samples = samples.astype(np.float64)
    if not np.all(np.isfinite(samples)):
        raise ValueError("Input WAV contains non-finite samples")
    return int(sample_rate), samples


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples
    divisor = gcd(source_rate, target_rate)
    return signal.resample_poly(samples, target_rate // divisor, source_rate // divisor)


def _synthetic_guitar(sample_rate: int) -> np.ndarray:
    """Create a short, deliberately simple plucked-string DI-style example."""

    notes = (82.41, 110.0, 146.83, 196.0, 110.0, 82.41)
    note_seconds = 0.46
    samples_per_note = int(sample_rate * note_seconds)
    output = np.zeros(samples_per_note * len(notes), dtype=np.float64)
    rng = np.random.default_rng(7)

    for index, frequency in enumerate(notes):
        time = np.arange(samples_per_note) / sample_rate
        envelope = (1.0 - np.exp(-time * 1_600.0)) * np.exp(-time * 4.1)
        string = sum(
            (1.0 / harmonic**1.3) * np.sin(2.0 * np.pi * frequency * harmonic * time)
            for harmonic in range(1, 8)
        )
        pick_noise = rng.normal(0.0, 1.0, samples_per_note) * np.exp(-time * 180.0)
        output[index * samples_per_note : (index + 1) * samples_per_note] = 0.075 * envelope * (string + 0.13 * pick_noise)
    return output


def _process_stage_taps(
    samples: np.ndarray,
    sample_rate: int,
    settings: AdvancedAmpSettings,
    cabinet_ir: np.ndarray | None = None,
    *,
    blocksize: int = 128,
) -> dict[str, np.ndarray]:
    amp = Version2Amp(sample_rate, settings=settings, cabinet_ir=cabinet_ir)
    collected = {name: [] for name in AMP_STAGE_TAP_NAMES}
    for start in range(0, len(samples), blocksize):
        _, taps = amp.process_block_with_taps(samples[start : start + blocksize])
        for name in AMP_STAGE_TAP_NAMES:
            collected[name].append(taps[name])
    return {name: np.concatenate(blocks) for name, blocks in collected.items()}


def _stage_audio_segment(samples: np.ndarray, length: int) -> np.ndarray:
    """Return an unmodified stage excerpt for faithful live-signal playback.

    In particular, do not peak-normalise a stage: changing its playback level
    makes pickup noise and gain changes misleading. The final segment is the
    same floating-point output as the live amp for the same input/settings.
    """

    return samples[:length].astype(np.float32, copy=True)


def _level_db(samples: np.ndarray) -> float:
    return 20.0 * np.log10(max(float(np.sqrt(np.mean(np.square(samples)))), 1e-8))


def _render_stage_frame(figure, grid, *, stage_index: int, samples: np.ndarray, sample_rate: int, source_note: str) -> None:
    import matplotlib.patches as patches

    figure.clear()
    flow_axis = figure.add_subplot(grid[0, :])
    wave_axis = figure.add_subplot(grid[1, 0])
    spectrum_axis = figure.add_subplot(grid[1, 1])
    stage_name = AMP_STAGE_TAP_NAMES[stage_index]
    description = _STAGE_DESCRIPTIONS[stage_name]
    labels = ("DI", "Input", "Stage 1", "Stage 2", "Stage 3", "EQ", "Power", "Cab")

    flow_axis.set_xlim(-0.3, len(labels) - 0.1)
    flow_axis.set_ylim(-0.45, 1.15)
    flow_axis.axis("off")
    for index, label in enumerate(labels):
        if index < stage_index:
            color = "#166534"
            edge = "#4ade80"
        elif index == stage_index:
            color = "#155e75"
            edge = "#22d3ee"
        else:
            color = "#273244"
            edge = "#64748b"
        box = patches.FancyBboxPatch(
            (index, 0.15), 0.78, 0.5, boxstyle="round,pad=0.03,rounding_size=0.06", facecolor=color, edgecolor=edge, linewidth=2
        )
        flow_axis.add_patch(box)
        flow_axis.text(index + 0.39, 0.4, label, ha="center", va="center", color="white", fontsize=10, weight="bold")
        if index < len(labels) - 1:
            flow_axis.annotate("", xy=(index + 0.96, 0.4), xytext=(index + 0.8, 0.4), arrowprops={"arrowstyle": "->", "color": "#94a3b8", "lw": 1.8})
    flow_axis.scatter([stage_index + 0.39], [0.88], s=110, color="#22d3ee", zorder=3)
    flow_axis.text(0, 1.05, "KRAKEN DSP — signal travelling through the amp", color="#e2e8f0", fontsize=15, weight="bold")

    window = min(len(samples), max(int(sample_rate * 0.035), 1_024))
    waveform = samples[:window]
    time_ms = np.arange(window) * 1_000.0 / sample_rate
    wave_axis.plot(time_ms, waveform, color="#22d3ee", linewidth=1.15)
    wave_axis.fill_between(time_ms, waveform, 0.0, color="#0e7490", alpha=0.22)
    wave_axis.axhline(0.0, color="#64748b", linewidth=0.8)
    wave_axis.set_title("Waveform", color="#e2e8f0", loc="left", weight="bold")
    wave_axis.set_xlabel("milliseconds", color="#cbd5e1")
    wave_axis.set_ylabel("level", color="#cbd5e1")
    wave_axis.tick_params(colors="#cbd5e1")
    wave_axis.grid(color="#334155", alpha=0.55)

    spectrum_window = min(len(samples), 4_096)
    spectrum_samples = samples[:spectrum_window] * np.hanning(spectrum_window)
    frequencies = np.fft.rfftfreq(spectrum_window, 1.0 / sample_rate)
    magnitude = 20.0 * np.log10(np.maximum(np.abs(np.fft.rfft(spectrum_samples)), 1e-8))
    spectrum_axis.semilogx(frequencies[1:], magnitude[1:], color="#a78bfa", linewidth=1.15)
    spectrum_axis.set_xlim(40, min(16_000, sample_rate / 2.0))
    spectrum_axis.set_ylim(-105, 5)
    spectrum_axis.set_title("Spectrum", color="#e2e8f0", loc="left", weight="bold")
    spectrum_axis.set_xlabel("Hz", color="#cbd5e1")
    spectrum_axis.set_ylabel("dB", color="#cbd5e1")
    spectrum_axis.tick_params(colors="#cbd5e1")
    spectrum_axis.grid(color="#334155", alpha=0.55, which="both")

    for axis in (wave_axis, spectrum_axis):
        axis.set_facecolor("#111827")
        for spine in axis.spines.values():
            spine.set_color("#475569")
    figure.text(0.07, 0.06, f"Stage {stage_index + 1}/{len(AMP_STAGE_TAP_NAMES)}  ·  {description.short_label}", color="white", fontsize=15, weight="bold")
    figure.text(0.07, 0.025, f"{description.detail}  |  RMS {_level_db(samples):.1f} dBFS  |  {source_note}", color="#cbd5e1", fontsize=9.5)


def _render_video(
    output_path: Path,
    stage_segments: dict[str, np.ndarray],
    sample_rate: int,
    seconds_per_stage: float,
    fps: int,
    source_note: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FFMpegWriter
    except ImportError as error:  # pragma: no cover - dependency error path
        raise RuntimeError("The walkthrough renderer needs matplotlib. Run: uv sync") from error

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to create an MP4. On Ubuntu: sudo apt install ffmpeg")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames_per_stage = max(1, round(seconds_per_stage * fps))
    audio = np.concatenate([stage_segments[name] for name in AMP_STAGE_TAP_NAMES])

    with tempfile.TemporaryDirectory(prefix="kraken-dsp-walkthrough-") as temporary_directory:
        temporary_path = Path(temporary_directory)
        silent_video = temporary_path / "video.mp4"
        audio_path = temporary_path / "audio.wav"
        wavfile.write(audio_path, sample_rate, audio)

        figure = plt.figure(figsize=(12.8, 7.2), facecolor="#0f172a")
        grid = figure.add_gridspec(2, 2, height_ratios=(0.8, 1.7), hspace=0.5, wspace=0.25)
        writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=1_800, extra_args=["-pix_fmt", "yuv420p"])
        with writer.saving(figure, str(silent_video), dpi=100):
            for index, name in enumerate(AMP_STAGE_TAP_NAMES):
                _render_stage_frame(
                    figure,
                    grid,
                    stage_index=index,
                    samples=stage_segments[name],
                    sample_rate=sample_rate,
                    source_note=source_note,
                )
                for _ in range(frames_per_stage):
                    writer.grab_frame(facecolor=figure.get_facecolor())
        plt.close(figure)

        result = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(silent_video),
                "-i",
                str(audio_path),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                "-shortest",
                str(output_path),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode:
            raise RuntimeError(f"ffmpeg could not add the audio track:\n{result.stderr[-1_500:]}")


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``kraken-dsp-walkthrough``."""

    args = _parse_args(argv)
    if args.seconds_per_stage is not None and args.seconds_per_stage <= 0:
        raise SystemExit("--seconds-per-stage must be positive")
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if args.blocksize <= 0:
        raise SystemExit("--blocksize must be positive")

    try:
        if args.input is None:
            source = _synthetic_guitar(args.sample_rate)
            source_note = "synthetic plucked-guitar fallback"
        else:
            source_rate, source = _decode_wav(args.input)
            source = _resample(source, source_rate, args.sample_rate)
            source_note = f"clean DI: {args.input.name}"
        source_seconds = len(source) / args.sample_rate
        requested_seconds = args.seconds_per_stage if args.seconds_per_stage is not None else source_seconds
        frames_per_stage = max(1, round(requested_seconds * args.fps))
        segment_length = round(args.sample_rate * frames_per_stage / args.fps)
        if len(source) < segment_length:
            raise ValueError(
                f"Input audio is only {len(source) / args.sample_rate:.2f} seconds; "
                f"record at least {segment_length / args.sample_rate:.2f} seconds or lower --seconds-per-stage"
            )
        source = source[:segment_length]
        cabinet_ir = load_cabinet_ir(args.cabinet_ir, args.sample_rate) if args.cabinet_ir else None
        settings = AdvancedAmpSettings(
            channel=args.channel,
            gain=args.gain,
            input_gain_db=args.input_gain_db,
            bass=args.bass,
            middle=args.middle,
            treble=args.treble,
            master=args.master,
            presence=args.presence,
            presence_bright=args.presence_bright,
            sag=args.sag,
            bass_focus=args.bass_focus,
            output_gain_db=args.output_gain_db,
            cabinet_bypass=args.cabinet_bypass,
        )
        all_taps = _process_stage_taps(source, args.sample_rate, settings, cabinet_ir, blocksize=args.blocksize)
        stage_segments = {
            name: _stage_audio_segment(all_taps[name], segment_length)
            for name in AMP_STAGE_TAP_NAMES
        }
        _render_video(
            args.output,
            stage_segments,
            args.sample_rate,
            frames_per_stage / args.fps,
            args.fps,
            source_note,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error

    input_description = str(args.input) if args.input else "the synthetic fallback"
    print(f"Created {args.output} from {input_description}.")


if __name__ == "__main__":
    main()
