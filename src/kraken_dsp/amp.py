"""Stateful real-time guitar-amp DSP blocks.

``Version1Amp`` remains as the small baseline chain. ``Version2Amp`` implements
the README's full three-stage preamp with distinct Gain I and Gain II voicings.
All filters and convolvers retain state between blocks for live use.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd, isfinite, pi, sin, sqrt
from pathlib import Path
from typing import Final

import numpy as np
from scipy import signal
from scipy.io import wavfile


DEFAULT_CABINET_TAPS: Final[int] = 257
MAX_CABINET_IR_SAMPLES: Final[int] = 4_096


@dataclass(frozen=True)
class AmpSettings:
    """Starting controls for Version 1 of the amp model.

    The values deliberately leave substantial output headroom.  Raise
    ``input_gain_db`` for more preamp drive; keep the Focusrite's analogue
    preamp below clipping and use this digital control for tonal gain.
    """

    input_gain_db: float = 18.0
    output_gain_db: float = -12.0
    highpass_hz: float = 30.0
    preemphasis_hz: float = 480.0
    preemphasis_db: float = 7.0
    drive: float = 2.4
    bias: float = 0.02
    positive_shape: float = 1.0
    negative_shape: float = 0.8
    post_lowpass_hz: float = 7_000.0
    cabinet_gain_db: float = 0.0
    cabinet_bypass: bool = False
    limiter_ceiling: float = 0.944  # -0.5 dBFS


@dataclass(frozen=True)
class AdvancedAmpSettings:
    """Controls for the Version 2 three-stage high-gain preamp.

    ``gain`` is an amp-style 0--10 control. Gain II is the tight, modern
    high-gain voice and is the default because it is the most useful starting
    point for the Kraken-inspired design.
    """

    channel: str = "ii"
    gain: float = 6.5
    input_gain_db: float = 24.0
    output_gain_db: float = -18.0
    cabinet_gain_db: float = 0.0
    cabinet_bypass: bool = False
    limiter_ceiling: float = 0.944  # -0.5 dBFS


class _SOSFilter:
    """A stateful SOS filter that accepts successive audio blocks."""

    def __init__(self, sos: np.ndarray) -> None:
        self._sos = np.asarray(sos, dtype=np.float64)
        self._zi = np.zeros((self._sos.shape[0], 2), dtype=np.float64)

    def process(self, samples: np.ndarray) -> np.ndarray:
        output, self._zi = signal.sosfilt(self._sos, samples, zi=self._zi)
        return output


class _IIRFilter:
    """A stateful direct-form IIR filter used for DC removal."""

    def __init__(self, b: np.ndarray, a: np.ndarray) -> None:
        self._b = np.asarray(b, dtype=np.float64)
        self._a = np.asarray(a, dtype=np.float64)
        self._zi = np.zeros(max(len(self._a), len(self._b)) - 1, dtype=np.float64)

    def process(self, samples: np.ndarray) -> np.ndarray:
        output, self._zi = signal.lfilter(self._b, self._a, samples, zi=self._zi)
        return output


class _FIRFilter:
    """A streaming FIR convolver for a cabinet impulse response."""

    def __init__(self, taps: np.ndarray) -> None:
        self._taps = np.asarray(taps, dtype=np.float64)
        self._zi = np.zeros(len(self._taps) - 1, dtype=np.float64)

    def process(self, samples: np.ndarray) -> np.ndarray:
        output, self._zi = signal.lfilter(self._taps, [1.0], samples, zi=self._zi)
        return output


def db_to_linear(value_db: float) -> float:
    return 10.0 ** (value_db / 20.0)


def _validate_frequency(name: str, frequency: float, sample_rate: float) -> None:
    if not 0.0 < frequency < sample_rate / 2.0:
        raise ValueError(
            f"{name} must be between 0 Hz and Nyquist ({sample_rate / 2.0:g} Hz), "
            f"got {frequency:g} Hz"
        )


def high_shelf_sos(
    sample_rate: float,
    frequency: float,
    gain_db: float,
    *,
    slope: float = 1.0,
) -> np.ndarray:
    """Return an RBJ high-shelf biquad as a second-order-section array."""

    _validate_frequency("Pre-emphasis frequency", frequency, sample_rate)
    if slope <= 0:
        raise ValueError("Shelf slope must be positive")

    amplitude = 10.0 ** (gain_db / 40.0)
    omega = 2.0 * pi * frequency / sample_rate
    cosine = np.cos(omega)
    alpha = sin(omega) / 2.0 * sqrt((amplitude + 1.0 / amplitude) * (1.0 / slope - 1.0) + 2.0)
    beta = 2.0 * sqrt(amplitude) * alpha

    b0 = amplitude * ((amplitude + 1.0) + (amplitude - 1.0) * cosine + beta)
    b1 = -2.0 * amplitude * ((amplitude - 1.0) + (amplitude + 1.0) * cosine)
    b2 = amplitude * ((amplitude + 1.0) + (amplitude - 1.0) * cosine - beta)
    a0 = (amplitude + 1.0) - (amplitude - 1.0) * cosine + beta
    a1 = 2.0 * ((amplitude - 1.0) - (amplitude + 1.0) * cosine)
    a2 = (amplitude + 1.0) - (amplitude - 1.0) * cosine - beta

    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def asymmetric_tanh(
    samples: np.ndarray,
    *,
    drive: float,
    bias: float,
    positive_shape: float,
    negative_shape: float,
) -> np.ndarray:
    """Apply the README's zero-centred asymmetric tanh waveshaper."""

    shaped_input = drive * (samples + bias)
    shapes = np.where(shaped_input >= 0.0, positive_shape, negative_shape)
    zero_input = drive * bias
    zero_shape = positive_shape if zero_input >= 0.0 else negative_shape
    return np.tanh(shapes * shaped_input) - np.tanh(zero_shape * zero_input)


def default_cabinet_ir(sample_rate: float, taps: int = DEFAULT_CABINET_TAPS) -> np.ndarray:
    """Create a small neutral speaker-like FIR for a usable out-of-box sound.

    It is not a measured cabinet.  It suppresses sub-bass and fizz until the
    user supplies a real mono/stereo WAV IR with ``--cabinet-ir``.
    """

    if taps < 3 or taps % 2 == 0:
        raise ValueError("The built-in cabinet FIR needs an odd number of at least 3 taps")

    nyquist = sample_rate / 2.0
    # The response loosely bounds a guitar speaker/mic: little sub-bass,
    # comparatively flat mids, and a steep upper roll-off.
    frequencies = np.array([0.0, 70.0, 110.0, 4_500.0, min(6_500.0, nyquist * 0.92), nyquist])
    gains = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])

    # Ensure all knots are strictly ordered even at unusual supported rates.
    frequencies = np.maximum.accumulate(frequencies)
    for index in range(1, len(frequencies)):
        if frequencies[index] <= frequencies[index - 1]:
            frequencies[index] = min(nyquist, frequencies[index - 1] + 0.01)
    frequencies[-1] = nyquist
    return signal.firwin2(taps, frequencies, gains, fs=sample_rate)


def _normalise_audio_samples(samples: np.ndarray) -> np.ndarray:
    """Convert common WAV sample formats to floating point in [-1, 1]."""

    samples = np.asarray(samples)
    if np.issubdtype(samples.dtype, np.floating):
        return samples.astype(np.float64, copy=False)
    if np.issubdtype(samples.dtype, np.signedinteger):
        return samples.astype(np.float64) / float(np.iinfo(samples.dtype).max)
    if np.issubdtype(samples.dtype, np.unsignedinteger):
        info = np.iinfo(samples.dtype)
        midpoint = (info.max + 1) / 2.0
        return (samples.astype(np.float64) - midpoint) / midpoint
    raise ValueError(f"Unsupported WAV sample type: {samples.dtype}")


def load_cabinet_ir(
    path: str | Path,
    target_sample_rate: float,
    *,
    max_samples: int = MAX_CABINET_IR_SAMPLES,
) -> np.ndarray:
    """Load, mono-sum, resample, and peak-normalise a short WAV cabinet IR.

    Direct FIR convolution is appropriate for the 1,024--4,096 sample IRs in
    the initial plan. Longer IRs need partitioned convolution, which this
    low-latency prototype deliberately does not implement yet.
    """

    ir_path = Path(path)
    if not ir_path.is_file():
        raise FileNotFoundError(f"Cabinet IR does not exist: {ir_path}")

    source_rate, samples = wavfile.read(ir_path)
    ir = _normalise_audio_samples(samples)
    if ir.ndim == 2:
        ir = ir.mean(axis=1)
    if ir.ndim != 1 or len(ir) == 0:
        raise ValueError(f"Cabinet IR must contain one or more audio samples: {ir_path}")

    if source_rate != int(round(target_sample_rate)):
        divisor = gcd(int(source_rate), int(round(target_sample_rate)))
        ir = signal.resample_poly(ir, int(round(target_sample_rate)) // divisor, int(source_rate) // divisor)

    if len(ir) > max_samples:
        raise ValueError(
            f"Cabinet IR has {len(ir)} samples after resampling; this real-time prototype accepts "
            f"at most {max_samples}. Trim it or use a shorter IR."
        )

    # A DC offset in an IR is not useful here and can make the amp drift.
    ir = ir - np.mean(ir)
    peak = np.max(np.abs(ir))
    if peak <= np.finfo(np.float64).eps:
        raise ValueError(f"Cabinet IR is silent: {ir_path}")
    return ir / peak


class Version1Amp:
    """Stateful Version 1 guitar amp model, intended for one mono channel."""

    def __init__(
        self,
        sample_rate: float,
        settings: AmpSettings | None = None,
        cabinet_ir: np.ndarray | None = None,
    ) -> None:
        if sample_rate not in (44_100, 48_000):
            raise ValueError("Version 1 currently supports 44,100 Hz and 48,000 Hz")
        self.sample_rate = float(sample_rate)
        self.settings = settings or AmpSettings()
        self._validate_settings()

        self._input_highpass = _SOSFilter(
            signal.butter(2, self.settings.highpass_hz, btype="highpass", fs=self.sample_rate, output="sos")
        )
        self._preemphasis = _SOSFilter(
            high_shelf_sos(
                self.sample_rate,
                self.settings.preemphasis_hz,
                self.settings.preemphasis_db,
            )
        )
        self._post_lowpass = _SOSFilter(
            signal.butter(2, self.settings.post_lowpass_hz, btype="lowpass", fs=self.sample_rate, output="sos")
        )
        # y[n] = x[n] - x[n - 1] + 0.995 * y[n - 1]
        self._dc_blocker = _IIRFilter(np.array([1.0, -1.0]), np.array([1.0, -0.995]))

        taps = self._validate_cabinet_taps(cabinet_ir) if cabinet_ir is not None else default_cabinet_ir(self.sample_rate)
        self._cabinet = _FIRFilter(taps)
        self._input_gain = db_to_linear(self.settings.input_gain_db)
        self._output_gain = db_to_linear(self.settings.output_gain_db + self.settings.cabinet_gain_db)

    def _validate_settings(self) -> None:
        numeric_settings = {
            "Input gain": self.settings.input_gain_db,
            "Output gain": self.settings.output_gain_db,
            "High-pass frequency": self.settings.highpass_hz,
            "Pre-emphasis frequency": self.settings.preemphasis_hz,
            "Pre-emphasis gain": self.settings.preemphasis_db,
            "Drive": self.settings.drive,
            "Bias": self.settings.bias,
            "Positive waveshaper shape": self.settings.positive_shape,
            "Negative waveshaper shape": self.settings.negative_shape,
            "Post-distortion low-pass frequency": self.settings.post_lowpass_hz,
            "Cabinet gain": self.settings.cabinet_gain_db,
            "Limiter ceiling": self.settings.limiter_ceiling,
        }
        for name, value in numeric_settings.items():
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
        _validate_frequency("High-pass frequency", self.settings.highpass_hz, self.sample_rate)
        _validate_frequency("Pre-emphasis frequency", self.settings.preemphasis_hz, self.sample_rate)
        _validate_frequency("Post-distortion low-pass frequency", self.settings.post_lowpass_hz, self.sample_rate)
        if not -36.0 <= self.settings.input_gain_db <= 48.0:
            raise ValueError("Input gain must be between -36 dB and +48 dB")
        if not -60.0 <= self.settings.output_gain_db <= 6.0:
            raise ValueError("Output gain must be between -60 dB and +6 dB")
        if not -24.0 <= self.settings.cabinet_gain_db <= 24.0:
            raise ValueError("Cabinet gain must be between -24 dB and +24 dB")
        if not 0.0 < self.settings.drive <= 20.0:
            raise ValueError("Drive must be greater than zero and no more than 20")
        if not -1.0 <= self.settings.bias <= 1.0:
            raise ValueError("Bias must be between -1 and +1")
        if not 0.0 < self.settings.positive_shape <= 5.0 or not 0.0 < self.settings.negative_shape <= 5.0:
            raise ValueError("Waveshaper shape values must be greater than zero and no more than 5")
        if not 0.0 < self.settings.limiter_ceiling <= 1.0:
            raise ValueError("Limiter ceiling must be in the range (0, 1]")

    @staticmethod
    def _validate_cabinet_taps(cabinet_ir: np.ndarray) -> np.ndarray:
        taps = np.asarray(cabinet_ir, dtype=np.float64)
        if taps.ndim != 1 or len(taps) == 0:
            raise ValueError("Cabinet IR must be a non-empty one-dimensional array")
        if len(taps) > MAX_CABINET_IR_SAMPLES:
            raise ValueError(
                f"Cabinet IR has {len(taps)} samples; this real-time prototype accepts at most "
                f"{MAX_CABINET_IR_SAMPLES}"
            )
        if not np.all(np.isfinite(taps)):
            raise ValueError("Cabinet IR must contain only finite samples")
        return taps

    def process_block(self, input_samples: np.ndarray) -> np.ndarray:
        """Process a mono floating-point block and return float32 output.

        The method is safe to call repeatedly with arbitrary block sizes, but
        callers must not call it concurrently because it owns filter state.
        """

        x = np.asarray(input_samples, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError("Version1Amp expects a one-dimensional mono block")
        if len(x) == 0:
            return np.empty(0, dtype=np.float32)

        x = self._input_highpass.process(x)
        x *= self._input_gain
        x = self._preemphasis.process(x)

        x = asymmetric_tanh(
            x,
            drive=self.settings.drive,
            bias=self.settings.bias,
            positive_shape=self.settings.positive_shape,
            negative_shape=self.settings.negative_shape,
        )

        x = self._post_lowpass.process(x)
        x = self._dc_blocker.process(x)
        if not self.settings.cabinet_bypass:
            x = self._cabinet.process(x)
        x *= self._output_gain
        np.clip(x, -self.settings.limiter_ceiling, self.settings.limiter_ceiling, out=x)
        return x.astype(np.float32)


@dataclass(frozen=True)
class _PreampProfile:
    stage1_shelf_db: float
    stage2_shelf_db: float
    stage2_highpass_hz: float
    stage3_highpass_hz: float
    post_lowpass_hz: float
    stage1_gain_start: float
    stage1_gain_span: float
    stage2_gain_start: float
    stage2_gain_span: float
    stage3_gain_start: float
    stage3_gain_span: float
    stage1_drive_start: float
    stage1_drive_span: float
    stage2_drive_start: float
    stage2_drive_span: float
    stage3_drive_start: float
    stage3_drive_span: float


_PREAMP_PROFILES: Final[dict[str, _PreampProfile]] = {
    # Gain I: rounder, darker, and more open through the low mids.
    "i": _PreampProfile(
        stage1_shelf_db=6.0,
        stage2_shelf_db=7.0,
        stage2_highpass_hz=45.0,
        stage3_highpass_hz=65.0,
        post_lowpass_hz=6_000.0,
        stage1_gain_start=3.0,
        stage1_gain_span=9.0,
        stage2_gain_start=2.0,
        stage2_gain_span=8.0,
        stage3_gain_start=1.5,
        stage3_gain_span=5.0,
        stage1_drive_start=1.4,
        stage1_drive_span=1.0,
        stage2_drive_start=2.4,
        stage2_drive_span=3.5,
        stage3_drive_start=2.0,
        stage3_drive_span=2.2,
    ),
    # Gain II: tighter bass before clipping and stronger later-stage drive.
    "ii": _PreampProfile(
        stage1_shelf_db=8.0,
        stage2_shelf_db=12.0,
        stage2_highpass_hz=85.0,
        stage3_highpass_hz=85.0,
        post_lowpass_hz=7_000.0,
        stage1_gain_start=4.0,
        stage1_gain_span=12.0,
        stage2_gain_start=3.0,
        stage2_gain_span=16.0,
        stage3_gain_start=2.0,
        stage3_gain_span=9.0,
        stage1_drive_start=1.6,
        stage1_drive_span=1.2,
        stage2_drive_start=3.2,
        stage2_drive_span=4.8,
        stage3_drive_start=2.5,
        stage3_drive_span=3.2,
    ),
}


class Version2Amp:
    """Three-stage high-gain preamp with the README's Gain I/II voicings.

    The nonlinear stages deliberately run at the native rate for this Python
    milestone. Oversampling is the next quality-focused step once the voicing
    is auditioned on real guitar input.
    """

    def __init__(
        self,
        sample_rate: float,
        settings: AdvancedAmpSettings | None = None,
        cabinet_ir: np.ndarray | None = None,
    ) -> None:
        if sample_rate not in (44_100, 48_000):
            raise ValueError("Version 2 currently supports 44,100 Hz and 48,000 Hz")
        self.sample_rate = float(sample_rate)
        self.settings = settings or AdvancedAmpSettings()
        self._validate_settings()
        self.channel = self.settings.channel.casefold()
        profile = _PREAMP_PROFILES[self.channel]
        gain_position = self.settings.gain / 10.0

        self._input_highpass = _SOSFilter(
            signal.butter(2, 30.0, btype="highpass", fs=self.sample_rate, output="sos")
        )
        self._stage1_shelf = _SOSFilter(high_shelf_sos(self.sample_rate, 480.0, profile.stage1_shelf_db))
        self._stage2_highpass = _SOSFilter(
            signal.butter(2, profile.stage2_highpass_hz, btype="highpass", fs=self.sample_rate, output="sos")
        )
        self._stage2_shelf = _SOSFilter(high_shelf_sos(self.sample_rate, 170.0, profile.stage2_shelf_db))
        self._stage3_highpass = _SOSFilter(
            signal.butter(2, profile.stage3_highpass_hz, btype="highpass", fs=self.sample_rate, output="sos")
        )
        self._stage3_lowpass = _SOSFilter(
            signal.butter(2, profile.post_lowpass_hz, btype="lowpass", fs=self.sample_rate, output="sos")
        )
        self._stage1_dc = _IIRFilter(np.array([1.0, -1.0]), np.array([1.0, -0.995]))
        self._stage2_dc = _IIRFilter(np.array([1.0, -1.0]), np.array([1.0, -0.995]))
        self._stage3_dc = _IIRFilter(np.array([1.0, -1.0]), np.array([1.0, -0.995]))

        self._input_gain = db_to_linear(self.settings.input_gain_db)
        self._output_gain = db_to_linear(self.settings.output_gain_db + self.settings.cabinet_gain_db)
        self._stage1_gain = profile.stage1_gain_start + profile.stage1_gain_span * gain_position
        self._stage2_gain = profile.stage2_gain_start + profile.stage2_gain_span * gain_position
        self._stage3_gain = profile.stage3_gain_start + profile.stage3_gain_span * gain_position
        self._stage1_drive = profile.stage1_drive_start + profile.stage1_drive_span * gain_position
        self._stage2_drive = profile.stage2_drive_start + profile.stage2_drive_span * gain_position
        self._stage3_drive = profile.stage3_drive_start + profile.stage3_drive_span * gain_position

        taps = Version1Amp._validate_cabinet_taps(cabinet_ir) if cabinet_ir is not None else default_cabinet_ir(self.sample_rate)
        self._cabinet = _FIRFilter(taps)

    def _validate_settings(self) -> None:
        numeric_settings = {
            "Gain": self.settings.gain,
            "Input gain": self.settings.input_gain_db,
            "Output gain": self.settings.output_gain_db,
            "Cabinet gain": self.settings.cabinet_gain_db,
            "Limiter ceiling": self.settings.limiter_ceiling,
        }
        for name, value in numeric_settings.items():
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.settings.channel.casefold() not in _PREAMP_PROFILES:
            raise ValueError("Channel must be 'i' or 'ii'")
        if not 0.0 <= self.settings.gain <= 10.0:
            raise ValueError("Gain must be between 0 and 10")
        if not -36.0 <= self.settings.input_gain_db <= 48.0:
            raise ValueError("Input gain must be between -36 dB and +48 dB")
        if not -60.0 <= self.settings.output_gain_db <= 6.0:
            raise ValueError("Output gain must be between -60 dB and +6 dB")
        if not -24.0 <= self.settings.cabinet_gain_db <= 24.0:
            raise ValueError("Cabinet gain must be between -24 dB and +24 dB")
        if not 0.0 < self.settings.limiter_ceiling <= 1.0:
            raise ValueError("Limiter ceiling must be in the range (0, 1]")

    def process_block(self, input_samples: np.ndarray) -> np.ndarray:
        """Process one mono block through all three preamp stages."""

        x = np.asarray(input_samples, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError("Version2Amp expects a one-dimensional mono block")
        if len(x) == 0:
            return np.empty(0, dtype=np.float32)

        x = self._input_highpass.process(x)
        x *= self._input_gain

        x = self._stage1_shelf.process(x)
        x *= self._stage1_gain
        x = asymmetric_tanh(x, drive=self._stage1_drive, bias=0.02, positive_shape=1.0, negative_shape=0.8)
        x = self._stage1_dc.process(x)

        x = self._stage2_highpass.process(x)
        x = self._stage2_shelf.process(x)
        x *= self._stage2_gain
        x = asymmetric_tanh(x, drive=self._stage2_drive, bias=0.04, positive_shape=1.2, negative_shape=0.75)
        x = self._stage2_dc.process(x)

        x = self._stage3_highpass.process(x)
        x *= self._stage3_gain
        x = asymmetric_tanh(x, drive=self._stage3_drive, bias=0.03, positive_shape=1.4, negative_shape=0.9)
        x = self._stage3_lowpass.process(x)
        x = self._stage3_dc.process(x)

        if not self.settings.cabinet_bypass:
            x = self._cabinet.process(x)
        x *= self._output_gain
        np.clip(x, -self.settings.limiter_ceiling, self.settings.limiter_ceiling, out=x)
        return x.astype(np.float32)
