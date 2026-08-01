from __future__ import annotations

import numpy as np
import pytest
from scipy.io import wavfile

from kraken_dsp.amp import (
    AMP_STAGE_TAP_NAMES,
    MAX_CABINET_IR_SAMPLES,
    AdvancedAmpSettings,
    AmpSettings,
    Version1Amp,
    Version2Amp,
    default_cabinet_ir,
    load_cabinet_ir,
)
from kraken_dsp.walkthrough import _parse_args, _process_stage_taps


@pytest.mark.parametrize("sample_rate", [44_100, 48_000])
def test_amp_produces_finite_limited_mono_output(sample_rate: int) -> None:
    amp = Version1Amp(sample_rate)
    time = np.arange(2_048) / sample_rate
    guitar_like_input = 0.08 * np.sin(2 * np.pi * 110 * time) + 0.02 * np.sin(2 * np.pi * 440 * time)

    output = amp.process_block(guitar_like_input)

    assert output.shape == guitar_like_input.shape
    assert output.dtype == np.float32
    assert np.all(np.isfinite(output))
    assert np.max(np.abs(output)) <= amp.settings.limiter_ceiling + 1e-6
    assert np.max(np.abs(output)) > 1e-4


@pytest.mark.parametrize("sample_rate", [44_100, 48_000])
def test_dc_blocker_removes_steady_asymmetric_offset(sample_rate: int) -> None:
    amp = Version1Amp(sample_rate, settings=AmpSettings(cabinet_bypass=True))

    for _ in range(60):
        output = amp.process_block(np.full(128, 0.1))

    assert abs(float(np.mean(output))) < 1e-3


def test_default_cabinet_has_expected_length_and_finite_values() -> None:
    cabinet = default_cabinet_ir(48_000)
    assert cabinet.shape == (257,)
    assert np.all(np.isfinite(cabinet))


def test_amp_rejects_unsupported_sample_rate() -> None:
    with pytest.raises(ValueError, match="44,100"):
        Version1Amp(96_000)


@pytest.mark.parametrize(
    "settings, message",
    [
        (AmpSettings(drive=float("nan")), "Drive must be finite"),
        (AmpSettings(input_gain_db=float("inf")), "Input gain must be finite"),
        (AmpSettings(drive=21.0), "Drive must be greater"),
    ],
)
def test_amp_rejects_nonfinite_and_unsafe_controls(settings: AmpSettings, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Version1Amp(48_000, settings=settings)


def test_amp_rejects_an_unbounded_direct_cabinet_ir() -> None:
    with pytest.raises(ValueError, match="at most"):
        Version1Amp(48_000, cabinet_ir=np.ones(MAX_CABINET_IR_SAMPLES + 1))


def test_loader_rejects_long_ir_after_resampling(tmp_path) -> None:
    ir_path = tmp_path / "long_ir.wav"
    wavfile.write(ir_path, 48_000, np.ones(MAX_CABINET_IR_SAMPLES + 1, dtype=np.float32))

    with pytest.raises(ValueError, match="after resampling"):
        load_cabinet_ir(ir_path, 48_000)


@pytest.mark.parametrize("sample_rate", [44_100, 48_000])
def test_version2_amp_is_finite_and_limited(sample_rate: int) -> None:
    amp = Version2Amp(sample_rate)
    time = np.arange(4_096) / sample_rate
    guitar_like_input = 0.035 * np.sin(2 * np.pi * 110 * time) + 0.012 * np.sin(2 * np.pi * 440 * time)

    output = amp.process_block(guitar_like_input)

    assert output.dtype == np.float32
    assert np.all(np.isfinite(output))
    assert np.max(np.abs(output)) <= amp.settings.limiter_ceiling + 1e-6
    assert np.max(np.abs(output)) > 1e-3


def test_gain_i_and_gain_ii_have_distinct_voicings() -> None:
    samples = 0.04 * np.sin(2 * np.pi * 110 * np.arange(4_096) / 48_000)
    gain_i = Version2Amp(48_000, AdvancedAmpSettings(channel="i", gain=6.5))
    gain_ii = Version2Amp(48_000, AdvancedAmpSettings(channel="ii", gain=6.5))

    output_i = gain_i.process_block(samples)
    output_ii = gain_ii.process_block(samples)

    assert not np.allclose(output_i, output_ii, atol=1e-4)


def test_clean_mode_is_distinct_from_gain_ii_and_uses_master_i() -> None:
    samples = 0.05 * np.sin(2 * np.pi * 220 * np.arange(4_096) / 48_000)
    clean = Version2Amp(48_000, AdvancedAmpSettings(channel="clean", gain=2, master_i=6, master_ii=1, presence=0, sag=0))
    clean_with_other_master = Version2Amp(
        48_000, AdvancedAmpSettings(channel="clean", gain=2, master_i=6, master_ii=9, presence=0, sag=0)
    )
    gain_ii = Version2Amp(48_000, AdvancedAmpSettings(channel="ii", gain=2, master_ii=6, presence=0, sag=0))

    clean_output = clean.process_block(samples)
    clean_other_master_output = clean_with_other_master.process_block(samples)
    gain_ii_output = gain_ii.process_block(samples)

    assert np.allclose(clean_output, clean_other_master_output)
    assert not np.allclose(clean_output, gain_ii_output, atol=1e-4)


def test_clean_gain_i_balance_changes_only_the_clean_mode_level() -> None:
    samples = 0.05 * np.sin(2 * np.pi * 220 * np.arange(4_096) / 48_000)
    quiet_clean = Version2Amp(48_000, AdvancedAmpSettings(channel="clean", gain_i_balance_db=-6, presence=0, sag=0))
    loud_clean = Version2Amp(48_000, AdvancedAmpSettings(channel="clean", gain_i_balance_db=6, presence=0, sag=0))

    assert np.sqrt(np.mean(np.square(loud_clean.process_block(samples)))) > np.sqrt(
        np.mean(np.square(quiet_clean.process_block(samples)))
    ) * 2


def test_tone_and_power_controls_change_the_version2_output() -> None:
    samples = 0.04 * np.sin(2 * np.pi * 110 * np.arange(4_096) / 48_000)
    neutral = Version2Amp(48_000, AdvancedAmpSettings(bass=5, middle=5, treble=5, master=6, presence=0, sag=0))
    shaped = Version2Amp(
        48_000,
        AdvancedAmpSettings(bass=8, middle=2, treble=8, master=9, presence=8, presence_bright=True, bass_focus="loose", sag=6),
    )

    neutral_output = neutral.process_block(samples)
    shaped_output = shaped.process_block(samples)

    assert not np.allclose(neutral_output, shaped_output, atol=1e-4)


def test_version2_stage_taps_follow_the_live_processing_path() -> None:
    samples = 0.04 * np.sin(2 * np.pi * 110 * np.arange(1_024) / 48_000)
    inspected_amp = Version2Amp(48_000)
    live_amp = Version2Amp(48_000)

    inspected_output, taps = inspected_amp.process_block_with_taps(samples)
    live_output = live_amp.process_block(samples)

    assert tuple(taps) == AMP_STAGE_TAP_NAMES
    assert all(tap.shape == samples.shape for tap in taps.values())
    assert np.allclose(inspected_output, live_output)
    assert np.allclose(taps["Cabinet + output"], inspected_output)


def test_walkthrough_accepts_the_same_bright_presence_and_cabinet_options_as_live_mode() -> None:
    args = _parse_args(["--presence-bright", "--cabinet-ir", "example.wav", "--cabinet-bypass"])

    assert args.presence_bright is True
    assert args.cabinet_ir.name == "example.wav"
    assert args.cabinet_bypass is True


def test_walkthrough_uses_the_full_input_by_default() -> None:
    args = _parse_args([])

    assert args.seconds_per_stage is None


def test_walkthrough_final_stage_matches_live_processing_at_the_same_blocksize() -> None:
    samples = 0.04 * np.sin(2 * np.pi * 110 * np.arange(1_024) / 48_000)
    settings = AdvancedAmpSettings(sag=3.0)
    walkthrough_taps = _process_stage_taps(samples, 48_000, settings, blocksize=128)
    live_amp = Version2Amp(48_000, settings)

    live_output = np.concatenate(
        [live_amp.process_block(samples[start : start + 128]) for start in range(0, len(samples), 128)]
    )

    assert np.allclose(walkthrough_taps["Cabinet + output"], live_output)


def test_master_control_reduces_output_level() -> None:
    samples = 0.03 * np.sin(2 * np.pi * 220 * np.arange(4_096) / 48_000)
    quiet = Version2Amp(48_000, AdvancedAmpSettings(master=1, sag=0))
    loud = Version2Amp(48_000, AdvancedAmpSettings(master=9, sag=0))

    quiet_output = quiet.process_block(samples)
    loud_output = loud.process_block(samples)

    assert np.sqrt(np.mean(np.square(loud_output))) > np.sqrt(np.mean(np.square(quiet_output))) * 4


@pytest.mark.parametrize(
    "settings, message",
    [
        (AdvancedAmpSettings(channel="iii"), "Channel"),
        (AdvancedAmpSettings(gain_i_balance_db=19), "Gain I balance"),
        (AdvancedAmpSettings(gain=10.1), "Gain"),
        (AdvancedAmpSettings(gain=float("nan")), "Gain must be finite"),
        (AdvancedAmpSettings(bass=10.1), "Bass"),
        (AdvancedAmpSettings(bass_focus="neutral"), "Bass Focus"),
    ],
)
def test_version2_rejects_invalid_controls(settings: AdvancedAmpSettings, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Version2Amp(48_000, settings)
