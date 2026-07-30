# Kraken-Inspired Guitar Amp DSP

A real-time digital guitar amplifier model inspired by the high-gain character of the Victory Kraken.

This is **not an exact Victory Kraken emulation**. No verified full Kraken schematic or hardware measurements are available for this project. The design is based on:

* General gray-box guitar amplifier modelling techniques
* Public information about the Victory Kraken
* An unofficial Kraken-inspired pedal/preamp schematic
* Standard high-gain amplifier DSP practices

The first goal is to create a convincing Kraken-like high-gain amplifier using conventional DSP blocks rather than neural amp capture.

## Signal Chain

```text
Guitar input
    │
    ▼
DC blocker
    │
    ▼
Input conditioning
    │
    ▼
4× or 8× oversampling
    │
    ▼
Preamp stage 1
    │
    ▼
Gain control
    │
    ▼
Preamp stage 2
    │
    ▼
Preamp stage 3
    │
    ▼
Bass / Middle / Treble tone stack
    │
    ▼
Master volume
    │
    ▼
Power-amp saturation
    │
    ▼
Presence and Bass Focus
    │
    ▼
Cabinet impulse response
    │
    ▼
Downsampling
    │
    ▼
Output
```

## Main DSP Structure

The amplifier should be implemented as a sequence of linear filters and nonlinear waveshaping stages:

```text
x → H1(z) → N1(x) → H2(z) → N2(x) → H3(z) → N3(x) → tone stack
```

Where:

* `Hn(z)` is a digital filter
* `Nn(x)` is a nonlinear clipping or saturation function
* Filters before distortion determine how different frequencies drive the clipping
* Filters after distortion shape the generated harmonics

For high-gain guitar tones, filtering **before** distortion is especially important.

Too much bass entering a distortion stage produces loose, muddy low-frequency intermodulation. A tight high-gain channel should reduce bass before the strongest clipping stages.

## Initial Preamp Design

### Stage 1

Purpose:

* Input amplification
* Initial harmonic generation
* Moderate asymmetric clipping
* Preserve some low-mid body

Starting parameters:

```text
High-pass frequency: 20–40 Hz
Pre-emphasis transition: approximately 480 Hz
Maximum gain: approximately 20–25×
Clipping: soft and asymmetric
```

Example:

```text
Input
  → high-pass
  → high-shelf or shelving gain
  → nonlinear waveshaper
  → DC blocker
```

### Stage 2

Purpose:

* Main high-gain distortion stage
* Bass tightening before clipping
* Stronger asymmetric saturation

Starting parameters:

```text
Pre-emphasis transition: approximately 170 Hz
Maximum gain: approximately 40–50×
Clipping: stronger than stage 1
Low frequencies should receive less gain than mids and highs
```

This stage will have a major effect on whether the amplifier sounds tight or muddy.

### Stage 3

Purpose:

* Additional saturation and compression
* Final preamp harmonic generation
* High-frequency smoothing

Starting parameters:

```text
Input high-pass: approximately 70 Hz
Linear gain: approximately 10×
Post-distortion low-pass: approximately 7 kHz
Clipping: moderately hard asymmetric saturation
```

## Waveshaping

Start with an asymmetric `tanh` waveshaper:

```cpp
float asymmetricClip(
    float x,
    float drive,
    float bias,
    float positiveShape,
    float negativeShape)
{
    const float input = drive * (x + bias);

    float output;

    if (input >= 0.0f) {
        output = std::tanh(positiveShape * input);
    } else {
        output = std::tanh(negativeShape * input);
    }

    const float zeroInput = drive * bias;

    const float zeroOutput =
        zeroInput >= 0.0f
            ? std::tanh(positiveShape * zeroInput)
            : std::tanh(negativeShape * zeroInput);

    return output - zeroOutput;
}
```

Example stage settings:

```text
Stage 1:
drive = 1.5–3.0
bias = 0.02
positiveShape = 1.0
negativeShape = 0.8

Stage 2:
drive = 3.0–8.0
bias = 0.04
positiveShape = 1.2
negativeShape = 0.75

Stage 3:
drive = 2.0–5.0
bias = 0.03
positiveShape = 1.4
negativeShape = 0.9
```

These values are only starting points.

Add a high-pass filter after each asymmetric stage to remove DC offset.

## Oversampling

Nonlinear waveshaping creates harmonics above the Nyquist frequency. Without oversampling, these harmonics fold back into the audible spectrum as aliasing.

Use at least:

```text
4× oversampling for development
8× oversampling for higher-quality rendering
```

Only the nonlinear preamp and power-amp sections need to run at the oversampled rate.

Suggested structure:

```text
Input
  → oversampling interpolation filter
  → nonlinear amplifier stages
  → oversampling decimation filter
  → cabinet processing
```

The cabinet impulse response can normally run at the original sample rate.

## Tone Stack

For the first implementation, use three biquad filters instead of attempting an exact passive tube-amp tone-stack simulation.

### Bass

```text
Filter: low shelf
Frequency: 100–150 Hz
Range: approximately ±10 dB
```

### Middle

```text
Filter: peaking EQ
Frequency: 600–900 Hz
Q: 0.5–1.0
Range: approximately ±10 dB
```

### Treble

```text
Filter: high shelf
Frequency: 3–4 kHz
Range: approximately ±10 dB
```

Later, replace this with a coupled passive tone-stack model using:

* State-space modelling
* Nodal analysis
* Wave digital filters

## Gain Channels

Implement two channels inspired by the Kraken’s Gain I and Gain II modes.

### Gain I

Looser, more British-style distortion.

Suggested characteristics:

```text
Lower stage 2 and stage 3 drive
Softer clipping
More low-mid content
Less aggressive bass filtering
Slightly darker post-distortion filtering
More dynamic response
```

### Gain II

Tighter modern high-gain distortion.

Suggested characteristics:

```text
Higher stage 2 and stage 3 drive
More bass removed before clipping
Harder clipping
More upper-mid and presence content
Stronger compression
Tighter power-amp low end
```

Do not implement Gain II as Gain I with only a larger gain value. The pre-distortion filtering should also change.

## Power-Amp Model

The first power-amp implementation can be simple.

```text
Master volume
  → soft saturation
  → sag model
  → presence filter
  → Bass Focus / resonance filter
```

### Power Saturation

Use a softer waveshaper than the preamp:

```cpp
float powerAmpClip(float x, float drive)
{
    return std::tanh(x * drive);
}
```

The power amp should distort mainly when the master volume is high.

### Sag

Sag simulates the temporary power-supply voltage reduction caused by loud signals.

Basic approach:

```text
absolute input
  → envelope follower
  → low-pass filter
  → reduce power-amp gain
```

Example:

```cpp
envelope += attackReleaseCoefficient *
            (std::abs(input) - envelope);

float sagGain = 1.0f / (1.0f + sagAmount * envelope);

output = powerAmpClip(input * sagGain, powerDrive);
```

Use relatively slow attack and release times, for example:

```text
Attack: 10–50 ms
Release: 100–500 ms
```

### Presence

Presence should boost or reduce high-frequency power-amp feedback behaviour.

For the first version, approximate it using a high shelf:

```text
Frequency: 2.5–4 kHz
Range: approximately 0–8 dB
```

### Bass Focus

Bass Focus should change low-frequency damping rather than behaving exactly like a normal bass EQ.

Initial approximation:

```text
Low shelf or resonant low-frequency filter
Frequency: 70–140 Hz
```

Low setting:

```text
More low-frequency resonance
Looser feel
Longer bass response
```

High setting:

```text
Less resonance
Tighter low end
Faster damping
```

## Cabinet Simulation

A distorted preamp without a guitar speaker simulation will sound extremely harsh.

Use a guitar cabinet impulse response after the amplifier model:

```text
Amp output → cabinet IR convolution → final output
```

Good starting cabinet types:

```text
4×12 cabinet
Celestion Vintage 30-style speakers
Dynamic microphone near the cone edge
```

Use either:

* Direct convolution for short IRs
* FFT convolution
* Partitioned convolution for low-latency real-time processing

A 1024–4096 sample IR is sufficient for an initial implementation.

## Suggested Controls

```text
Input
Gain
Channel: Gain I / Gain II
Bass
Middle
Treble
Master
Presence
Bass Focus
Cabinet bypass
Output
```

Optional later controls:

```text
Noise gate
Sag
Power-amp drive
Microphone position
Cabinet selection
Oversampling quality
```

## Recommended Processing Order

```cpp
float processSample(float input)
{
    float x = input;

    x = inputDCBlocker.process(x);
    x *= inputGain;

    // Enter oversampled processing here.

    x = stage1PreFilter.process(x);
    x = stage1Clip(x);
    x = stage1DCBlocker.process(x);

    x *= gainControl;

    x = stage2PreFilter.process(x);
    x = stage2Clip(x);
    x = stage2DCBlocker.process(x);

    x = stage3HighPass.process(x);
    x *= stage3Gain;
    x = stage3Clip(x);
    x = stage3LowPass.process(x);
    x = stage3DCBlocker.process(x);

    x = bassFilter.process(x);
    x = midFilter.process(x);
    x = trebleFilter.process(x);

    x *= masterVolume;

    x = processSag(x);
    x = powerAmpClip(x, powerAmpDrive);

    x = presenceFilter.process(x);
    x = bassFocusFilter.process(x);

    // Leave oversampled processing here.

    x = cabinetConvolution.process(x);

    x *= outputGain;

    return x;
}
```

## Development Order

### Version 1: Basic distortion

Implement:

```text
Input gain
One pre-emphasis filter
One tanh waveshaper
One cabinet IR
Output gain
```

Confirm that real-time audio works before adding complexity.

### Version 2: Full preamp

Add:

```text
Three distortion stages
Stage-specific filtering
Asymmetric clipping
DC blockers
Gain I and Gain II modes
```

### Version 3: Tone controls

Add:

```text
Bass
Middle
Treble
```

### Version 4: Power amp

Add:

```text
Master-dependent saturation
Sag
Presence
Bass Focus
```

### Version 5: Optimization

Add:

```text
4× or 8× oversampling
SIMD processing
Parameter smoothing
Low-latency convolution
Preset support
```

## Testing

Use a clean DI guitar recording so every version receives the same input.

Test:

```text
Single notes
Palm-muted low notes
Open chords
Power chords
Fast alternate picking
Volume-knob cleanup
Long sustained notes
```

Listen for:

```text
Aliasing
Harsh high frequencies
Muddy palm mutes
Excessive DC offset
Clicks when parameters change
Unstable output level
Too much compression
Poor cleanup when guitar volume is reduced
```

Also inspect:

```text
Input and output waveforms
Frequency spectrum
DC level
Peak level
RMS level
Harmonic content from sine-wave tests
```

## Important Implementation Details

### Parameter smoothing

Do not change gain or filter coefficients instantly.

Smooth controls over approximately:

```text
5–50 ms
```

This prevents clicking and zipper noise.

### Level management

Each nonlinear stage can greatly increase or reduce signal level.

Add explicit scaling around each stage:

```text
preGain → waveshaper → postGain
```

Do not rely on the final output control to fix every internal level problem.

### Noise gate

High-gain stages amplify guitar and interface noise.

Add a gate later, preferably before the main distortion stages:

```text
Input → gate → amplifier
```

### Sample rates

Initially support:

```text
44.1 kHz
48 kHz
```

Test the filter calculations at both sample rates.

## Possible Project Structure

```text
src/
├── AmpModel.cpp
├── AmpModel.h
├── Biquad.cpp
├── Biquad.h
├── Waveshaper.cpp
├── Waveshaper.h
├── Oversampler.cpp
├── Oversampler.h
├── CabinetConvolver.cpp
├── CabinetConvolver.h
├── EnvelopeFollower.cpp
├── EnvelopeFollower.h
└── ParameterSmoother.h

tests/
├── test_biquad.cpp
├── test_waveshaper.cpp
├── test_dc_blocker.cpp
└── test_amp_model.cpp

resources/
└── cabinet_ir.wav
```

## Useful Framework Options

### JUCE

Recommended for building a standalone application or audio plugin.

Useful JUCE classes include:

```text
juce::dsp::IIR::Filter
juce::dsp::Oversampling
juce::dsp::Convolution
juce::SmoothedValue
juce::AudioProcessorValueTreeState
```

### Python prototype

The repository now includes a live Python implementation of the first milestone
as well as a useful foundation for offline testing. It uses `sounddevice`
(PortAudio/Core Audio on macOS), NumPy, and SciPy to take a mono input from an
audio interface, process it, and return the processed mono signal on the
interface outputs.

The current live chain is intentionally small:

```text
Focusrite guitar input
  → 30 Hz high-pass
  → 480 Hz high-shelf pre-emphasis
  → asymmetric tanh distortion
  → 7 kHz low-pass + DC blocker
  → cabinet FIR
  → output safety limiter
  → Focusrite output
```

This is Version 1 only: it deliberately does not yet include the three-stage
preamp, tone controls, sag, or oversampling. Those come after proving that the
live input and output routing is stable.

#### Run the live Python prototype

Connect the Focusrite, plug the guitar into an **INST/Hi-Z** input, and begin
with the interface's output volume low. Use headphones while testing. Turn down
or disable Direct Monitor if you only want to hear the processed signal; do not
route the speakers back into the guitar pickup/microphone path.

Install the project in a Python 3.10+ environment:

```bash
python3 -m pip install -e .
```

First list the devices that Core Audio exposes:

```bash
kraken-dsp devices
```

Then verify capture without playing anything back. Focusrite input channels are
shown to users as **one-based** numbers, so the physical Input 1 is
`--input-channel 1` and physical Input 2 is `--input-channel 2`.

```bash
kraken-dsp capture --input-channel 1
```

Strum the guitar and watch the input meter. Set the analogue Focusrite gain so
hard strums peak roughly between -12 and -6 dBFS; avoid reaching 0 dBFS because
that means the interface input itself is clipping.

Once capture is confirmed, run the amp. It automatically chooses a uniquely
named Focusrite/Scarlett/Clarett/Vocaster device when possible:

```bash
kraken-dsp run --input-channel 1
```

If more than one audio device is connected, use the device index or an
unambiguous name fragment from `kraken-dsp devices`:

```bash
kraken-dsp run \
  --input-device 3 \
  --output-device 3 \
  --input-channel 1 \
  --sample-rate 48000 \
  --blocksize 128
```

The live output starts conservatively at -12 dB. The two primary controls for
the first tone pass are digital preamp gain and waveshaper drive:

```bash
kraken-dsp run --input-gain-db 20 --drive 3.0 --output-gain-db -12
```

The app includes a short, neutral speaker-like FIR so it works without a binary
asset. For a much more realistic result, supply a cabinet WAV IR; stereo IRs
are mono-summed and IRs at a different 44.1/48 kHz rate are resampled. For
low-latency direct convolution, use an IR no longer than 4,096 samples after
resampling:

```bash
kraken-dsp run --cabinet-ir /path/to/cabinet_ir.wav
```

If macOS lists no inputs, grant the terminal/Python host **Microphone** access
in System Settings → Privacy & Security, reconnect the Focusrite, and verify
its sample rate in Audio MIDI Setup. Keep the input and output on the same
Focusrite. The app enforces that by default; using separate devices requires
the explicit `--allow-split-devices` switch and can cause clocking issues.

## Current Limitations

This model is based on approximate DSP design rather than measurements from a real amplifier.

Therefore:

* Filter frequencies are starting estimates
* Nonlinearities are not fitted to physical tube stages
* Gain I and Gain II behaviour is interpretive
* The power amp is heavily simplified
* The cabinet IR will strongly influence the final sound

The objective is initially to produce a convincing high-gain amplifier, not a component-perfect reconstruction.

## References

* DAFx gray-box guitar amplifier modelling research
* Victory VX Kraken public product information
* PedalPCB Cetus Preamp schematic
* General virtual-analog and nonlinear audio DSP techniques

## First Milestone

The first usable milestone should be:

```text
Guitar input
  → high-pass
  → pre-emphasis
  → asymmetric tanh distortion
  → post-distortion low-pass
  → cabinet IR
  → output
```

Once this sounds acceptable, expand it into the complete three-stage architecture.
