"""A small live-control GUI for the Kraken-inspired Python prototype."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .amp import AdvancedAmpSettings, Version2Amp, load_cabinet_ir
from .audio_io import LiveAmp, StreamConfig


def launch_gui(args: Any) -> None:
    """Open the desktop control panel.

    Tkinter is intentionally imported only here, so the regular CLI remains
    usable on headless Linux machines without the optional Tk system package.
    """

    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    config = StreamConfig(
        input_device=args.input_device,
        output_device=args.output_device,
        input_channel=args.input_channel,
        output_channels=args.output_channels,
        sample_rate=args.sample_rate,
        blocksize=args.blocksize,
        allow_split_devices=args.allow_split_devices,
    )

    class KrakenControlPanel:
        def __init__(self) -> None:
            self.root = tk.Tk()
            self.root.title("Kraken DSP")
            self.root.minsize(760, 560)
            self.root.protocol("WM_DELETE_WINDOW", self.close)
            self.live_amp: LiveAmp | None = None
            self.stream: Any | None = None
            self._apply_job: str | None = None

            self.channel = tk.StringVar(value=args.channel)
            self.gain = tk.DoubleVar(value=args.gain)
            self.input_gain_db = tk.DoubleVar(value=args.input_gain_db)
            self.bass = tk.DoubleVar(value=args.bass)
            self.middle = tk.DoubleVar(value=args.middle)
            self.treble = tk.DoubleVar(value=args.treble)
            self.master = tk.DoubleVar(value=args.master)
            self.presence = tk.DoubleVar(value=args.presence)
            self.presence_bright = tk.BooleanVar(value=args.presence_bright)
            self.bass_focus = tk.StringVar(value=args.bass_focus)
            self.sag = tk.DoubleVar(value=args.sag)
            self.output_gain_db = tk.DoubleVar(value=args.output_gain_db)
            self.cabinet_bypass = tk.BooleanVar(value=args.cabinet_bypass)
            self.cabinet_ir = tk.StringVar(value=str(args.cabinet_ir or ""))
            self.input_meter = tk.StringVar(value="Input: -- dBFS")
            self.output_meter = tk.StringVar(value="Output: -- dBFS")
            self.status = tk.StringVar(value="Stopped")

            self._build()
            self._tick_meters()

        def _build(self) -> None:
            root = self.root
            root.columnconfigure(0, weight=1)
            root.columnconfigure(1, weight=1)

            header = ttk.Frame(root, padding=(12, 12, 12, 6))
            header.grid(row=0, column=0, columnspan=2, sticky="ew")
            header.columnconfigure(1, weight=1)
            ttk.Label(header, text="Kraken DSP", font=("TkDefaultFont", 16, "bold")).grid(row=0, column=0, sticky="w")
            ttk.Label(
                header,
                text=f"Input {config.input_device or 'auto'}, output {config.output_device or 'auto'}, "
                f"channel {config.input_channel}, {config.sample_rate} Hz",
            ).grid(row=0, column=1, sticky="e")

            preamp = ttk.LabelFrame(root, text="Preamp", padding=12)
            preamp.grid(row=1, column=0, padx=(12, 6), pady=6, sticky="nsew")
            tone = ttk.LabelFrame(root, text="Tone Stack", padding=12)
            tone.grid(row=1, column=1, padx=(6, 12), pady=6, sticky="nsew")
            power = ttk.LabelFrame(root, text="Power Amp", padding=12)
            power.grid(row=2, column=0, padx=(12, 6), pady=6, sticky="nsew")
            output = ttk.LabelFrame(root, text="Cabinet and Output", padding=12)
            output.grid(row=2, column=1, padx=(6, 12), pady=6, sticky="nsew")

            ttk.Label(preamp, text="Channel").grid(row=0, column=0, sticky="w")
            ttk.Radiobutton(preamp, text="Gain I (looser)", variable=self.channel, value="i", command=self.schedule_apply).grid(
                row=0, column=1, sticky="w"
            )
            ttk.Radiobutton(preamp, text="Gain II (tight)", variable=self.channel, value="ii", command=self.schedule_apply).grid(
                row=0, column=2, sticky="w"
            )
            self._slider(preamp, 1, "Gain", self.gain, 0, 10)
            self._slider(preamp, 2, "Input trim (dB)", self.input_gain_db, 0, 36)

            self._slider(tone, 0, "Bass", self.bass, 0, 10)
            self._slider(tone, 1, "Middle", self.middle, 0, 10)
            self._slider(tone, 2, "Treble", self.treble, 0, 10)

            self._slider(power, 0, "Master", self.master, 0, 10)
            self._slider(power, 1, "Presence", self.presence, 0, 10)
            ttk.Checkbutton(power, text="Bright presence", variable=self.presence_bright, command=self.schedule_apply).grid(
                row=2, column=0, columnspan=3, sticky="w", pady=(4, 0)
            )
            ttk.Label(power, text="Bass Focus").grid(row=3, column=0, sticky="w", pady=(8, 0))
            ttk.Radiobutton(power, text="Tight", variable=self.bass_focus, value="tight", command=self.schedule_apply).grid(
                row=3, column=1, sticky="w", pady=(8, 0)
            )
            ttk.Radiobutton(power, text="Loose", variable=self.bass_focus, value="loose", command=self.schedule_apply).grid(
                row=3, column=2, sticky="w", pady=(8, 0)
            )
            self._slider(power, 4, "Sag", self.sag, 0, 10)

            ttk.Label(output, text="Cabinet IR (optional)").grid(row=0, column=0, sticky="w")
            ir_entry = ttk.Entry(output, textvariable=self.cabinet_ir, width=34)
            ir_entry.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 4))
            ir_entry.bind("<KeyRelease>", lambda _event: self.schedule_apply())
            ttk.Button(output, text="Browse…", command=self.choose_ir).grid(row=1, column=2, padx=(6, 0))
            ttk.Checkbutton(output, text="Bypass cabinet", variable=self.cabinet_bypass, command=self.schedule_apply).grid(
                row=2, column=0, columnspan=3, sticky="w"
            )
            self._slider(output, 3, "Output trim (dB)", self.output_gain_db, -30, 0)
            output.columnconfigure(1, weight=1)

            footer = ttk.Frame(root, padding=(12, 8, 12, 12))
            footer.grid(row=3, column=0, columnspan=2, sticky="ew")
            footer.columnconfigure(3, weight=1)
            ttk.Button(footer, text="Start", command=self.start).grid(row=0, column=0, padx=(0, 6))
            ttk.Button(footer, text="Stop", command=self.stop).grid(row=0, column=1, padx=(0, 18))
            ttk.Label(footer, textvariable=self.input_meter).grid(row=0, column=2, sticky="w")
            ttk.Label(footer, textvariable=self.output_meter).grid(row=0, column=3, sticky="w", padx=(18, 0))
            ttk.Label(footer, textvariable=self.status).grid(row=1, column=0, columnspan=4, sticky="w", pady=(6, 0))

        def _slider(
            self,
            parent: Any,
            row: int,
            label: str,
            variable: Any,
            minimum: float,
            maximum: float,
        ) -> None:
            value_label = tk.StringVar()

            def changed(value: str) -> None:
                value_label.set(f"{float(value):.1f}")
                self.schedule_apply()

            value_label.set(f"{variable.get():.1f}")
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(6, 0))
            ttk.Scale(parent, variable=variable, from_=minimum, to=maximum, command=changed).grid(
                row=row, column=1, sticky="ew", padx=(8, 8), pady=(6, 0)
            )
            ttk.Label(parent, textvariable=value_label, width=4).grid(row=row, column=2, sticky="e", pady=(6, 0))
            parent.columnconfigure(1, weight=1)

        def _settings(self) -> AdvancedAmpSettings:
            return AdvancedAmpSettings(
                channel=self.channel.get(),
                gain=self.gain.get(),
                input_gain_db=self.input_gain_db.get(),
                bass=self.bass.get(),
                middle=self.middle.get(),
                treble=self.treble.get(),
                master=self.master.get(),
                presence=self.presence.get(),
                presence_bright=self.presence_bright.get(),
                bass_focus=self.bass_focus.get(),
                sag=self.sag.get(),
                output_gain_db=self.output_gain_db.get(),
                cabinet_bypass=self.cabinet_bypass.get(),
            )

        def _create_amp(self) -> Version2Amp:
            ir_path = self.cabinet_ir.get().strip()
            cabinet_ir = load_cabinet_ir(Path(ir_path), config.sample_rate) if ir_path else None
            return Version2Amp(config.sample_rate, settings=self._settings(), cabinet_ir=cabinet_ir)

        def choose_ir(self) -> None:
            chosen = filedialog.askopenfilename(
                title="Select cabinet impulse response",
                filetypes=(("WAV files", "*.wav"), ("All files", "*")),
            )
            if chosen:
                self.cabinet_ir.set(chosen)
                self.schedule_apply()

        def start(self) -> None:
            if self.stream is not None:
                return
            try:
                self.live_amp = LiveAmp(self._create_amp(), config)
                self.stream = self.live_amp.open_stream()
                self.stream.start()
                self.status.set("Running — changes crossfade in over 25 ms")
            except Exception as error:
                self.stream = None
                self.live_amp = None
                messagebox.showerror("Unable to start audio", str(error))
                self.status.set("Stopped")

        def stop(self) -> None:
            stream, self.stream = self.stream, None
            self.live_amp = None
            if stream is not None:
                try:
                    stream.stop()
                finally:
                    stream.close()
            self.status.set("Stopped")

        def schedule_apply(self) -> None:
            if self.live_amp is None:
                return
            if self._apply_job is not None:
                self.root.after_cancel(self._apply_job)
            self._apply_job = self.root.after(75, self.apply)

        def apply(self) -> None:
            self._apply_job = None
            if self.live_amp is None:
                return
            try:
                self.live_amp.request_processor(self._create_amp())
            except Exception as error:
                messagebox.showerror("Invalid control value", str(error))

        def _tick_meters(self) -> None:
            if self.live_amp is not None:
                meter = self.live_amp.meter
                self.input_meter.set(f"Input: {meter.dbfs(meter.input_peak):.1f} dBFS")
                self.output_meter.set(f"Output: {meter.dbfs(meter.output_peak):.1f} dBFS")
                if self.live_amp.callback_error is not None:
                    self.status.set(f"Audio error: {self.live_amp.callback_error}")
                    self.stop()
            self.root.after(100, self._tick_meters)

        def close(self) -> None:
            self.stop()
            self.root.destroy()

    panel = KrakenControlPanel()
    panel.root.mainloop()
