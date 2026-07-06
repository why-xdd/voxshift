"""VoxShift — real-time voice changer.

Captures the microphone, pitch-shifts the signal with a phase vocoder,
applies optional ring modulation and plays the result to any output
device (e.g. VB-Cable, so the changed voice can be used as a virtual
microphone in Discord or games).

Run:  python voxshift.py
"""

import threading

import numpy as np

SAMPLE_RATE = 44100
BLOCK_SIZE = 512

PRESETS = {
    "Normal":   {"semitones": 0,  "ring_freq": 0.0,  "ring_mix": 0.0, "emoji": "🎤"},
    "Robot":    {"semitones": 0,  "ring_freq": 70.0, "ring_mix": 0.9, "emoji": "🤖"},
    "Chipmunk": {"semitones": 6,  "ring_freq": 0.0,  "ring_mix": 0.0, "emoji": "🐿️"},
    "Deep":     {"semitones": -5, "ring_freq": 0.0,  "ring_mix": 0.0, "emoji": "🎩"},
    "Demon":    {"semitones": -7, "ring_freq": 35.0, "ring_mix": 0.45, "emoji": "😈"},
}


class PhaseVocoderPitchShifter:
    """Streaming pitch shifter (classic STFT phase-vocoder approach).

    Feed it arbitrary-length blocks; it returns blocks of the same length
    with ~frame_size samples of latency. Based on the analysis/synthesis
    scheme from S. Bernsee's smbPitchShift, vectorized with NumPy.
    """

    def __init__(self, frame_size=1024, oversampling=4, samplerate=SAMPLE_RATE):
        self.n = frame_size
        self.osamp = oversampling
        self.hop = frame_size // oversampling
        self.sr = samplerate
        self.window = np.hanning(self.n).astype(np.float64)
        self.bins = self.n // 2 + 1
        self.freq_per_bin = samplerate / self.n
        # expected phase advance per hop for each bin
        self.expected = 2.0 * np.pi * self.hop / self.n * np.arange(self.bins)
        # hann^2 overlap-add gain at 75% overlap
        self.ola_gain = 0.375 * oversampling
        self.reset()

    def reset(self):
        self.in_ring = np.zeros(self.n)
        self.out_acc = np.zeros(self.n)
        self.last_phase = np.zeros(self.bins)
        self.sum_phase = np.zeros(self.bins)
        self.in_buf = np.zeros(0)
        self.out_buf = np.zeros(self.n)  # initial latency
        self.env_in = 0.0
        self.env_out = 0.0

    def _process_frame(self, ratio):
        spec = np.fft.rfft(self.in_ring * self.window)
        mag = np.abs(spec)
        phase = np.angle(spec)

        # true frequency of each bin from the phase difference
        delta = phase - self.last_phase - self.expected
        self.last_phase = phase
        delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
        true_freq = (np.arange(self.bins) + delta * self.osamp / (2.0 * np.pi)) * self.freq_per_bin

        # move each bin to its shifted position, splitting the magnitude
        # between the two nearest target bins (plain integer mapping leaves
        # gaps in the spectrum and the level collapses at some ratios)
        pos = np.arange(self.bins) * ratio
        lo = pos.astype(np.int64)
        frac = pos - lo
        new_mag = np.zeros(self.bins)
        freq_sum = np.zeros(self.bins)
        v = lo < self.bins
        np.add.at(new_mag, lo[v], mag[v] * (1.0 - frac[v]))
        np.add.at(freq_sum, lo[v], mag[v] * (1.0 - frac[v]) * true_freq[v] * ratio)
        v = lo + 1 < self.bins
        np.add.at(new_mag, lo[v] + 1, mag[v] * frac[v])
        np.add.at(freq_sum, lo[v] + 1, mag[v] * frac[v] * true_freq[v] * ratio)
        new_freq = np.zeros(self.bins)
        nz = new_mag > 1e-12
        new_freq[nz] = freq_sum[nz] / new_mag[nz]

        # rebuild phases and synthesize
        self.sum_phase += 2.0 * np.pi * new_freq / self.freq_per_bin / self.osamp
        frame = np.fft.irfft(new_mag * np.exp(1j * self.sum_phase))
        frame *= self.window / self.ola_gain

        self.out_acc += frame
        out = self.out_acc[: self.hop].copy()
        self.out_acc[: -self.hop] = self.out_acc[self.hop:]
        self.out_acc[-self.hop:] = 0.0
        return out

    def process(self, block, semitones):
        """Process one audio block (1-D float array), returns same length."""
        if abs(semitones) < 1e-3:
            # bypass, but keep the same latency so switching is seamless
            self.in_buf = np.concatenate([self.in_buf, block])
            while len(self.in_buf) >= self.hop:
                chunk, self.in_buf = self.in_buf[: self.hop], self.in_buf[self.hop:]
                self.out_buf = np.concatenate([self.out_buf, chunk])
        else:
            ratio = 2.0 ** (semitones / 12.0)
            self.in_buf = np.concatenate([self.in_buf, block])
            while len(self.in_buf) >= self.hop:
                chunk, self.in_buf = self.in_buf[: self.hop], self.in_buf[self.hop:]
                self.in_ring[: -self.hop] = self.in_ring[self.hop:]
                self.in_ring[-self.hop:] = chunk
                self.out_buf = np.concatenate([self.out_buf, self._process_frame(ratio)])

        out, self.out_buf = self.out_buf[: len(block)], self.out_buf[len(block):]
        if len(out) < len(block):  # startup: pad with silence
            out = np.pad(out, (len(block) - len(out), 0))

        if abs(semitones) >= 1e-3:
            # overlap-added frames partially cancel depending on the ratio;
            # match the output level to the input with a slow envelope follower
            self.env_in = 0.8 * self.env_in + 0.2 * float(np.sqrt(np.mean(block ** 2)))
            self.env_out = 0.8 * self.env_out + 0.2 * float(np.sqrt(np.mean(out ** 2)))
            if self.env_out > 1e-5 and self.env_in > 1e-5:
                out = out * np.clip(self.env_in / self.env_out, 0.25, 4.0)
        return out


class RingModulator:
    """Multiplies the signal with a sine carrier — the classic robot voice."""

    def __init__(self, samplerate=SAMPLE_RATE):
        self.sr = samplerate
        self.phase = 0.0

    def process(self, block, freq, mix):
        if mix <= 0.0 or freq <= 0.0:
            return block
        step = 2.0 * np.pi * freq / self.sr
        phases = self.phase + step * np.arange(1, len(block) + 1)
        self.phase = float(phases[-1] % (2.0 * np.pi))
        return block * (1.0 - mix) + block * np.sin(phases) * mix


class Engine:
    """Owns the audio stream and DSP state shared with the GUI."""

    def __init__(self):
        self.shifter = PhaseVocoderPitchShifter()
        self.ring = RingModulator()
        self.semitones = 0.0
        self.ring_freq = 0.0
        self.ring_mix = 0.0
        self.gain = 1.0
        self.level_in = 0.0
        self.level_out = 0.0
        self.stream = None
        self.lock = threading.Lock()

    def callback(self, indata, outdata, frames, time_info, status):
        x = indata[:, 0].astype(np.float64)
        self.level_in = float(np.sqrt(np.mean(x * x)))
        with self.lock:
            y = self.shifter.process(x, self.semitones)
            y = self.ring.process(y, self.ring_freq, self.ring_mix)
        y = np.clip(y * self.gain, -1.0, 1.0)
        self.level_out = float(np.sqrt(np.mean(y * y)))
        outdata[:, 0] = y.astype(np.float32)

    def start(self, input_device, output_device):
        import sounddevice as sd

        self.stop()
        self.shifter.reset()
        self.stream = sd.Stream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            device=(input_device, output_device),
            channels=1,
            dtype="float32",
            callback=self.callback,
        )
        self.stream.start()

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.level_in = self.level_out = 0.0

    def apply_preset(self, name):
        p = PRESETS[name]
        with self.lock:
            self.semitones = float(p["semitones"])
            self.ring_freq = p["ring_freq"]
            self.ring_mix = p["ring_mix"]


def main():
    import sounddevice as sd
    import customtkinter as ctk

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    engine = Engine()

    app = ctk.CTk()
    app.title("VoxShift")
    app.geometry("560x640")
    app.resizable(False, False)

    ctk.CTkLabel(app, text="VoxShift", font=ctk.CTkFont(size=28, weight="bold")).pack(pady=(18, 0))
    ctk.CTkLabel(app, text="real-time voice changer", text_color="gray60").pack()

    # --- devices -----------------------------------------------------------
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()

    def device_label(i, d):
        return f"[{i}] {d['name']} ({hostapis[d['hostapi']]['name']})"

    inputs = {device_label(i, d): i for i, d in enumerate(devices) if d["max_input_channels"] > 0}
    outputs = {device_label(i, d): i for i, d in enumerate(devices) if d["max_output_channels"] > 0}

    dev_frame = ctk.CTkFrame(app)
    dev_frame.pack(fill="x", padx=20, pady=(16, 6))
    ctk.CTkLabel(dev_frame, text="Microphone").grid(row=0, column=0, sticky="w", padx=12, pady=(10, 0))
    in_box = ctk.CTkComboBox(dev_frame, values=list(inputs), width=380)
    in_box.grid(row=1, column=0, padx=12, pady=(2, 8))
    ctk.CTkLabel(dev_frame, text="Output (pick CABLE Input to use as virtual mic)").grid(
        row=2, column=0, sticky="w", padx=12)
    out_box = ctk.CTkComboBox(dev_frame, values=list(outputs), width=380)
    out_box.grid(row=3, column=0, padx=12, pady=(2, 12))

    try:
        default_in, default_out = sd.default.device
        for label, i in inputs.items():
            if i == default_in:
                in_box.set(label)
        cable = next((l for l in outputs if "CABLE Input" in l), None)
        if cable:
            out_box.set(cable)
        else:
            for label, i in outputs.items():
                if i == default_out:
                    out_box.set(label)
    except Exception:
        pass

    # --- presets -----------------------------------------------------------
    preset_frame = ctk.CTkFrame(app)
    preset_frame.pack(fill="x", padx=20, pady=6)
    ctk.CTkLabel(preset_frame, text="Preset").pack(anchor="w", padx=12, pady=(10, 2))
    row = ctk.CTkFrame(preset_frame, fg_color="transparent")
    row.pack(pady=(0, 12))

    pitch_var = ctk.DoubleVar(value=0.0)
    pitch_label = None  # set below

    def select_preset(name):
        engine.apply_preset(name)
        pitch_var.set(PRESETS[name]["semitones"])
        update_pitch_label()

    for name, p in PRESETS.items():
        ctk.CTkButton(row, text=f"{p['emoji']}\n{name}", width=92, height=56,
                      command=lambda n=name: select_preset(n)).pack(side="left", padx=4)

    # --- sliders -----------------------------------------------------------
    slider_frame = ctk.CTkFrame(app)
    slider_frame.pack(fill="x", padx=20, pady=6)

    pitch_label = ctk.CTkLabel(slider_frame, text="Pitch: 0 st")
    pitch_label.grid(row=0, column=0, sticky="w", padx=12, pady=(10, 0))

    def update_pitch_label(*_):
        st = round(pitch_var.get())
        pitch_label.configure(text=f"Pitch: {st:+d} st")
        with engine.lock:
            engine.semitones = float(st)

    ctk.CTkSlider(slider_frame, from_=-12, to=12, number_of_steps=24, variable=pitch_var,
                  command=update_pitch_label, width=480).grid(row=1, column=0, padx=12, pady=(2, 8))

    gain_label = ctk.CTkLabel(slider_frame, text="Volume: 100%")
    gain_label.grid(row=2, column=0, sticky="w", padx=12)
    gain_var = ctk.DoubleVar(value=1.0)

    def update_gain(*_):
        engine.gain = gain_var.get()
        gain_label.configure(text=f"Volume: {engine.gain * 100:.0f}%")

    ctk.CTkSlider(slider_frame, from_=0.0, to=2.0, variable=gain_var,
                  command=update_gain, width=480).grid(row=3, column=0, padx=12, pady=(2, 12))

    # --- meters + start ----------------------------------------------------
    meter_frame = ctk.CTkFrame(app)
    meter_frame.pack(fill="x", padx=20, pady=6)
    ctk.CTkLabel(meter_frame, text="In").grid(row=0, column=0, padx=(12, 6), pady=(12, 4))
    in_meter = ctk.CTkProgressBar(meter_frame, width=440)
    in_meter.grid(row=0, column=1, pady=(12, 4))
    ctk.CTkLabel(meter_frame, text="Out").grid(row=1, column=0, padx=(12, 6), pady=(0, 12))
    out_meter = ctk.CTkProgressBar(meter_frame, width=440)
    out_meter.grid(row=1, column=1, pady=(0, 12))
    in_meter.set(0)
    out_meter.set(0)

    status = ctk.CTkLabel(app, text="stopped", text_color="gray60")

    def toggle():
        if engine.stream is None:
            try:
                engine.start(inputs[in_box.get()], outputs[out_box.get()])
            except Exception as exc:
                status.configure(text=f"error: {exc}", text_color="#e05f5f")
                return
            start_btn.configure(text="■  Stop", fg_color="#a33c3c", hover_color="#8a3232")
            latency_ms = 1000 * (engine.shifter.n + BLOCK_SIZE) / SAMPLE_RATE
            status.configure(text=f"running · ~{latency_ms:.0f} ms latency", text_color="gray60")
        else:
            engine.stop()
            start_btn.configure(text="▶  Start", fg_color=["#3B8ED0", "#1F6AA5"],
                                hover_color=["#36719F", "#144870"])
            status.configure(text="stopped", text_color="gray60")

    start_btn = ctk.CTkButton(app, text="▶  Start", height=44,
                              font=ctk.CTkFont(size=16, weight="bold"), command=toggle)
    start_btn.pack(fill="x", padx=20, pady=(10, 4))
    status.pack(pady=(0, 10))

    def poll_meters():
        in_meter.set(min(1.0, engine.level_in * 4))
        out_meter.set(min(1.0, engine.level_out * 4))
        app.after(50, poll_meters)

    poll_meters()

    def on_close():
        engine.stop()
        app.destroy()

    app.protocol("WM_DELETE_WINDOW", on_close)
    app.mainloop()


if __name__ == "__main__":
    main()
