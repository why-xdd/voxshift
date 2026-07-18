"""VoxShift — real-time voice changer with a full effect chain.

Captures the microphone, runs it through a chain of effects (noise gate,
pitch + independent formant shift, EQ, distortion, bitcrusher, ring mod,
tremolo, vibrato, echo, reverb) and plays the result to any output device.

To use the changed voice as a microphone in Discord / games you need a
virtual audio cable (e.g. VB-Cable): set VoxShift's output to the cable and
pick the cable's output as the mic in the target app. Without a cable you
can still monitor through your headphones and record to a .wav file.

Run:  python voxshift.py
"""

import datetime
import json
import os
import threading
import wave
from collections import deque

import numpy as np

import ai_engine
import driver
import dsp

SAMPLE_RATE = dsp.SAMPLE_RATE
BLOCK_SIZE = 512

PRESET_FILE = os.path.join(os.path.expanduser("~"), ".voxshift_presets.json")
REC_DIR = os.path.join(os.path.expanduser("~"), "VoxShift Recordings")

# every parameter the engine exposes, with defaults (a "clean" voice)
DEFAULTS = {
    "gate_threshold": 0.0,
    "pitch": 0.0, "formant": 0.0, "preserve": True,
    "eq_enabled": False, "eq_low": 0.0, "eq_mid": 0.0, "eq_high": 0.0,
    "dist_enabled": False, "dist_drive": 8.0, "dist_mix": 0.6,
    "crush_enabled": False, "crush_bits": 8, "crush_downsample": 1, "crush_mix": 0.7,
    "ring_enabled": False, "ring_freq": 70.0, "ring_mix": 0.0,
    "trem_enabled": False, "trem_rate": 5.0, "trem_depth": 0.5,
    "vib_enabled": False, "vib_rate": 5.0, "vib_depth_ms": 2.0,
    "echo_enabled": False, "echo_time_ms": 250.0, "echo_feedback": 0.35, "echo_mix": 0.0,
    "reverb_enabled": False, "reverb_size": 0.5, "reverb_damp": 0.4, "reverb_mix": 0.0,
    "gain": 1.0,
}

CATEGORIES = ["Human", "Cartoon", "Sci-Fi", "Horror", "Famous", "FX", "Custom"]

# built-in presets — each overrides a subset of DEFAULTS. "cat" groups them in the UI.
PRESETS = {
    # ---- Human --------------------------------------------------------------
    "Normal":    {"emoji": "🎤", "cat": "Human", "params": {}},
    "Man":       {"emoji": "👨", "cat": "Human", "params": {"pitch": -4, "formant": -3, "eq_enabled": True, "eq_low": 4, "eq_high": -1}},
    "Woman":     {"emoji": "👩", "cat": "Human", "params": {"pitch": 7, "formant": 3, "eq_enabled": True, "eq_low": -4, "eq_mid": 1, "eq_high": 4}},
    "Kid":       {"emoji": "🧒", "cat": "Human", "params": {"pitch": 10, "formant": 5, "eq_enabled": True, "eq_low": -5, "eq_high": 4}},
    "Baby":      {"emoji": "👶", "cat": "Human", "params": {"pitch": 12, "formant": 8, "eq_enabled": True, "eq_low": -7, "eq_high": 5}},
    "Deep":      {"emoji": "🎩", "cat": "Human", "params": {"pitch": -5, "formant": -2}},
    "Old Man":   {"emoji": "👴", "cat": "Human", "params": {"pitch": -2, "formant": -1, "trem_enabled": True, "trem_rate": 5.5, "trem_depth": 0.16,
                                                            "dist_enabled": True, "dist_drive": 3, "dist_mix": 0.15}},
    "Announcer": {"emoji": "🎙️", "cat": "Human", "params": {"pitch": -2, "formant": -1, "eq_enabled": True, "eq_low": 4, "eq_mid": 1, "eq_high": 3,
                                                            "reverb_enabled": True, "reverb_size": 0.3, "reverb_mix": 0.12}},
    # ---- Cartoon ------------------------------------------------------------
    "Chipmunk":  {"emoji": "🐿️", "cat": "Cartoon", "params": {"pitch": 9, "preserve": False}},
    "Minion":    {"emoji": "🍌", "cat": "Cartoon", "params": {"pitch": 9, "formant": 7, "eq_enabled": True, "eq_high": 3,
                                                             "trem_enabled": True, "trem_rate": 8, "trem_depth": 0.18}},
    "Mouse":     {"emoji": "🐭", "cat": "Cartoon", "params": {"pitch": 17, "formant": 7, "eq_enabled": True, "eq_low": -8, "eq_high": 5}},
    "Helium":    {"emoji": "🎈", "cat": "Cartoon", "params": {"pitch": 4, "formant": 12, "eq_enabled": True, "eq_high": 3}},
    "Duck":      {"emoji": "🦆", "cat": "Cartoon", "params": {"pitch": 3, "formant": 5, "ring_enabled": True, "ring_freq": 220, "ring_mix": 0.32,
                                                             "eq_enabled": True, "eq_mid": 5, "dist_enabled": True, "dist_drive": 4, "dist_mix": 0.3}},
    "Parrot":    {"emoji": "🦜", "cat": "Cartoon", "params": {"pitch": 8, "formant": 5, "trem_enabled": True, "trem_rate": 7, "trem_depth": 0.22}},
    "Blue Elf":  {"emoji": "🔵", "cat": "Cartoon", "params": {"pitch": 7, "formant": 5}},
    "Sponge":    {"emoji": "🧽", "cat": "Cartoon", "params": {"pitch": 6, "formant": 6, "eq_enabled": True, "eq_mid": 3,
                                                             "trem_enabled": True, "trem_rate": 6, "trem_depth": 0.12}},
    "Troll":     {"emoji": "👹", "cat": "Cartoon", "params": {"pitch": -6, "formant": -5, "dist_enabled": True, "dist_drive": 5, "dist_mix": 0.3,
                                                             "eq_enabled": True, "eq_low": 4}},
    # ---- Sci-Fi -------------------------------------------------------------
    "Robot":     {"emoji": "🤖", "cat": "Sci-Fi", "params": {"ring_enabled": True, "ring_freq": 70, "ring_mix": 0.9,
                                                            "crush_enabled": True, "crush_bits": 6, "crush_downsample": 2, "crush_mix": 0.4}},
    "Cyborg":    {"emoji": "🦾", "cat": "Sci-Fi", "params": {"pitch": -2, "ring_enabled": True, "ring_freq": 110, "ring_mix": 0.5,
                                                            "dist_enabled": True, "dist_drive": 6, "dist_mix": 0.4}},
    "AI Core":   {"emoji": "🧠", "cat": "Sci-Fi", "params": {"ring_enabled": True, "ring_freq": 90, "ring_mix": 0.22,
                                                            "reverb_enabled": True, "reverb_size": 0.2, "reverb_mix": 0.1, "eq_enabled": True, "eq_high": 2}},
    "Alien":     {"emoji": "👽", "cat": "Sci-Fi", "params": {"pitch": 3, "formant": 5, "ring_enabled": True, "ring_freq": 130, "ring_mix": 0.4,
                                                            "vib_enabled": True, "vib_rate": 6, "vib_depth_ms": 2}},
    "Dalek":     {"emoji": "🛸", "cat": "Sci-Fi", "params": {"pitch": -2, "ring_enabled": True, "ring_freq": 30, "ring_mix": 0.85,
                                                            "dist_enabled": True, "dist_drive": 5, "dist_mix": 0.4}},
    "Drone":     {"emoji": "🛰️", "cat": "Sci-Fi", "params": {"crush_enabled": True, "crush_bits": 5, "crush_downsample": 3, "crush_mix": 0.6,
                                                            "ring_enabled": True, "ring_freq": 60, "ring_mix": 0.4}},
    "Glitch":    {"emoji": "📺", "cat": "Sci-Fi", "params": {"crush_enabled": True, "crush_bits": 4, "crush_downsample": 4, "crush_mix": 0.7,
                                                            "dist_enabled": True, "dist_drive": 6, "dist_mix": 0.4, "ring_enabled": True, "ring_freq": 200, "ring_mix": 0.3}},
    "Comms":     {"emoji": "📡", "cat": "Sci-Fi", "params": {"eq_enabled": True, "eq_low": -14, "eq_mid": 4, "eq_high": -8,
                                                            "crush_enabled": True, "crush_bits": 7, "crush_downsample": 2, "crush_mix": 0.4,
                                                            "dist_enabled": True, "dist_drive": 4, "dist_mix": 0.3}},
    # ---- Horror -------------------------------------------------------------
    "Demon":     {"emoji": "😈", "cat": "Horror", "params": {"pitch": -7, "formant": -4, "ring_enabled": True, "ring_freq": 35, "ring_mix": 0.4,
                                                            "dist_enabled": True, "dist_drive": 5, "dist_mix": 0.3, "reverb_enabled": True, "reverb_mix": 0.25}},
    "Ghost":     {"emoji": "👻", "cat": "Horror", "params": {"pitch": -3, "reverb_enabled": True, "reverb_size": 0.85, "reverb_mix": 0.5,
                                                            "echo_enabled": True, "echo_time_ms": 300, "echo_feedback": 0.4, "echo_mix": 0.35,
                                                            "vib_enabled": True, "vib_rate": 4, "vib_depth_ms": 3}},
    "Giant":     {"emoji": "🗿", "cat": "Horror", "params": {"pitch": -8, "formant": -6, "reverb_enabled": True, "reverb_size": 0.7, "reverb_mix": 0.3}},
    "Goblin":    {"emoji": "👺", "cat": "Horror", "params": {"pitch": -3, "formant": -4, "dist_enabled": True, "dist_drive": 5, "dist_mix": 0.35,
                                                            "ring_enabled": True, "ring_freq": 160, "ring_mix": 0.25}},
    "Vampire":   {"emoji": "🧛", "cat": "Horror", "params": {"pitch": -3, "formant": -2, "reverb_enabled": True, "reverb_size": 0.5, "reverb_mix": 0.2}},
    "Zombie":    {"emoji": "🧟", "cat": "Horror", "params": {"pitch": -4, "formant": -3, "dist_enabled": True, "dist_drive": 4, "dist_mix": 0.3,
                                                            "trem_enabled": True, "trem_rate": 3, "trem_depth": 0.3}},
    "Monster":   {"emoji": "👹", "cat": "Horror", "params": {"pitch": -9, "formant": -5, "dist_enabled": True, "dist_drive": 6, "dist_mix": 0.4,
                                                            "ring_enabled": True, "ring_freq": 45, "ring_mix": 0.3}},
    "Skeleton":  {"emoji": "💀", "cat": "Horror", "params": {"pitch": -1, "crush_enabled": True, "crush_bits": 6, "crush_downsample": 2, "crush_mix": 0.5,
                                                            "dist_enabled": True, "dist_drive": 3, "dist_mix": 0.2}},
    # ---- Famous (archetypes) ------------------------------------------------
    "Dark Lord": {"emoji": "🖤", "cat": "Famous", "params": {"pitch": -4, "formant": -3, "dist_enabled": True, "dist_drive": 3, "dist_mix": 0.2,
                                                            "reverb_enabled": True, "reverb_size": 0.4, "reverb_mix": 0.18, "eq_enabled": True, "eq_low": 3}},
    "Green Ogre":{"emoji": "🟢", "cat": "Famous", "params": {"pitch": -3, "formant": -2, "eq_enabled": True, "eq_low": 3}},
    "Wizard":    {"emoji": "🧙", "cat": "Famous", "params": {"pitch": -2, "reverb_enabled": True, "reverb_size": 0.5, "reverb_mix": 0.25,
                                                            "echo_enabled": True, "echo_time_ms": 250, "echo_feedback": 0.3, "echo_mix": 0.2}},
    "Pirate":    {"emoji": "🏴", "cat": "Famous", "params": {"pitch": -3, "formant": -2, "dist_enabled": True, "dist_drive": 3, "dist_mix": 0.2}},
    "Clown":     {"emoji": "🤡", "cat": "Famous", "params": {"pitch": 5, "formant": 4, "trem_enabled": True, "trem_rate": 6, "trem_depth": 0.2}},
    "Santa":     {"emoji": "🎅", "cat": "Famous", "params": {"pitch": -3, "formant": -2, "reverb_enabled": True, "reverb_size": 0.35, "reverb_mix": 0.15}},
    "Cowboy":    {"emoji": "🤠", "cat": "Famous", "params": {"pitch": -2, "eq_enabled": True, "eq_low": 2}},
    # ---- FX / places --------------------------------------------------------
    "Telephone": {"emoji": "☎️", "cat": "FX", "params": {"eq_enabled": True, "eq_low": -14, "eq_mid": 4, "eq_high": -10,
                                                         "crush_enabled": True, "crush_bits": 7, "crush_downsample": 2, "crush_mix": 0.5,
                                                         "dist_enabled": True, "dist_drive": 4, "dist_mix": 0.25}},
    "Radio":     {"emoji": "📻", "cat": "FX", "params": {"eq_enabled": True, "eq_low": -6, "eq_mid": 5, "eq_high": -3,
                                                         "dist_enabled": True, "dist_drive": 6, "dist_mix": 0.4}},
    "Cave":      {"emoji": "🕳️", "cat": "FX", "params": {"pitch": -2, "reverb_enabled": True, "reverb_size": 0.9, "reverb_damp": 0.6, "reverb_mix": 0.55,
                                                         "echo_enabled": True, "echo_time_ms": 220, "echo_feedback": 0.45, "echo_mix": 0.4}},
    "Hall":      {"emoji": "🏛️", "cat": "FX", "params": {"reverb_enabled": True, "reverb_size": 0.75, "reverb_damp": 0.3, "reverb_mix": 0.4}},
    "Underwater":{"emoji": "🌊", "cat": "FX", "params": {"eq_enabled": True, "eq_low": 4, "eq_high": -14, "vib_enabled": True, "vib_rate": 3, "vib_depth_ms": 4,
                                                         "reverb_enabled": True, "reverb_size": 0.5, "reverb_mix": 0.3}},
    "Megaphone": {"emoji": "📢", "cat": "FX", "params": {"eq_enabled": True, "eq_low": -10, "eq_mid": 6, "eq_high": -6,
                                                         "dist_enabled": True, "dist_drive": 7, "dist_mix": 0.5}},
}

BUILTIN_NAMES = set(PRESETS)


def load_user_presets():
    try:
        with open(PRESET_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for name, p in data.items():
            p.setdefault("emoji", "⭐")
            p.setdefault("params", {})
            p["cat"] = "Custom"
            PRESETS[name] = p
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass


def save_user_presets():
    custom = {n: p for n, p in PRESETS.items() if n not in BUILTIN_NAMES}
    try:
        with open(PRESET_FILE, "w", encoding="utf-8") as f:
            json.dump(custom, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


class Engine:
    """Owns the audio stream, the effect chain and shared DSP state."""

    def __init__(self):
        self.gate = dsp.NoiseGate()
        self.shifter = dsp.PitchFormantShifter()
        self.eq = dsp.EQ3()
        self.dist = dsp.Distortion()
        self.crush = dsp.Bitcrusher()
        self.ring = dsp.RingModulator()
        self.trem = dsp.Tremolo()
        self.vib = dsp.Vibrato()
        self.echo = dsp.Echo()
        self.reverb = dsp.Reverb()
        # order matters: clean -> pitch -> tone -> modulation -> space
        self.chain = [self.gate, self.shifter, self.eq, self.dist, self.crush,
                      self.ring, self.trem, self.vib, self.echo, self.reverb]

        # optional neural voice-conversion stage (pass-through until a model
        # is installed — see ai_engine.py). Runs right after the noise gate.
        self.ai = ai_engine.NullConverter()

        self.gain = 1.0
        self.muted = False
        self.level_in = 0.0
        self.level_out = 0.0
        self.stream = None
        self.lock = threading.Lock()

        # separate "hear yourself" monitor stream (to your headphones) fed from
        # the main callback through a small queue
        self.monitor_stream = None
        self.mon_q = deque(maxlen=32)

        self.recording = False
        self._rec_frames = []

    # -- parameter snapshot / apply (used by presets) -----------------------
    def snapshot(self):
        s = self.shifter
        return {
            "gate_threshold": self.gate.threshold,
            "pitch": s.pitch, "formant": s.formant, "preserve": s.preserve,
            "eq_enabled": self.eq.enabled, "eq_low": self.eq.low, "eq_mid": self.eq.mid, "eq_high": self.eq.high,
            "dist_enabled": self.dist.enabled, "dist_drive": self.dist.drive, "dist_mix": self.dist.mix,
            "crush_enabled": self.crush.enabled, "crush_bits": self.crush.bits,
            "crush_downsample": self.crush.downsample, "crush_mix": self.crush.mix,
            "ring_enabled": self.ring.enabled, "ring_freq": self.ring.freq, "ring_mix": self.ring.mix,
            "trem_enabled": self.trem.enabled, "trem_rate": self.trem.rate, "trem_depth": self.trem.depth,
            "vib_enabled": self.vib.enabled, "vib_rate": self.vib.rate, "vib_depth_ms": self.vib.depth_ms,
            "echo_enabled": self.echo.enabled, "echo_time_ms": self.echo.time_ms,
            "echo_feedback": self.echo.feedback, "echo_mix": self.echo.mix,
            "reverb_enabled": self.reverb.enabled, "reverb_size": self.reverb.size,
            "reverb_damp": self.reverb.damp, "reverb_mix": self.reverb.mix,
            "gain": self.gain,
        }

    def apply(self, params):
        p = dict(DEFAULTS)
        p.update(params)
        with self.lock:
            self.gate.threshold = p["gate_threshold"]
            self.shifter.pitch = float(p["pitch"])
            self.shifter.formant = float(p["formant"])
            self.shifter.preserve = bool(p["preserve"])
            self.eq.enabled = p["eq_enabled"]; self.eq.low = p["eq_low"]; self.eq.mid = p["eq_mid"]; self.eq.high = p["eq_high"]
            self.dist.enabled = p["dist_enabled"]; self.dist.drive = p["dist_drive"]; self.dist.mix = p["dist_mix"]
            self.crush.enabled = p["crush_enabled"]; self.crush.bits = int(p["crush_bits"])
            self.crush.downsample = int(p["crush_downsample"]); self.crush.mix = p["crush_mix"]
            self.ring.enabled = p["ring_enabled"]; self.ring.freq = p["ring_freq"]; self.ring.mix = p["ring_mix"]
            self.trem.enabled = p["trem_enabled"]; self.trem.rate = p["trem_rate"]; self.trem.depth = p["trem_depth"]
            self.vib.enabled = p["vib_enabled"]; self.vib.rate = p["vib_rate"]; self.vib.depth_ms = p["vib_depth_ms"]
            self.echo.enabled = p["echo_enabled"]; self.echo.time_ms = p["echo_time_ms"]
            self.echo.feedback = p["echo_feedback"]; self.echo.mix = p["echo_mix"]
            self.reverb.enabled = p["reverb_enabled"]; self.reverb.size = p["reverb_size"]
            self.reverb.damp = p["reverb_damp"]; self.reverb.mix = p["reverb_mix"]
            self.gain = p["gain"]

    # -- audio --------------------------------------------------------------
    def callback(self, indata, outdata, frames, time_info, status):
        x = indata[:, 0].astype(np.float64)
        self.level_in = float(np.sqrt(np.mean(x * x)))
        with self.lock:
            x = self.gate.process(x)
            x = self.ai.process(x, SAMPLE_RATE)
            for eff in self.chain[1:]:  # gate already applied above
                x = eff.process(x)
            g = 0.0 if self.muted else self.gain
        y = np.clip(x * g, -1.0, 1.0)
        self.level_out = float(np.sqrt(np.mean(y * y)))
        if self.recording:
            self._rec_frames.append(y.copy())
        yf = y.astype(np.float32)
        if self.monitor_stream is not None:
            self.mon_q.append(yf.copy())
        outdata[:, 0] = yf

    def monitor_callback(self, outdata, frames, time_info, status):
        try:
            blk = self.mon_q.popleft()
        except IndexError:
            outdata.fill(0.0)
            return
        n = min(len(blk), frames)
        outdata[:n, 0] = blk[:n]
        if n < frames:
            outdata[n:, 0] = 0.0

    def start_monitor(self, device):
        import sounddevice as sd
        self.stop_monitor()
        self.mon_q.clear()
        self.monitor_stream = sd.OutputStream(
            samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, device=device,
            channels=1, dtype="float32", callback=self.monitor_callback)
        self.monitor_stream.start()

    def stop_monitor(self):
        if self.monitor_stream is not None:
            self.monitor_stream.stop()
            self.monitor_stream.close()
            self.monitor_stream = None
        self.mon_q.clear()

    def start(self, input_device, output_device):
        import sounddevice as sd
        self.stop()
        for eff in self.chain:
            eff.reset()
        self.stream = sd.Stream(
            samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE,
            device=(input_device, output_device), channels=1,
            dtype="float32", callback=self.callback,
        )
        self.stream.start()

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.stop_monitor()
        self.level_in = self.level_out = 0.0

    def start_recording(self):
        self._rec_frames = []
        self.recording = True

    def stop_recording(self):
        self.recording = False
        if not self._rec_frames:
            return None
        audio = np.concatenate(self._rec_frames)
        self._rec_frames = []
        os.makedirs(REC_DIR, exist_ok=True)
        fname = os.path.join(REC_DIR, "voice_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".wav")
        data = np.clip(audio, -1.0, 1.0)
        pcm = (data * 32767).astype("<i2")
        with wave.open(fname, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm.tobytes())
        return fname


def main():
    import sounddevice as sd
    import customtkinter as ctk

    load_user_presets()

    try:
        import keyboard
        HAVE_KEYBOARD = True
    except Exception:
        keyboard = None
        HAVE_KEYBOARD = False

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    engine = Engine()
    engine.apply(PRESETS["Normal"]["params"])

    app = ctk.CTk()
    app.title("VoxShift")
    app.geometry("620x860")
    app.minsize(620, 700)

    ctk.CTkLabel(app, text="VoxShift", font=ctk.CTkFont(size=26, weight="bold")).pack(pady=(12, 0))
    ctk.CTkLabel(app, text="real-time voice changer", text_color="gray60").pack()

    # --- devices -----------------------------------------------------------
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()

    def device_label(i, d):
        return f"[{i}] {d['name']} ({hostapis[d['hostapi']]['name']})"

    inputs = {device_label(i, d): i for i, d in enumerate(devices) if d["max_input_channels"] > 0}
    outputs = {device_label(i, d): i for i, d in enumerate(devices) if d["max_output_channels"] > 0}

    dev_frame = ctk.CTkFrame(app)
    dev_frame.pack(fill="x", padx=16, pady=(10, 6))
    ctk.CTkLabel(dev_frame, text="Microphone (input)").grid(row=0, column=0, sticky="w", padx=12, pady=(8, 0))
    in_box = ctk.CTkComboBox(dev_frame, values=list(inputs), width=560)
    in_box.grid(row=1, column=0, padx=12, pady=(2, 6))
    ctk.CTkLabel(dev_frame, text="Output — your headphones to monitor, or a virtual cable to use as a mic").grid(
        row=2, column=0, sticky="w", padx=12)
    out_box = ctk.CTkComboBox(dev_frame, values=list(outputs), width=560)
    out_box.grid(row=3, column=0, padx=12, pady=(2, 4))

    VIRTUAL_HINTS = ("CABLE Input", "VoiceMeeter Input", "VoiceMeeter Aux Input", "VB-Audio")
    virtual_out = next((l for l in outputs if any(h in l for h in VIRTUAL_HINTS)), None)
    vmsg = ("virtual cable detected — pick its 'Output' side as the mic in Discord"
            if virtual_out else
            "no virtual cable — output goes to your speakers/headphones (install VB-Cable to use in Discord)")
    cable_lbl = ctk.CTkLabel(dev_frame, text=vmsg, text_color=("#4a9d5b" if virtual_out else "#c98a3a"),
                             font=ctk.CTkFont(size=11))
    cable_lbl.grid(row=4, column=0, sticky="w", padx=12, pady=(0, 2))

    if not virtual_out:
        def setup_driver():
            drv_btn.configure(state="disabled", text="installing…")

            def report(msg, color="#c9a33a"):
                app.after(0, lambda: cable_lbl.configure(text=msg, text_color=color))

            def work():
                try:
                    driver.install_cable(log=lambda m: report(m))
                    report("installer launched — click 'Install Driver', reboot, then restart VoxShift", "#4a9d5b")
                except Exception as exc:
                    report(f"install failed: {exc}", "#e05f5f")
                app.after(0, lambda: drv_btn.configure(state="normal", text="🎚 Set up virtual mic (VB-Cable)"))
            threading.Thread(target=work, daemon=True).start()

        drv_btn = ctk.CTkButton(dev_frame, text="🎚 Set up virtual mic (VB-Cable)", height=28,
                                command=setup_driver)
        drv_btn.grid(row=5, column=0, sticky="w", padx=12, pady=(0, 8))

    default_out = None
    try:
        default_in, default_out = sd.default.device
        for label, i in inputs.items():
            if i == default_in:
                in_box.set(label)
        if virtual_out:
            out_box.set(virtual_out)
        else:
            for label, i in outputs.items():
                if i == default_out:
                    out_box.set(label)
    except Exception:
        pass

    # --- monitor ("hear yourself") -----------------------------------------
    mon_frame = ctk.CTkFrame(app)
    mon_frame.pack(fill="x", padx=16, pady=(0, 6))
    monitor_var = ctk.BooleanVar(value=True)
    mon_top = ctk.CTkFrame(mon_frame, fg_color="transparent")
    mon_top.pack(fill="x", padx=12, pady=(8, 0))
    ctk.CTkSwitch(mon_top, text="🎧 Hear myself (monitor)", variable=monitor_var,
                  command=lambda: update_monitor()).pack(side="left")
    ctk.CTkLabel(mon_frame, text="…through the headphones/speakers you actually listen on:",
                 text_color="gray60", font=ctk.CTkFont(size=11)).pack(anchor="w", padx=12, pady=(2, 0))
    mon_box = ctk.CTkComboBox(mon_frame, values=list(outputs), width=560, command=lambda _v: update_monitor())
    mon_box.pack(padx=12, pady=(2, 10))
    if outputs:
        mon_box.set(next((l for l, i in outputs.items() if i == default_out), list(outputs)[0]))

    def update_monitor():
        # play the processed voice to a second device so you hear yourself even
        # when the main Output goes to a virtual cable (skip if same device)
        if engine.stream is None:
            return
        try:
            mdev = outputs[mon_box.get()]
        except KeyError:
            engine.stop_monitor()
            return
        same = mdev == outputs.get(out_box.get())
        if monitor_var.get() and not same:
            try:
                engine.start_monitor(mdev)
            except Exception as exc:
                status.configure(text=f"monitor error: {exc}", text_color="#e05f5f")
        else:
            engine.stop_monitor()

    # --- presets (categorised browser) -------------------------------------
    preset_frame = ctk.CTkFrame(app)
    preset_frame.pack(fill="x", padx=16, pady=6)

    current = {"preset": "Normal", "cat": "Human"}
    preset_buttons = {}
    cat_buttons = {}

    cat_row = ctk.CTkFrame(preset_frame, fg_color="transparent")
    cat_row.pack(fill="x", padx=10, pady=(8, 2))

    grid_holder = ctk.CTkScrollableFrame(preset_frame, height=110, fg_color="transparent")
    grid_holder.pack(fill="x", padx=8, pady=(0, 4))

    def presets_in_cat(cat):
        return [n for n, p in PRESETS.items() if p.get("cat", "Custom") == cat]

    def highlight_active():
        for n, b in preset_buttons.items():
            on = (n == current["preset"])
            b.configure(fg_color="#1F6AA5" if on else "#333333",
                        hover_color="#144870" if on else "#3d3d3d")

    def build_grid():
        for w in grid_holder.winfo_children():
            w.destroy()
        preset_buttons.clear()
        names = presets_in_cat(current["cat"])
        if not names:
            ctk.CTkLabel(grid_holder, text="(empty — dial in a voice and press 💾 Save)",
                         text_color="gray50").pack(pady=10)
        row_f = None
        for i, n in enumerate(names):
            if i % 4 == 0:
                row_f = ctk.CTkFrame(grid_holder, fg_color="transparent")
                row_f.pack(fill="x")
            p = PRESETS[n]
            b = ctk.CTkButton(row_f, text=f"{p['emoji']} {n}", width=132, height=30,
                              fg_color="#333333", hover_color="#3d3d3d",
                              command=lambda nm=n: select_preset(nm))
            b.pack(side="left", padx=3, pady=3)
            preset_buttons[n] = b
        highlight_active()

    def set_category(cat):
        current["cat"] = cat
        for c, btn in cat_buttons.items():
            btn.configure(fg_color="#1F6AA5" if c == cat else "transparent")
        build_grid()

    for c in CATEGORIES:
        b = ctk.CTkButton(cat_row, text=c, width=74, height=26, fg_color="transparent",
                          command=lambda cc=c: set_category(cc))
        b.pack(side="left", padx=2)
        cat_buttons[c] = b

    def select_preset(name):
        if name not in PRESETS:
            return
        engine.apply(PRESETS[name]["params"])
        current["preset"] = name
        cat = PRESETS[name].get("cat", "Custom")
        if cat != current["cat"]:
            set_category(cat)
        else:
            highlight_active()
        refresh_controls()

    srow = ctk.CTkFrame(preset_frame, fg_color="transparent")
    srow.pack(fill="x", padx=12, pady=(2, 8))
    name_entry = ctk.CTkEntry(srow, placeholder_text="name to save current settings", width=280)
    name_entry.pack(side="left")

    def save_current():
        name = name_entry.get().strip()
        if not name or name in BUILTIN_NAMES:
            return
        PRESETS[name] = {"emoji": "⭐", "cat": "Custom", "params": engine.snapshot()}
        save_user_presets()
        name_entry.delete(0, "end")
        current["preset"] = name
        set_category("Custom")
        register_hotkeys()

    def delete_current():
        name = current["preset"]
        if name in BUILTIN_NAMES or name not in PRESETS:
            return
        del PRESETS[name]
        save_user_presets()
        select_preset("Normal")
        build_grid()
        register_hotkeys()

    ctk.CTkButton(srow, text="💾 Save", width=76, command=save_current).pack(side="left", padx=6)
    ctk.CTkButton(srow, text="🗑 Delete", width=84, fg_color="#7a3a3a", hover_color="#8a3232",
                  command=delete_current).pack(side="left")

    # --- effect tabs -------------------------------------------------------
    tabs = ctk.CTkTabview(app, height=300)
    tabs.pack(fill="both", expand=True, padx=16, pady=6)
    for t in ("Voice", "Tone", "Modulation", "Space"):
        tabs.add(t)

    controls = []  # (getter, widget-updater) to refresh after preset change

    def add_slider(parent, text, frm, to, getter, setter, fmt="{:.0f}", steps=None, row=None, width=360):
        holder = ctk.CTkFrame(parent, fg_color="transparent")
        holder.pack(fill="x", padx=10, pady=(4, 0))
        lbl = ctk.CTkLabel(holder, text=text, width=150, anchor="w")
        lbl.pack(side="left")
        val = ctk.CTkLabel(holder, text="", width=60, anchor="e")
        val.pack(side="right")
        var = ctk.DoubleVar(value=getter())

        def on_change(_=None):
            v = var.get()
            setter(v)
            val.configure(text=fmt.format(v))
        sl = ctk.CTkSlider(holder, from_=frm, to=to, variable=var, command=on_change,
                           width=width, number_of_steps=steps)
        sl.pack(side="right", padx=8)

        def refresh():
            var.set(getter())
            val.configure(text=fmt.format(getter()))
        refresh()
        controls.append(refresh)
        return var

    def add_switch(parent, text, getter, setter):
        var = ctk.BooleanVar(value=getter())
        sw = ctk.CTkSwitch(parent, text=text, variable=var, command=lambda: setter(var.get()))
        sw.pack(anchor="w", padx=12, pady=(10, 0))

        def refresh():
            var.set(getter())
        controls.append(refresh)
        return var

    def sect(parent, title):
        ctk.CTkLabel(parent, text=title, font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=12, pady=(12, 0))

    lk = engine.lock

    # ---- Voice tab ----
    v = tabs.tab("Voice")
    add_slider(v, "Pitch (semitones)", -24, 24, lambda: engine.shifter.pitch,
               lambda x: setattr(engine.shifter, "pitch", round(x * 2) / 2), fmt="{:+.1f} st", steps=96)
    add_slider(v, "Formant (timbre)", -12, 12, lambda: engine.shifter.formant,
               lambda x: setattr(engine.shifter, "formant", round(x * 2) / 2), fmt="{:+.1f} st", steps=48)
    add_switch(v, "Preserve formants (natural timbre)", lambda: engine.shifter.preserve,
               lambda b: setattr(engine.shifter, "preserve", b))
    add_slider(v, "Noise gate", 0, 0.05, lambda: engine.gate.threshold,
               lambda x: setattr(engine.gate, "threshold", x), fmt="{:.3f}")
    add_slider(v, "Volume", 0, 2, lambda: engine.gain,
               lambda x: setattr(engine, "gain", x), fmt="{:.0%}")

    # ---- Tone tab ----
    to = tabs.tab("Tone")
    add_switch(to, "Equalizer", lambda: engine.eq.enabled, lambda b: setattr(engine.eq, "enabled", b))
    add_slider(to, "Low (150 Hz)", -18, 18, lambda: engine.eq.low, lambda x: setattr(engine.eq, "low", x), fmt="{:+.0f} dB")
    add_slider(to, "Mid (1 kHz)", -18, 18, lambda: engine.eq.mid, lambda x: setattr(engine.eq, "mid", x), fmt="{:+.0f} dB")
    add_slider(to, "High (4 kHz)", -18, 18, lambda: engine.eq.high, lambda x: setattr(engine.eq, "high", x), fmt="{:+.0f} dB")
    add_switch(to, "Distortion", lambda: engine.dist.enabled, lambda b: setattr(engine.dist, "enabled", b))
    add_slider(to, "Drive", 1, 30, lambda: engine.dist.drive, lambda x: setattr(engine.dist, "drive", x), fmt="{:.0f}")
    add_slider(to, "Dist. mix", 0, 1, lambda: engine.dist.mix, lambda x: setattr(engine.dist, "mix", x), fmt="{:.0%}")
    add_switch(to, "Bitcrusher (lo-fi)", lambda: engine.crush.enabled, lambda b: setattr(engine.crush, "enabled", b))
    add_slider(to, "Bits", 2, 16, lambda: engine.crush.bits, lambda x: setattr(engine.crush, "bits", int(x)), fmt="{:.0f}", steps=14)
    add_slider(to, "Downsample", 1, 20, lambda: engine.crush.downsample, lambda x: setattr(engine.crush, "downsample", int(x)), fmt="{:.0f}x", steps=19)

    # ---- Modulation tab ----
    m = tabs.tab("Modulation")
    add_switch(m, "Ring modulator (robot)", lambda: engine.ring.enabled, lambda b: setattr(engine.ring, "enabled", b))
    add_slider(m, "Ring freq", 10, 300, lambda: engine.ring.freq, lambda x: setattr(engine.ring, "freq", x), fmt="{:.0f} Hz")
    add_slider(m, "Ring mix", 0, 1, lambda: engine.ring.mix, lambda x: setattr(engine.ring, "mix", x), fmt="{:.0%}")
    add_switch(m, "Tremolo", lambda: engine.trem.enabled, lambda b: setattr(engine.trem, "enabled", b))
    add_slider(m, "Tremolo rate", 0.5, 15, lambda: engine.trem.rate, lambda x: setattr(engine.trem, "rate", x), fmt="{:.1f} Hz")
    add_slider(m, "Tremolo depth", 0, 1, lambda: engine.trem.depth, lambda x: setattr(engine.trem, "depth", x), fmt="{:.0%}")
    add_switch(m, "Vibrato", lambda: engine.vib.enabled, lambda b: setattr(engine.vib, "enabled", b))
    add_slider(m, "Vibrato rate", 0.5, 12, lambda: engine.vib.rate, lambda x: setattr(engine.vib, "rate", x), fmt="{:.1f} Hz")
    add_slider(m, "Vibrato depth", 0, 8, lambda: engine.vib.depth_ms, lambda x: setattr(engine.vib, "depth_ms", x), fmt="{:.1f} ms")

    # ---- Space tab ----
    sp = tabs.tab("Space")
    add_switch(sp, "Echo / delay", lambda: engine.echo.enabled, lambda b: setattr(engine.echo, "enabled", b))
    add_slider(sp, "Echo time", 50, 800, lambda: engine.echo.time_ms, lambda x: setattr(engine.echo, "time_ms", x), fmt="{:.0f} ms")
    add_slider(sp, "Echo feedback", 0, 0.9, lambda: engine.echo.feedback, lambda x: setattr(engine.echo, "feedback", x), fmt="{:.0%}")
    add_slider(sp, "Echo mix", 0, 1, lambda: engine.echo.mix, lambda x: setattr(engine.echo, "mix", x), fmt="{:.0%}")
    add_switch(sp, "Reverb", lambda: engine.reverb.enabled, lambda b: setattr(engine.reverb, "enabled", b))
    add_slider(sp, "Room size", 0, 1, lambda: engine.reverb.size, lambda x: setattr(engine.reverb, "size", x), fmt="{:.0%}")
    add_slider(sp, "Damping", 0, 0.95, lambda: engine.reverb.damp, lambda x: setattr(engine.reverb, "damp", x), fmt="{:.0%}")
    add_slider(sp, "Reverb mix", 0, 1, lambda: engine.reverb.mix, lambda x: setattr(engine.reverb, "mix", x), fmt="{:.0%}")

    def refresh_controls():
        for r in controls:
            r()

    # --- meters + transport ------------------------------------------------
    meter_frame = ctk.CTkFrame(app)
    meter_frame.pack(fill="x", padx=16, pady=(6, 4))
    ctk.CTkLabel(meter_frame, text="In").grid(row=0, column=0, padx=(12, 6), pady=(10, 2))
    in_meter = ctk.CTkProgressBar(meter_frame, width=490)
    in_meter.grid(row=0, column=1, pady=(10, 2))
    ctk.CTkLabel(meter_frame, text="Out").grid(row=1, column=0, padx=(12, 6), pady=(0, 10))
    out_meter = ctk.CTkProgressBar(meter_frame, width=490)
    out_meter.grid(row=1, column=1, pady=(0, 10))
    in_meter.set(0); out_meter.set(0)

    status = ctk.CTkLabel(app, text="stopped", text_color="gray60")

    transport = ctk.CTkFrame(app, fg_color="transparent")
    transport.pack(fill="x", padx=16, pady=(4, 2))

    def toggle():
        if engine.stream is None:
            try:
                engine.start(inputs[in_box.get()], outputs[out_box.get()])
            except Exception as exc:
                status.configure(text=f"error: {exc}", text_color="#e05f5f")
                return
            start_btn.configure(text="■  Stop", fg_color="#a33c3c", hover_color="#8a3232")
            latency_ms = 1000 * (engine.shifter.n + BLOCK_SIZE) / SAMPLE_RATE
            hint = "  ·  Ctrl+Alt+1…9 presets" if HAVE_KEYBOARD else ""
            status.configure(text=f"running · ~{latency_ms:.0f} ms latency{hint}", text_color="gray60")
            update_monitor()
        else:
            engine.stop()
            start_btn.configure(text="▶  Start", fg_color=["#3B8ED0", "#1F6AA5"], hover_color=["#36719F", "#144870"])
            status.configure(text="stopped", text_color="gray60")

    def toggle_rec():
        if not engine.recording:
            engine.start_recording()
            rec_btn.configure(text="● Recording", fg_color="#a33c3c", hover_color="#8a3232")
            status.configure(text="recording…", text_color="#e0a24a")
        else:
            path = engine.stop_recording()
            rec_btn.configure(text="● Rec", fg_color="#555", hover_color="#666")
            status.configure(text=f"saved {os.path.basename(path)}" if path else "nothing recorded",
                             text_color="gray60")

    start_btn = ctk.CTkButton(transport, text="▶  Start", height=42,
                              font=ctk.CTkFont(size=15, weight="bold"), command=toggle)
    start_btn.pack(side="left", fill="x", expand=True, padx=(0, 6))
    rec_btn = ctk.CTkButton(transport, text="● Rec", width=120, height=42, fg_color="#555",
                            hover_color="#666", command=toggle_rec)
    rec_btn.pack(side="left")
    status.pack(pady=(2, 8))

    # --- hotkeys -----------------------------------------------------------
    def register_hotkeys():
        if not HAVE_KEYBOARD:
            return
        try:
            keyboard.remove_all_hotkeys()
        except Exception:
            pass
        for idx, name in enumerate(list(PRESETS)[:9], start=1):
            keyboard.add_hotkey(f"ctrl+alt+{idx}", lambda n=name: app.after(0, lambda: select_preset(n)))
        keyboard.add_hotkey("ctrl+alt+0", lambda: app.after(0, toggle_mute))

    def toggle_mute():
        engine.muted = not engine.muted
        status.configure(text="muted" if engine.muted else "running",
                         text_color="#e0a24a" if engine.muted else "gray60")

    register_hotkeys()
    set_category("Human")
    select_preset("Normal")

    sig_hint = ctk.CTkLabel(app, text="", text_color="#c98a3a", font=ctk.CTkFont(size=11))
    sig_hint.pack(pady=(0, 4))
    silent_polls = {"n": 0}

    def poll_meters():
        in_meter.set(min(1.0, engine.level_in * 4))
        out_meter.set(min(1.0, engine.level_out * 4))
        if engine.stream is not None:
            silent_polls["n"] = silent_polls["n"] + 1 if engine.level_in < 0.002 else 0
            sig_hint.configure(text="⚠ no signal from this microphone — pick your real mic above"
                               if silent_polls["n"] > 40 else "")
        else:
            sig_hint.configure(text="")
            silent_polls["n"] = 0
        app.after(50, poll_meters)
    poll_meters()

    def on_close():
        if HAVE_KEYBOARD:
            try:
                keyboard.remove_all_hotkeys()
            except Exception:
                pass
        if engine.recording:
            engine.stop_recording()
        engine.stop()
        app.destroy()

    app.protocol("WM_DELETE_WINDOW", on_close)
    app.mainloop()


if __name__ == "__main__":
    main()
