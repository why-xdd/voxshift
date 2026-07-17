"""VoxShift DSP — real-time voice effect blocks.

Every effect is a small stateful object with:
    * public parameter attributes (set from the GUI, read in the audio thread)
    * an ``enabled`` flag
    * ``process(block)`` returning a same-length 1-D float64 array
    * ``reset()``

All blocks are vectorized with NumPy/SciPy so the whole chain runs inside a
~11 ms audio callback without per-sample Python loops (which would drop out).
Delay-based effects use ring buffers that require the delay to be >= the audio
block size, which is always true at 512-sample blocks / 44.1 kHz.
"""

import numpy as np
from scipy.signal import lfilter, lfilter_zi

SAMPLE_RATE = 44100


# ---------------------------------------------------------------------------
# Pitch + formant shifter (phase vocoder)
# ---------------------------------------------------------------------------
class PitchFormantShifter:
    """STFT phase vocoder with independent pitch and formant control.

    - ``pitch`` (semitones) shifts the whole spectrum (pitch *and* harmonics).
    - ``preserve`` extracts the spectral envelope (formants / timbre) via
      cepstral liftering and re-applies it so the voice keeps a natural
      character instead of sounding like a chipmunk or a giant.
    - ``formant`` (semitones) shifts that envelope *independently* of pitch —
      this is the gender/character control: pitch down + formant down = deeper
      male, pitch up + formant up = higher female, formant-only = same pitch
      but different timbre.
    """

    def __init__(self, frame_size=1024, oversampling=8, samplerate=SAMPLE_RATE):
        self.n = frame_size
        self.osamp = oversampling
        self.hop = frame_size // oversampling
        self.sr = samplerate
        self.window = np.hanning(self.n).astype(np.float64)
        self.bins = self.n // 2 + 1
        self.freq_per_bin = samplerate / self.n
        self.expected = 2.0 * np.pi * self.hop / self.n * np.arange(self.bins)
        self.ola_gain = 0.375 * oversampling
        # smooth low-quefrency lifter to isolate the spectral (formant) envelope.
        # a raised-cosine taper avoids the ringing/coloration a hard cutoff adds.
        q = np.minimum(np.arange(self.n), self.n - np.arange(self.n))
        lifter, taper = 48, 28
        ramp = np.clip((lifter + taper - q) / taper, 0.0, 1.0)
        self.lifter_win = 0.5 - 0.5 * np.cos(np.pi * ramp)  # smootherstep-ish 0..1
        self._bin_idx = np.arange(self.bins)

        self.pitch = 0.0       # semitones
        self.formant = 0.0     # semitones (envelope shift)
        self.preserve = True
        self.lock_mode = "rigid"  # "rigid" phase-locking or "none"
        self.transient_preserve = False  # experimental onset phase reset (off: hurt quality here)
        self.reset()

    def reset(self):
        self.in_ring = np.zeros(self.n)
        self.out_acc = np.zeros(self.n)
        self.last_phase = np.zeros(self.bins)
        self.sum_phase = np.zeros(self.bins)
        self.in_buf = np.zeros(0)
        self.out_buf = np.zeros(self.n)
        self.env_in = 0.0
        self.env_out = 0.0
        self.level_gain = 1.0
        self.prev_mag = np.zeros(self.bins)
        self.flux_avg = 0.0

    def _envelope(self, mag):
        log_mag = np.log(mag + 1e-9)
        cep = np.fft.irfft(log_mag, n=self.n)
        cep = cep * self.lifter_win
        return np.exp(np.fft.rfft(cep).real[: self.bins])

    def _locked_phase(self, mag):
        """Return synthesis phases, optionally phase-locked to spectral peaks.

        Rigid (identity) phase locking assigns every bin to its nearest
        magnitude peak and gives it that peak's propagated phase, so all the
        bins of one harmonic stay phase-coherent. This removes most of the
        "phasey / metallic / robotic" character of the plain phase vocoder.
        """
        if self.lock_mode != "rigid":
            return self.sum_phase
        m = mag
        # local maxima above a small fraction of the frame's peak magnitude
        thr = 1e-4 * (m.max() if m.size else 0.0)
        pk = (m[1:-1] > m[:-2]) & (m[1:-1] >= m[2:]) & (m[1:-1] > thr)
        peak_idx = np.nonzero(pk)[0] + 1
        if peak_idx.size == 0:
            return self.sum_phase
        # assign each bin to the nearest peak (regions split at peak midpoints)
        bounds = (peak_idx[:-1] + peak_idx[1:] + 1) // 2
        region = np.searchsorted(bounds, self._bin_idx)
        return self.sum_phase[peak_idx[region]]

    def _process_frame(self, ratio, fscale):
        spec = np.fft.rfft(self.in_ring * self.window)
        mag = np.abs(spec)
        phase = np.angle(spec)

        delta = phase - self.last_phase - self.expected
        self.last_phase = phase
        delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
        true_freq = (self._bin_idx + delta * self.osamp / (2.0 * np.pi)) * self.freq_per_bin

        env = None
        src = mag
        if self.preserve:
            env = self._envelope(mag)
            src = mag / (env + 1e-9)

        pos = self._bin_idx * ratio
        lo = pos.astype(np.int64)
        frac = pos - lo
        new_mag = np.zeros(self.bins)
        freq_sum = np.zeros(self.bins)
        v = lo < self.bins
        np.add.at(new_mag, lo[v], src[v] * (1.0 - frac[v]))
        np.add.at(freq_sum, lo[v], src[v] * (1.0 - frac[v]) * true_freq[v] * ratio)
        v = lo + 1 < self.bins
        np.add.at(new_mag, lo[v] + 1, src[v] * frac[v])
        np.add.at(freq_sum, lo[v] + 1, src[v] * frac[v] * true_freq[v] * ratio)
        new_freq = np.zeros(self.bins)
        nz = new_mag > 1e-12
        new_freq[nz] = freq_sum[nz] / new_mag[nz]

        if env is not None:
            if abs(fscale - 1.0) > 1e-3:
                warped = np.interp(self._bin_idx / fscale, self._bin_idx, env)
            else:
                warped = env
            new_mag = new_mag * warped

        # transient detection via spectral flux — on an onset (consonant), reset
        # the synthesis phase to the (shifted) analysis phase so the attack stays
        # sharp instead of smearing into the "robotic mush" the vocoder produces
        transient = False
        if self.transient_preserve:
            flux = float(np.sum(np.maximum(0.0, mag - self.prev_mag)))
            self.prev_mag = mag
            if flux > 3.0 * self.flux_avg and flux > 5e-3:
                transient = True
            self.flux_avg = 0.9 * self.flux_avg + 0.1 * flux

        self.sum_phase += 2.0 * np.pi * new_freq / self.freq_per_bin / self.osamp
        if transient:
            src_bins = np.clip((self._bin_idx / ratio).astype(np.int64), 0, self.bins - 1)
            self.sum_phase = phase[src_bins].copy()
        synth_phase = self._locked_phase(new_mag)
        frame = np.fft.irfft(new_mag * np.exp(1j * synth_phase))
        frame *= self.window / self.ola_gain

        self.out_acc += frame
        out = self.out_acc[: self.hop].copy()
        self.out_acc[: -self.hop] = self.out_acc[self.hop:]
        self.out_acc[-self.hop:] = 0.0
        return out

    def process(self, block):
        active = abs(self.pitch) > 1e-3 or (self.preserve and abs(self.formant) > 1e-3)
        if not active:
            self.in_buf = np.concatenate([self.in_buf, block])
            while len(self.in_buf) >= self.hop:
                chunk, self.in_buf = self.in_buf[: self.hop], self.in_buf[self.hop:]
                self.out_buf = np.concatenate([self.out_buf, chunk])
        else:
            ratio = 2.0 ** (self.pitch / 12.0)
            fscale = 2.0 ** (self.formant / 12.0)
            self.in_buf = np.concatenate([self.in_buf, block])
            while len(self.in_buf) >= self.hop:
                chunk, self.in_buf = self.in_buf[: self.hop], self.in_buf[self.hop:]
                self.in_ring[: -self.hop] = self.in_ring[self.hop:]
                self.in_ring[-self.hop:] = chunk
                self.out_buf = np.concatenate([self.out_buf, self._process_frame(ratio, fscale)])

        out, self.out_buf = self.out_buf[: len(block)], self.out_buf[len(block):]
        if len(out) < len(block):
            out = np.pad(out, (len(block) - len(out), 0))

        if active:
            self.env_in = 0.9 * self.env_in + 0.1 * float(np.sqrt(np.mean(block ** 2)))
            self.env_out = 0.9 * self.env_out + 0.1 * float(np.sqrt(np.mean(out ** 2)))
            if self.env_out > 1e-5 and self.env_in > 1e-5:
                target = float(np.clip(self.env_in / self.env_out, 0.25, 4.0))
            else:
                target = self.level_gain
            # smooth the make-up gain and ramp it across the block so the level
            # never jumps at a block boundary (that would sound like roughness)
            new_gain = 0.85 * self.level_gain + 0.15 * target
            out = out * np.linspace(self.level_gain, new_gain, len(out))
            self.level_gain = new_gain
        return out


# ---------------------------------------------------------------------------
# Dynamics / cleanup
# ---------------------------------------------------------------------------
class NoiseGate:
    """Attenuates the signal while it sits below ``threshold`` (RMS amplitude)."""

    def __init__(self, samplerate=SAMPLE_RATE):
        self.sr = samplerate
        self.threshold = 0.0
        self.gain = 1.0
        self.env = 0.0

    def reset(self):
        self.gain = 1.0
        self.env = 0.0

    def process(self, block):
        if self.threshold <= 0.0:
            self.gain = 1.0
            return block
        rms = float(np.sqrt(np.mean(block ** 2)))
        self.env = max(rms, 0.85 * self.env)
        target = 1.0 if self.env >= self.threshold else 0.0
        coeff = 0.5 if target > self.gain else 0.05
        new_gain = self.gain + (target - self.gain) * coeff
        ramp = np.linspace(self.gain, new_gain, len(block))
        self.gain = new_gain
        return block * ramp


# ---------------------------------------------------------------------------
# Tone shaping: 3-band EQ (RBJ biquads)
# ---------------------------------------------------------------------------
def _biquad_lowshelf(f0, gain_db, sr, S=1.0):
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / sr
    c, s = np.cos(w0), np.sin(w0)
    alpha = s / 2 * np.sqrt((A + 1 / A) * (1 / S - 1) + 2)
    tsa = 2 * np.sqrt(A) * alpha
    b0 = A * ((A + 1) - (A - 1) * c + tsa)
    b1 = 2 * A * ((A - 1) - (A + 1) * c)
    b2 = A * ((A + 1) - (A - 1) * c - tsa)
    a0 = (A + 1) + (A - 1) * c + tsa
    a1 = -2 * ((A - 1) + (A + 1) * c)
    a2 = (A + 1) + (A - 1) * c - tsa
    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])


def _biquad_highshelf(f0, gain_db, sr, S=1.0):
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / sr
    c, s = np.cos(w0), np.sin(w0)
    alpha = s / 2 * np.sqrt((A + 1 / A) * (1 / S - 1) + 2)
    tsa = 2 * np.sqrt(A) * alpha
    b0 = A * ((A + 1) + (A - 1) * c + tsa)
    b1 = -2 * A * ((A - 1) + (A + 1) * c)
    b2 = A * ((A + 1) + (A - 1) * c - tsa)
    a0 = (A + 1) - (A - 1) * c + tsa
    a1 = 2 * ((A - 1) - (A + 1) * c)
    a2 = (A + 1) - (A - 1) * c - tsa
    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])


def _biquad_peak(f0, gain_db, sr, Q=1.0):
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / sr
    c, s = np.cos(w0), np.sin(w0)
    alpha = s / (2 * Q)
    b0 = 1 + alpha * A
    b1 = -2 * c
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * c
    a2 = 1 - alpha / A
    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])


class EQ3:
    """Low-shelf @ 150 Hz, mid peak @ 1 kHz, high-shelf @ 4 kHz, gains in dB."""

    def __init__(self, samplerate=SAMPLE_RATE):
        self.sr = samplerate
        self.enabled = False
        self.low = 0.0
        self.mid = 0.0
        self.high = 0.0
        self._cache = None
        self._zi = [None, None, None]
        self.reset()

    def reset(self):
        self._zi = [None, None, None]

    def _coeffs(self):
        key = (round(self.low, 2), round(self.mid, 2), round(self.high, 2))
        if self._cache and self._cache[0] == key:
            return self._cache[1]
        bands = [
            _biquad_lowshelf(150.0, self.low, self.sr),
            _biquad_peak(1000.0, self.mid, self.sr, Q=0.9),
            _biquad_highshelf(4000.0, self.high, self.sr),
        ]
        self._cache = (key, bands)
        return bands

    def process(self, block):
        if not self.enabled or (abs(self.low) < 0.1 and abs(self.mid) < 0.1 and abs(self.high) < 0.1):
            return block
        y = block
        for i, (b, a) in enumerate(self._coeffs()):
            if self._zi[i] is None:
                self._zi[i] = lfilter_zi(b, a) * y[0]
            y, self._zi[i] = lfilter(b, a, y, zi=self._zi[i])
        return y


# ---------------------------------------------------------------------------
# Distortion / overdrive
# ---------------------------------------------------------------------------
class Distortion:
    """Soft-clipping overdrive with a wet/dry mix. ``drive`` 1..50."""

    def __init__(self):
        self.enabled = False
        self.drive = 5.0
        self.mix = 1.0

    def reset(self):
        pass

    def process(self, block):
        if not self.enabled or self.mix <= 0.0:
            return block
        d = max(1.0, self.drive)
        wet = np.tanh(d * block) / np.tanh(d)
        return block * (1.0 - self.mix) + wet * self.mix


# ---------------------------------------------------------------------------
# Bitcrusher / sample-rate reducer (lo-fi, telephone, retro robot)
# ---------------------------------------------------------------------------
class Bitcrusher:
    """Reduces bit depth and/or sample rate. ``bits`` 2..16, ``downsample`` 1..50."""

    def __init__(self):
        self.enabled = False
        self.bits = 8
        self.downsample = 1
        self.mix = 1.0
        self._hold = 0.0
        self._count = 0

    def reset(self):
        self._hold = 0.0
        self._count = 0

    def process(self, block):
        if not self.enabled or self.mix <= 0.0:
            return block
        y = block.copy()
        ds = int(max(1, self.downsample))
        if ds > 1:
            # sample-and-hold decimation (keeps every ds-th sample)
            idx = np.arange(len(y))
            keep = ((idx + self._count) % ds) == 0
            held = np.where(keep, y, np.nan)
            # forward-fill the held values
            valid = ~np.isnan(held)
            if not valid[0]:
                held[0] = self._hold
                valid[0] = True
            fill_idx = np.where(valid, np.arange(len(held)), 0)
            np.maximum.accumulate(fill_idx, out=fill_idx)
            y = held[fill_idx]
            self._hold = y[-1]
            self._count = (self._count + len(block)) % ds
        levels = float(2 ** int(self.bits))
        y = np.round(y * (levels / 2)) / (levels / 2)
        return block * (1.0 - self.mix) + y * self.mix


# ---------------------------------------------------------------------------
# Modulation: ring mod, tremolo, vibrato
# ---------------------------------------------------------------------------
class RingModulator:
    """Multiplies the signal with a sine carrier — the classic robot voice."""

    def __init__(self, samplerate=SAMPLE_RATE):
        self.sr = samplerate
        self.enabled = False
        self.freq = 70.0
        self.mix = 0.0
        self.phase = 0.0

    def reset(self):
        self.phase = 0.0

    def process(self, block):
        if not self.enabled or self.mix <= 0.0 or self.freq <= 0.0:
            return block
        step = 2.0 * np.pi * self.freq / self.sr
        phases = self.phase + step * np.arange(1, len(block) + 1)
        self.phase = float(phases[-1] % (2.0 * np.pi))
        return block * (1.0 - self.mix) + block * np.sin(phases) * self.mix


class Tremolo:
    """Amplitude LFO. ``rate`` in Hz, ``depth`` 0..1."""

    def __init__(self, samplerate=SAMPLE_RATE):
        self.sr = samplerate
        self.enabled = False
        self.rate = 5.0
        self.depth = 0.5
        self.phase = 0.0

    def reset(self):
        self.phase = 0.0

    def process(self, block):
        if not self.enabled or self.depth <= 0.0:
            return block
        step = 2.0 * np.pi * self.rate / self.sr
        ph = self.phase + step * np.arange(1, len(block) + 1)
        self.phase = float(ph[-1] % (2.0 * np.pi))
        lfo = 1.0 - self.depth + self.depth * (0.5 + 0.5 * np.sin(ph))
        return block * lfo


class Vibrato:
    """Pitch wobble via a modulated fractional delay. ``rate`` Hz, ``depth`` ms."""

    def __init__(self, samplerate=SAMPLE_RATE):
        self.sr = samplerate
        self.enabled = False
        self.rate = 5.0
        self.depth_ms = 2.0
        self.maxd = int(samplerate * 0.03)  # 30 ms buffer
        self.phase = 0.0
        self.reset()

    def reset(self):
        self.tail = np.zeros(self.maxd)
        self.phase = 0.0

    def process(self, block):
        if not self.enabled or self.depth_ms <= 0.0:
            return block
        L = len(block)
        center = self.maxd / 2.0
        depth = min(center, self.depth_ms / 1000.0 * self.sr)
        step = 2.0 * np.pi * self.rate / self.sr
        t = self.phase + step * np.arange(L)
        self.phase = float((t[-1] + step) % (2.0 * np.pi))
        delay = center + depth * np.sin(t)
        ext = np.concatenate([self.tail, block])
        base = len(self.tail)
        read = base + np.arange(L) - delay
        i0 = np.floor(read).astype(np.int64)
        frac = read - i0
        i0 = np.clip(i0, 0, len(ext) - 2)
        y = ext[i0] * (1.0 - frac) + ext[i0 + 1] * frac
        self.tail = ext[-self.maxd:].copy()
        return y


# ---------------------------------------------------------------------------
# Space: echo (feedback delay) and reverb (Schroeder combs + allpass)
# ---------------------------------------------------------------------------
class Echo:
    """Feedback delay. ``time_ms`` >= block, ``feedback`` 0..0.95, wet ``mix``."""

    def __init__(self, samplerate=SAMPLE_RATE):
        self.sr = samplerate
        self.enabled = False
        self.time_ms = 250.0
        self.feedback = 0.35
        self.mix = 0.3
        self._D = 0
        self._hist = np.zeros(1)

    def reset(self):
        self._D = 0
        self._hist = np.zeros(1)

    def _ensure(self):
        D = max(64, int(self.time_ms / 1000.0 * self.sr))
        if D != self._D:
            self._D = D
            self._hist = np.zeros(D)

    def process(self, block):
        if not self.enabled or self.mix <= 0.0:
            return block
        self._ensure()
        L = len(block)
        D = self._D
        fb = float(np.clip(self.feedback, 0.0, 0.95))
        if L <= D:
            delayed = self._hist[:L].copy()
            wet = block + fb * delayed
            self._hist = np.concatenate([self._hist[L:], wet])
        else:  # fallback if block somehow exceeds delay
            wet = block.copy()
            for n in range(L):
                d = self._hist[0]
                v = block[n] + fb * d
                wet[n] = v
                self._hist = np.roll(self._hist, -1)
                self._hist[-1] = v
        return block * (1.0 - self.mix) + wet * self.mix


class _Comb:
    def __init__(self, delay, fb, damp):
        self.D = delay
        self.fb = fb
        self.damp = damp
        self.hist = np.zeros(delay)
        self._zi = None  # one-pole damping lowpass state

    def process(self, x):
        L = len(x)
        delayed = self.hist[:L].copy()
        # one-pole lowpass damping on the delayed signal (C-speed via lfilter)
        d = float(np.clip(self.damp, 0.0, 0.95))
        b, a = np.array([1.0 - d]), np.array([1.0, -d])
        if self._zi is None:
            self._zi = np.zeros(1)
        out, self._zi = lfilter(b, a, delayed, zi=self._zi)
        y = x + self.fb * out
        if L < self.D:
            self.hist = np.concatenate([self.hist[L:], y])
        else:
            self.hist = y[-self.D:].copy()
        return y


class _Allpass:
    def __init__(self, delay, g=0.5):
        self.D = delay
        self.g = g
        self.xhist = np.zeros(delay)
        self.yhist = np.zeros(delay)

    def process(self, x):
        L = len(x)
        xd = self.xhist[:L].copy()
        yd = self.yhist[:L].copy()
        y = -self.g * x + xd + self.g * yd
        if L < self.D:
            self.xhist = np.concatenate([self.xhist[L:], x])
            self.yhist = np.concatenate([self.yhist[L:], y])
        else:
            self.xhist = x[-self.D:].copy()
            self.yhist = y[-self.D:].copy()
        return y


class Reverb:
    """Schroeder reverb: 4 parallel damped combs + 2 series allpass. Wet ``mix``."""

    # comb / allpass delays chosen > block size (512) so processing stays block-vectorized
    COMBS = [1116, 1188, 1277, 1356]
    ALLPASS = [556, 683]

    def __init__(self, samplerate=SAMPLE_RATE):
        self.sr = samplerate
        self.enabled = False
        self.size = 0.5     # 0..1 -> feedback / tail length
        self.damp = 0.4     # 0..1 -> high-frequency damping
        self.mix = 0.3
        self._build()

    def _build(self):
        fb = 0.7 + 0.28 * self.size
        self.combs = [_Comb(d, fb, self.damp) for d in self.COMBS]
        self.allpass = [_Allpass(d, 0.5) for d in self.ALLPASS]

    def reset(self):
        self._build()

    def process(self, block):
        if not self.enabled or self.mix <= 0.0:
            return block
        fb = 0.7 + 0.28 * self.size
        for c in self.combs:
            c.fb = fb
            c.damp = self.damp
        wet = np.zeros(len(block))
        for c in self.combs:
            wet += c.process(block)
        wet /= len(self.combs)
        for ap in self.allpass:
            wet = ap.process(wet)
        return block * (1.0 - self.mix) + wet * self.mix
