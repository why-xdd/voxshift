"""VoxShift RVC engine — file-free ONNX voice conversion.

Wraps the OnnxRVC pipeline from tts-with-rvc-onnx so it can convert a NumPy
audio array directly (no temp files), which is the building block for both
the "convert a recording" flow and real-time chunked streaming.

Requires the project venv (onnxruntime, tts-with-rvc, librosa, ...).
Base models (vec-768-layer-12.onnx, rmvpe.onnx) are expected in the working
directory or are auto-downloaded on first use by the underlying library.
"""

import os

import librosa
import numpy as np
from scipy import signal

from tts_with_rvc.lib.infer_pack.onnx_inference import (
    OnnxRVC, get_f0_predictor, change_rms, RMVPEONNXPredictor,
)

HERE = os.path.dirname(os.path.abspath(__file__))


class RVCEngine:
    """Loads one RVC voice model and converts audio arrays to that voice."""

    def __init__(self, model_path, device="cpu", f0_method="rmvpe",
                 vec_path=None, sr=40000, hop=512):
        self.device = device
        self.f0_method = f0_method
        vec_path = vec_path or os.path.join(HERE, "vec-768-layer-12.onnx")
        self.m = OnnxRVC(model_path=model_path, sr=sr, hop_size=hop,
                         vec_path=vec_path, device=device)
        self._f0p = None
        self._f0m = None

    def _f0(self, method):
        if self._f0p is None or self._f0m != method:
            self._f0p = get_f0_predictor(
                method, hop_length=self.m.f0_hop_size,
                sampling_rate=self.m.sampling_rate, device=self.device,
                cr_threshold=0.05)
            self._f0m = method
        return self._f0p

    @property
    def sr(self):
        return self.m.sampling_rate

    def convert(self, wav, sr_in, sid=0, pitch=0, f0_method=None,
                index_file=None, index_rate=0.0, filter_radius=5,
                rms_mix_rate=0.25, protect=0.35):
        """Convert a float audio array to the loaded voice.

        Returns (float32 array at self.sr, self.sr).
        """
        m = self.m
        f0_method = f0_method or self.f0_method
        wav = np.asarray(wav, dtype=np.float64)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        if sr_in != m.sampling_rate:
            wav_main = librosa.resample(wav, orig_sr=sr_in, target_sr=m.sampling_rate, res_type="soxr_hq")
        else:
            wav_main = wav
        original_length = len(wav_main)
        if original_length < m.f0_hop_size * 2:
            return np.zeros(original_length, dtype=np.float32), m.sampling_rate

        wav_padded = np.pad(wav_main, (m.t_pad_main_sr, m.t_pad_main_sr), mode="reflect")
        wav16k = librosa.resample(wav_main, orig_sr=m.sampling_rate, target_sr=m.sr_hubert, res_type="soxr_hq")
        wav16k_padded = np.pad(wav16k, (m.t_pad_hubert_sr, m.t_pad_hubert_sr), mode="reflect")

        p_len = wav_padded.shape[0] // m.f0_hop_size
        f0p = self._f0(f0_method)
        kw = {"wav": wav_padded, "p_len": p_len}
        if isinstance(f0p, RMVPEONNXPredictor):
            kw["orig_sr"] = m.sampling_rate
        pitchf = f0p.compute_f0(**kw)

        if filter_radius >= 3:
            pad = int((filter_radius - 1) / 2)
            if pad > 0:
                pitchf = signal.medfilt(np.pad(pitchf, pad, mode="reflect"), filter_radius)[pad:-pad]
            else:
                pitchf = signal.medfilt(pitchf, filter_radius)

        pitchf = pitchf * (2 ** (pitch / 12.0))
        pit = pitchf.copy()
        f0_min, f0_max = 50.0, 1100.0
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        f0_mel = 1127 * np.log(1 + pit / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        pit = np.rint(f0_mel).astype(np.int64)

        hub = m.vec_model(wav16k_padded)
        hub = np.repeat(hub, 2, axis=2)
        hlen = hub.shape[2]

        if hlen != len(pit):
            tgt = hlen
            if len(pit) < tgt:
                pit = np.pad(pit, (0, tgt - len(pit)), "constant", constant_values=pit[-1] if len(pit) else 1)
            else:
                pit = pit[:tgt]
            if len(pitchf) < tgt:
                pitchf = np.interp(np.linspace(0, 1, tgt), np.linspace(0, 1, len(pitchf)), pitchf)
            else:
                pitchf = pitchf[:tgt]

        if index_file:
            index, big = m.load_index(index_file)
            if index is not None and index_rate > 0:
                hub = m.apply_index(hub, index, big, index_rate)
        hub = m.apply_protection(hub, pitchf, protect)

        pit = pit.reshape(1, hlen)
        pitf = pitchf.reshape(1, hlen).astype(np.float32)
        ds = np.array([sid]).astype(np.int64)
        rnd = np.random.randn(1, 192, hlen).astype(np.float32)
        hlen_np = np.array([hlen]).astype(np.int64)

        out = m.forward(hub, hlen_np, pit, pitf, ds, rnd).squeeze()
        ts = m.t_pad_main_sr
        te = min(ts + original_length, len(out))
        out = out[ts:te]
        if len(out) < original_length:
            out = np.pad(out, (0, original_length - len(out)), "constant")

        if rms_mix_rate < 1.0:
            out = change_rms(wav_main, m.sampling_rate, out, m.sampling_rate, rms_mix_rate)
        mx = np.abs(out).max() / 0.99
        if mx > 1:
            out = out / mx
        return out.astype(np.float32), m.sampling_rate
