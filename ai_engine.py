"""AI voice-conversion engine — scaffold for the "natural / undetectable" path.

DSP effects (pitch, formant, EQ…) change the voice but stay recognisably
synthetic. To make it genuinely natural you need **neural voice conversion**
(RVC / so-vits-svc style): a model trained on a target voice re-synthesises
your speech in that voice's timbre while keeping your words and prosody.

This module defines the interface VoxShift will use and a pass-through stub,
so the audio engine already has the integration point. The real backend is
opt-in (big download + an NVIDIA GPU) and is tracked on the README roadmap.

Planned real backend:
    requirements (requirements-ai.txt):
        torch (CUDA build for your RTX GPU), torchaudio, onnxruntime-gpu,
        faiss-cpu, and a streaming RVC inference package
    models:
        a .pth/.onnx voice model + feature index per character in
        ./models/<name>/
    latency:
        ~100-300 ms (hence a separate engine from the instant DSP path)
"""

from __future__ import annotations

import numpy as np


class VoiceConverter:
    """Interface every conversion backend implements."""

    name = "base"

    def load(self, model_dir: str) -> None:
        pass

    def process(self, block: np.ndarray, samplerate: int) -> np.ndarray:
        return block

    @property
    def ready(self) -> bool:
        return False


class NullConverter(VoiceConverter):
    """Pass-through used until a real AI backend is installed (current default)."""

    name = "none"

    def process(self, block, samplerate):
        return block


def available_backends():
    """AI backends that could actually run on this machine (needs CUDA)."""
    backends = []
    try:
        import torch
        if torch.cuda.is_available():
            backends.append("rvc")
    except Exception:
        pass
    return backends


def load_converter(name: str = "none", model_dir: str = "") -> VoiceConverter:
    if name in ("none", "", None):
        return NullConverter()
    if name == "rvc":
        raise NotImplementedError(
            "The RVC backend isn't bundled yet. Install the AI stack "
            "(requirements-ai.txt), drop a voice model under ./models/, and this "
            "loader will wrap it. See the roadmap in README.md.")
    raise ValueError(f"unknown AI backend: {name}")
