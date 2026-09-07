# VoxGuard - AASIST Inference Module
# Model: Pretrained AASIST checkpoint (clovaai/aasist)
# Dataset: Evaluated on ASVspoof 2019 LA
# Note: No fine-tuning performed. Pretrained inference only.

"""
Single entry point: ``analyze_audio(wav_path) -> float``.

Loads audio, resamples to 16 kHz mono, peak-normalises, runs the pretrained
AASIST model and returns a spoof score in [0.0, 1.0] where:

    0.0  -> genuine / bonafide speech
    1.0  -> spoofed / synthetic / cloned speech

Any failure (missing weights, bad audio, CUDA OOM, ...) returns 0.5 so the
caller always gets a usable float.
"""

import json
import os
import sys
import traceback

import librosa
import numpy as np
import torch

# --- paths -------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_AASIST_DIR = os.path.join(_HERE, "aasist")
_CONFIG_PATH = os.path.join(_AASIST_DIR, "config", "AASIST.conf")
_WEIGHTS_PATH = os.path.join(_AASIST_DIR, "models", "weights", "AASIST.pth")

# make ``from models.AASIST import Model`` resolve against the cloned repo
if _AASIST_DIR not in sys.path:
    sys.path.insert(0, _AASIST_DIR)

# --- constants -------------------------------------------------------------
_TARGET_SR = 16000
# AASIST consumes ~4 s of raw waveform (64600 samples @ 16 kHz)
_NB_SAMP = 64600

# --- lazily-initialised singletons ---------------------------------------
_model = None
_device = None


def _resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_state_dict(path: str, device: torch.device):
    """torch>=2.6 defaults to weights_only=True; the AASIST checkpoint is a
    plain state-dict so that works, but fall back just in case."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


def _load_model():
    """Build the AASIST architecture and load pretrained weights (once)."""
    global _model, _device
    if _model is not None:
        return _model, _device

    device = _resolve_device()

    if not os.path.isfile(_CONFIG_PATH):
        raise FileNotFoundError(f"AASIST config not found: {_CONFIG_PATH}")
    if not os.path.isfile(_WEIGHTS_PATH):
        raise FileNotFoundError(
            f"AASIST weights not found: {_WEIGHTS_PATH}. "
            "Expected in the cloned clovaai/aasist repo under models/weights/."
        )

    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        config = json.load(fh)
    model_config = config["model_config"]

    # architecture lives in aasist/models/AASIST.py
    from models.AASIST import Model

    model = Model(model_config).to(device)
    model.load_state_dict(_load_state_dict(_WEIGHTS_PATH, device))
    model.eval()

    _model = model
    _device = device
    return _model, _device


def _preprocess(wav_path: str) -> np.ndarray:
    """Load -> 16 kHz mono -> peak-normalise -> pad/truncate to _NB_SAMP."""
    y, _ = librosa.load(wav_path, sr=_TARGET_SR, mono=True)
    y = np.asarray(y, dtype=np.float32)

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y = y / peak

    if y.size >= _NB_SAMP:
        y = y[:_NB_SAMP]
    else:
        # tile the clip to reach the required length (AASIST-style padding)
        repeats = int(np.ceil(_NB_SAMP / max(y.size, 1)))
        y = np.tile(y, repeats)[:_NB_SAMP]

    return np.ascontiguousarray(y, dtype=np.float32)


def analyze_audio(wav_path: str) -> float:
    """
    Loads audio, preprocesses to 16kHz mono, runs AASIST inference,
    returns spoof_score between 0.0 (genuine) and 1.0 (spoof).
    Returns 0.5 as safe fallback on any error.
    """
    try:
        model, device = _load_model()
        waveform = _preprocess(wav_path)

        x = torch.from_numpy(waveform).unsqueeze(0).to(device)  # [1, 64600]
        with torch.no_grad():
            out = model(x)
            logits = out[1] if isinstance(out, (tuple, list)) else out
            # AASIST: column 0 = spoof, column 1 = bonafide
            probs = torch.softmax(logits, dim=1)
            spoof_score = float(probs[0, 0].item())

        if not np.isfinite(spoof_score):
            return 0.5
        return max(0.0, min(1.0, spoof_score))
    except Exception:
        traceback.print_exc()
        return 0.5


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    if target:
        print(f"{target}: spoof_score={analyze_audio(target):.4f}")
    else:
        print("usage: python model.py <path-to-wav>")
