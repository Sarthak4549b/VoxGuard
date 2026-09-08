# VoxGuard - speaker verification via SpeechBrain ECAPA-TDNN (VoxCeleb)
"""
Speaker enrollment + verification backed by SpeechBrain's pretrained
``speechbrain/spkrec-ecapa-voxceleb`` embedding model.

- ``enroll_speaker(wav_path)``   -> extracts an embedding and persists it to
  disk, overwriting any previously enrolled speaker (single-speaker demo).
- ``verify_speaker(wav_path)``   -> (similarity, status), comparing against
  the enrolled embedding. status is one of MATCH / MISMATCH / UNKNOWN.

Any failure returns a safe fallback (0.0 similarity, UNKNOWN) rather than
raising, so the FastAPI layer never crashes on a bad clip.
"""

import os
import traceback

import numpy as np
from speechbrain.dataio import audio_io
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

_HERE = os.path.dirname(os.path.abspath(__file__))
_STORAGE_DIR = os.path.join(_HERE, "data")
_ENROLLED_PATH = os.path.join(_STORAGE_DIR, "enrolled_speaker.npy")
# Lowered 0.70 -> 0.60: laptop-mic capture + browser WebM/Opus compression
# pull a genuine enrolled speaker's similarity down enough that 0.70 caused
# false MISMATCHes in the live demo.
_MATCH_THRESHOLD = 0.6

# ECAPA is small; keep it on CPU so it doesn't open a second CUDA context
# alongside AASIST's (each CUDA context has real host-RAM overhead on top of
# VRAM, which mattered on this 4GB-VRAM / memory-constrained machine).
_device = os.environ.get("SPEAKER_DEVICE", "cpu")

_classifier: EncoderClassifier | None = None


def _load_classifier() -> EncoderClassifier:
    global _classifier
    if _classifier is None:
        _classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.join(_HERE, "pretrained_models", "spkrec-ecapa-voxceleb"),
            run_opts={"device": _device},
            # SYMLINK (the default) needs elevated privileges on Windows;
            # COPY works everywhere.
            local_strategy=LocalStrategy.COPY,
        )
    return _classifier


def _embed(wav_path: str) -> np.ndarray:
    # NOTE: classifier.load_audio() is meant for named/fetchable HF resources
    # (it re-roots plain local paths against savedir/cwd), so we load the
    # already-16kHz-mono WAV directly instead, per speechbrain's own
    # EncoderClassifier docstring example.
    classifier = _load_classifier()
    signal, _sr = audio_io.load(wav_path)
    embedding = classifier.encode_batch(signal)
    vec = embedding.squeeze().detach().cpu().numpy().astype(np.float32)
    return vec


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    sim = float(np.dot(a, b) / denom)
    # cosine similarity is in [-1, 1]; clamp into [0, 1] for a 0-1 UI score
    return max(0.0, min(1.0, (sim + 1.0) / 2.0))


def has_enrolled_speaker() -> bool:
    return os.path.isfile(_ENROLLED_PATH)


def enroll_speaker(wav_path: str) -> bool:
    """Extract an embedding from wav_path and persist it. Returns success bool."""
    try:
        os.makedirs(_STORAGE_DIR, exist_ok=True)
        vec = _embed(wav_path)
        np.save(_ENROLLED_PATH, vec)
        return True
    except Exception:
        traceback.print_exc()
        return False


def verify_speaker(wav_path: str) -> tuple[float, str]:
    """
    Compare wav_path's speaker embedding against the enrolled speaker.
    Returns (speaker_similarity, speaker_status).
    """
    if not has_enrolled_speaker():
        return 0.0, "UNKNOWN"

    try:
        enrolled = np.load(_ENROLLED_PATH)
        candidate = _embed(wav_path)
        similarity = _cosine_similarity(enrolled, candidate)
        status = "MATCH" if similarity > _MATCH_THRESHOLD else "MISMATCH"
        return similarity, status
    except Exception:
        traceback.print_exc()
        return 0.0, "UNKNOWN"
