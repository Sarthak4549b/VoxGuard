# VoxGuard - local speech-to-text via faster-whisper
"""
Entry point: ``transcribe(wav_path, model_size=..., fast=...) -> str``.

Runs a local faster-whisper model (no external API calls) over a 16 kHz mono
WAV file and returns the transcript text. Any failure returns "".

Two pipeline paths use different models:
  - live  /analyze      -> "tiny.en" + greedy fast-decode (< ~1s per 3s chunk)
  - demo  /analyze-full -> "base"    + beam search (accuracy over latency)

Each model size is loaded lazily and cached separately, so switching between
the two paths never reloads a model.
"""

import os
import traceback

from faster_whisper import WhisperModel

_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")

# Default model sizes per path (env-overridable).
LIVE_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE_LIVE", "tiny.en")
FULL_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "base")

# One cached WhisperModel per size. Populated on first use of that size.
_models: dict[str, WhisperModel] = {}

# Greedy, no temperature fallback sweep, silence trimmed by VAD - keeps a
# 3-second live chunk decoding in well under a second on CPU int8.
_FAST_DECODE: dict = dict(
    beam_size=1,
    best_of=1,
    temperature=0.0,
    condition_on_previous_text=False,
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500),
)

# Beam search for the demo upload path where latency does not matter.
_ACCURATE_DECODE: dict = dict(
    beam_size=5,
    condition_on_previous_text=False,
)


def _load_model(model_size: str) -> WhisperModel:
    """Lazily construct and cache one WhisperModel per model size."""
    model = _models.get(model_size)
    if model is None:
        model = WhisperModel(model_size, device=_DEVICE, compute_type=_COMPUTE_TYPE)
        _models[model_size] = model
    return model


def transcribe(
    wav_path: str,
    model_size: str = LIVE_MODEL_SIZE,
    *,
    fast: bool = True,
) -> str:
    """Transcribe a WAV file to text.

    Args:
        wav_path:   path to a 16 kHz mono WAV file.
        model_size: faster-whisper model size ("tiny.en", "base", ...).
        fast:       True  -> greedy + VAD fast-decode (live path).
                    False -> beam search (demo path).

    Returns "" on any failure.
    """
    try:
        model = _load_model(model_size)
        opts = _FAST_DECODE if fast else _ACCURATE_DECODE
        try:
            segments, _info = model.transcribe(wav_path, **opts)
            text = " ".join(segment.text.strip() for segment in segments)
        except Exception:
            # e.g. VAD assets unavailable - retry once with plain greedy decode.
            traceback.print_exc()
            segments, _info = model.transcribe(wav_path, beam_size=1)
            text = " ".join(segment.text.strip() for segment in segments)
        return text.strip()
    except Exception:
        traceback.print_exc()
        return ""


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else None
    if target:
        print(transcribe(target))
    else:
        print("usage: python transcribe.py <path-to-wav>")
