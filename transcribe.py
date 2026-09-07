# VoxGuard - local speech-to-text via faster-whisper
"""
Single entry point: ``transcribe(wav_path) -> str``.

Runs a local faster-whisper model (no external API calls) over a 16 kHz
mono WAV file and returns the transcript text. Any failure returns "".
"""

import os
import traceback

from faster_whisper import WhisperModel

_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "tiny")
_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")

_model: WhisperModel | None = None


def _load_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(_MODEL_SIZE, device=_DEVICE, compute_type=_COMPUTE_TYPE)
    return _model


def transcribe(wav_path: str) -> str:
    """Transcribe a WAV file to text. Returns "" on any failure."""
    try:
        model = _load_model()
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
