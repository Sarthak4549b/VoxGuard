# VoxGuard - FastAPI ML service
"""
Serves the AASIST anti-spoofing model (ml/model.py), SpeechBrain speaker
verification (speaker_verify.py) and local faster-whisper transcription
(transcribe.py) behind a single HTTP API on http://localhost:8000.

Endpoints:
    GET  /health   -> { status: 'ok' }
    POST /analyze  -> { speaker_similarity, speaker_status, spoof_score,
                        spoof_label, transcript }
    POST /enroll   -> { enrolled: bool }

Never crashes: /analyze always returns the documented shape, falling back to
{ speaker_similarity: 0, speaker_status: 'UNKNOWN', spoof_score: 0.5,
  spoof_label: 'GENUINE', transcript: '' } on any internal failure.
"""

import base64
import binascii
import os
import subprocess
import sys
import tempfile
import traceback

import truststore

# Route Python's TLS verification through the OS (Windows) trust store.
# Without this, huggingface_hub downloads (SpeechBrain / faster-whisper
# pretrained weights) fail with CERTIFICATE_VERIFY_FAILED on machines where
# a corporate/AV network layer's root CA is trusted by Windows but isn't in
# certifi's bundled CA list.
truststore.inject_into_ssl()

import imageio_ffmpeg
import librosa
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# make the bundled ffmpeg binary available to librosa's audioread/ffmpeg
# backend so MP3/OGG/WebM inputs decode even without a system ffmpeg install
_FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
_FFMPEG_DIR = os.path.dirname(_FFMPEG_EXE)
os.environ["PATH"] = _FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "ml"))

from ml.model import analyze_audio  # noqa: E402
from speaker_verify import enroll_speaker, verify_speaker  # noqa: E402
from transcribe import transcribe  # noqa: E402

app = FastAPI(title="VoxGuard ML Service")

_TARGET_SR = 16000

_FALLBACK_RESPONSE = {
    "speaker_similarity": 0,
    "speaker_status": "UNKNOWN",
    "spoof_score": 0.5,
    "spoof_label": "GENUINE",
    "transcript": "",
}


class AnalyzeRequest(BaseModel):
    audio_base64: str


class EnrollRequest(BaseModel):
    audio_base64: str


def _decode_to_wav(audio_base64: str) -> str:
    """
    Decode base64 audio (any container ffmpeg can read - WAV, MP3, OGG,
    WebM/Opus from the browser's MediaRecorder, ...) and write it out as a
    16kHz mono WAV temp file. Returns the path to that WAV file; caller is
    responsible for deleting it.

    The browser's MediaRecorder API emits WebM/Opus, which libsndfile (and
    therefore a bare librosa/soundfile load) rejects with "Format not
    recognised". So we always run the bytes through ffmpeg first: it sniffs
    the container from the stream itself (no -f needed on the input), and
    transcodes to a plain 16kHz mono PCM WAV that librosa can load directly.
    """
    raw_bytes = base64.b64decode(audio_base64, validate=False)

    ff_fd, ff_path = tempfile.mkstemp(suffix=".wav")
    os.close(ff_fd)
    try:
        # -i pipe:0  -> read the input container from stdin; ffmpeg
        #               auto-detects WebM/MP3/OGG/WAV from the byte stream.
        # -ar/-ac    -> resample to 16kHz, downmix to mono.
        # -f wav ... -> force WAV muxing to the temp path, overwrite (-y).
        cmd = [
            _FFMPEG_EXE,
            "-hide_banner",
            "-loglevel", "error",
            "-i", "pipe:0",
            "-ar", str(_TARGET_SR),
            "-ac", "1",
            "-f", "wav",
            "-y",
            ff_path,
        ]
        proc = subprocess.run(
            cmd,
            input=raw_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "ffmpeg failed to decode audio (returncode "
                f"{proc.returncode}): {proc.stderr.decode('utf-8', 'replace')}"
            )

        y, _ = librosa.load(ff_path, sr=_TARGET_SR, mono=True)
    except Exception:
        # Surface the real cause in the server log, then let the caller's
        # try/except fall back to _FALLBACK_RESPONSE.
        traceback.print_exc()
        raise
    finally:
        os.remove(ff_path)

    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(wav_fd)
    sf.write(wav_path, y, _TARGET_SR, subtype="PCM_16")
    return wav_path


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    wav_path = None
    try:
        wav_path = _decode_to_wav(req.audio_base64)

        spoof_score = analyze_audio(wav_path)
        spoof_label = "SPOOF" if spoof_score >= 0.5 else "GENUINE"

        speaker_similarity, speaker_status = verify_speaker(wav_path)

        transcript = transcribe(wav_path)

        return {
            "speaker_similarity": speaker_similarity,
            "speaker_status": speaker_status,
            "spoof_score": spoof_score,
            "spoof_label": spoof_label,
            "transcript": transcript,
        }
    except (binascii.Error, ValueError):
        traceback.print_exc()
        return JSONResponse(content=_FALLBACK_RESPONSE)
    except Exception:
        traceback.print_exc()
        return JSONResponse(content=_FALLBACK_RESPONSE)
    finally:
        if wav_path and os.path.isfile(wav_path):
            os.remove(wav_path)


@app.post("/enroll")
def enroll(req: EnrollRequest):
    wav_path = None
    try:
        wav_path = _decode_to_wav(req.audio_base64)
        ok = enroll_speaker(wav_path)
        return {"enrolled": ok}
    except Exception:
        traceback.print_exc()
        return JSONResponse(content={"enrolled": False})
    finally:
        if wav_path and os.path.isfile(wav_path):
            os.remove(wav_path)
