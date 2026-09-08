# VoxGuard - FastAPI ML service
"""
Serves the AASIST anti-spoofing model (ml/model.py), SpeechBrain speaker
verification (speaker_verify.py) and local faster-whisper transcription
(transcribe.py) behind a single HTTP API on http://localhost:8000.

Endpoints:
    GET  /health       -> { status: 'ok' }
    POST /analyze      -> live path: AASIST + speaker verification + a fast
                          "tiny.en" faster-whisper transcript so fraud
                          keywords reach the LLM scorer in real time
    POST /analyze-full -> /analyze with a larger "base" transcript for the
                          demo upload path (accuracy over latency)
    POST /enroll       -> { enrolled: bool }

/analyze and /analyze-full both return:
    { speaker_similarity, speaker_status, spoof_score, spoof_label,
      spoof_strong, risk_hint, transcript }
  - spoof_score is the raw AASIST score (kept for logging; it saturates on
    compressed browser audio).
  - spoof_strong is True only when spoof_score is near-certain AND the
    speaker is a MISMATCH.
  - risk_hint (LOW / HIGH / NEUTRAL) tells the Node fusion layer how the
    speaker signal should steer escalation.

Never crashes: both always return the documented shape, falling back to
{ speaker_similarity: 0, speaker_status: 'UNKNOWN', spoof_score: 0.5,
  spoof_label: 'GENUINE', spoof_strong: False, risk_hint: 'NEUTRAL',
  transcript: '' } on any internal failure.
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

# NOTE: transcribe (faster-whisper) is imported lazily via _get_transcriber()
# so importing this module never pulls a Whisper model into memory. The model
# itself is loaded (and cached) inside transcribe.py on the first call for a
# given size: "tiny.en" for the live /analyze path, "base" for /analyze-full.
_transcribe = None


def _get_transcriber():
    global _transcribe
    if _transcribe is None:
        from transcribe import transcribe as _t  # noqa: PLC0415

        _transcribe = _t
    return _transcribe


# Model sizes per pipeline path (see transcribe.py).
_LIVE_WHISPER_SIZE = "tiny.en"
_FULL_WHISPER_SIZE = "base"

app = FastAPI(title="VoxGuard ML Service")

_TARGET_SR = 16000

# AASIST over-predicts spoof on compressed browser (WebM/Opus) audio: its raw
# score saturates near 1.0 even for genuine laptop-mic speech. So the raw
# spoof_score is kept only for logging, and spoof is treated as a *strong*
# signal only when it is near-certain (>= this) AND the speaker does not match
# an enrolled user.
_SPOOF_STRONG_THRESHOLD = 0.98


def _derive_signals(spoof_score: float, speaker_status: str) -> tuple[str, bool, str]:
    """Turn the raw AASIST score + speaker status into calibrated signals for
    the Node backend's fusion / escalation step.

    Returns (spoof_label, spoof_strong, risk_hint) where risk_hint is one of
    LOW / HIGH / NEUTRAL:
      - MATCH  (enrolled genuine user speaking): LOW, ignore a high spoof
        score entirely.
      - MISMATCH (different / cloned voice): HIGH -- the real danger signal.
      - UNKNOWN (no enrollment): NEUTRAL -- lean on spoof_score + LLM only,
        never auto-escalate to HIGH from the absent speaker signal alone.
    """
    spoof_strong = (
        spoof_score >= _SPOOF_STRONG_THRESHOLD and speaker_status == "MISMATCH"
    )

    if speaker_status == "MATCH":
        return "GENUINE", False, "LOW"
    if speaker_status == "MISMATCH":
        return ("SPOOF" if spoof_strong else "GENUINE"), spoof_strong, "HIGH"
    # UNKNOWN
    label = "SPOOF" if spoof_score >= _SPOOF_STRONG_THRESHOLD else "GENUINE"
    return label, spoof_strong, "NEUTRAL"


_FALLBACK_RESPONSE = {
    "speaker_similarity": 0,
    "speaker_status": "UNKNOWN",
    "spoof_score": 0.5,
    "spoof_label": "GENUINE",
    "spoof_strong": False,
    "risk_hint": "NEUTRAL",
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
    """Fast path for the live WebSocket pipeline.

    Runs AASIST spoof detection + SpeechBrain speaker verification, plus a
    fast "tiny.en" faster-whisper transcript (greedy decode, VAD-trimmed)
    so fraud keywords (OTP, transfer money, urgent, bank) reach the LLM
    scorer in real time. Target: < ~1.5s per 3s chunk. /analyze-full uses
    the larger "base" model where latency does not matter.
    """
    wav_path = None
    try:
        wav_path = _decode_to_wav(req.audio_base64)

        # raw AASIST score -- kept as-is in the response for logging only
        spoof_score = analyze_audio(wav_path)

        # no enrollment -> (0.0, "UNKNOWN"); enrolled -> MATCH/MISMATCH vs the
        # persisted embedding at the (now 0.60) threshold. UNKNOWN never
        # escalates risk on its own.
        speaker_similarity, speaker_status = verify_speaker(wav_path)

        spoof_label, spoof_strong, risk_hint = _derive_signals(
            spoof_score, speaker_status
        )

        # Fast live transcript. transcribe() swallows its own errors and
        # returns "" on failure, so a Whisper hiccup never breaks /analyze.
        transcript = _get_transcriber()(
            wav_path, model_size=_LIVE_WHISPER_SIZE, fast=True
        )

        return {
            "speaker_similarity": speaker_similarity,
            "speaker_status": speaker_status,
            "spoof_score": spoof_score,
            "spoof_label": spoof_label,
            "spoof_strong": spoof_strong,
            "risk_hint": risk_hint,
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


@app.post("/analyze-full")
def analyze_full(req: AnalyzeRequest):
    """Full pipeline for Demo Upload Mode where latency does not matter.

    Same input/output shape as /analyze, but transcribes with the larger
    "base" faster-whisper model and beam search for better accuracy. That
    model is lazy-loaded and cached separately from the live "tiny.en" one
    (see transcribe.py), so the two paths never evict each other.
    """
    wav_path = None
    try:
        wav_path = _decode_to_wav(req.audio_base64)

        spoof_score = analyze_audio(wav_path)

        speaker_similarity, speaker_status = verify_speaker(wav_path)

        spoof_label, spoof_strong, risk_hint = _derive_signals(
            spoof_score, speaker_status
        )

        transcript = _get_transcriber()(
            wav_path, model_size=_FULL_WHISPER_SIZE, fast=False
        )

        return {
            "speaker_similarity": speaker_similarity,
            "speaker_status": speaker_status,
            "spoof_score": spoof_score,
            "spoof_label": spoof_label,
            "spoof_strong": spoof_strong,
            "risk_hint": risk_hint,
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
