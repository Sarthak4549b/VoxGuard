# VoxGuard - AASIST inference smoke test
#
# Runs analyze_audio() against any .wav files found near this script. If none
# exist, it synthesises a few test signals (tones + noise) so the inference
# path is always exercised. Prints the spoof_score for every file.

import glob
import os
import wave

import numpy as np

from model import analyze_audio

_HERE = os.path.dirname(os.path.abspath(__file__))
_SR = 16000


def _write_wav(path: str, samples: np.ndarray, sr: int = _SR) -> None:
    samples = np.clip(samples, -1.0, 1.0)
    pcm16 = (samples * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())


def _make_synthetic(out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    dur = 4.0
    t = np.linspace(0, dur, int(_SR * dur), endpoint=False)
    rng = np.random.default_rng(42)

    # a crude "voiced vowel": harmonic stack under two formants, with a
    # slow pitch glide, vibrato and an amplitude envelope. Still not real
    # speech, but closer to it than a pure tone.
    f0 = 120 + 15 * np.sin(2 * np.pi * 5 * t) + 10 * (t / dur)
    phase = 2 * np.pi * np.cumsum(f0) / _SR
    vowel = np.zeros_like(t)
    for k, gain in enumerate([1.0, 0.6, 0.4, 0.25, 0.15, 0.1], start=1):
        vowel += gain * np.sin(k * phase)
    formant = 1.0 + 0.8 * np.sin(2 * np.pi * 700 * t) + 0.5 * np.sin(
        2 * np.pi * 1220 * t
    )
    env = np.clip(np.sin(np.pi * ((t * 2.5) % 1.0)), 0, None)
    vowel = 0.5 * (vowel / 6.0) * formant * env

    signals = {
        "synthetic_tone_220hz.wav": 0.6 * np.sin(2 * np.pi * 220 * t),
        "synthetic_tone_440hz.wav": 0.6 * np.sin(2 * np.pi * 440 * t),
        "synthetic_sweep.wav": 0.5
        * np.sin(2 * np.pi * (200 + 300 * t / dur) * t),
        "synthetic_white_noise.wav": 0.3 * rng.standard_normal(t.shape[0]),
        "synthetic_voiced_vowel.wav": vowel.astype(np.float32),
    }

    paths = []
    for name, sig in signals.items():
        p = os.path.join(out_dir, name)
        _write_wav(p, sig.astype(np.float32))
        paths.append(p)
    return paths


def _find_wavs() -> list[str]:
    patterns = [
        os.path.join(_HERE, "*.wav"),
        os.path.join(_HERE, "real_samples", "*.wav"),
        os.path.join(_HERE, "samples", "*.wav"),
        os.path.join(_HERE, "audio", "*.wav"),
        os.path.join(_HERE, "test_audio", "*.wav"),
    ]
    found: list[str] = []
    for pat in patterns:
        found.extend(sorted(glob.glob(pat)))
    return found


def main() -> None:
    import torch  # local import: just for the environment banner

    print("=" * 60)
    print("VoxGuard AASIST inference test")
    print(f"  torch            : {torch.__version__}")
    print(f"  CUDA available   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device      : {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    wavs = _find_wavs()
    if wavs:
        print(f"Found {len(wavs)} .wav file(s).\n")
    else:
        print("No .wav files found - generating synthetic test signals.\n")
        wavs = _make_synthetic(os.path.join(_HERE, "test_audio"))

    results = []
    for path in wavs:
        score = analyze_audio(path)
        ok = isinstance(score, float) and 0.0 <= score <= 1.0
        label = "SPOOF" if score >= 0.5 else "GENUINE"
        print(
            f"  {os.path.basename(path):<32} "
            f"spoof_score={score:.4f}  ({label})  "
            f"{'OK' if ok else 'INVALID'}"
        )
        results.append((path, score, ok))

    print()
    all_valid = all(ok for _, _, ok in results)
    print(
        f"{len(results)} file(s) scored. "
        f"All returned a valid float in [0,1]: {all_valid}"
    )
    if not all_valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
