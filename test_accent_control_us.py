"""
test_accent_control_us.py -- US-English control for the Indian-accent test.

Generates the SAME 5 scam sentences with US-English edge-tts voices
(en-US-GuyNeural / en-US-AriaNeural), writes them to samples/control_us/, and
runs analyze_audio() on each. Lets us tell "the accent broke detection" apart
from "the model just doesn't generalise to modern neural TTS".

Usage:
    python test_accent_control_us.py
"""

import asyncio
import os
import sys

import soundfile as sf

sys.path.insert(0, r"C:\VoxGuard\ml")
from model import analyze_audio  # noqa: E402

try:
    import edge_tts  # noqa: E402
except Exception as exc:  # noqa: BLE001
    print("ERROR: edge-tts not installed. Run:  pip install edge-tts")
    print("       ", repr(exc))
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "samples", "control_us")
US_VOICES = ["en-US-GuyNeural", "en-US-AriaNeural"]
SENTENCES = [
    "Hello sir, this is calling from your bank, please share your OTP number.",
    "Aapka account block ho jayega, turant paisa transfer kariye.",
    "Namaste, main aapki madad karne ke liye call kar raha hoon.",
    "Your electricity bill is pending, pay immediately or connection will be cut.",
    "Beta, mujhe urgent paise chahiye, abhi transfer karo.",
]
SPOOF_THRESHOLD = 0.5


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for idx, text in enumerate(SENTENCES, start=1):
        voice = US_VOICES[(idx - 1) % len(US_VOICES)]
        mp3_path = os.path.join(OUT_DIR, f"_tmp_{idx}.mp3")
        wav_path = os.path.join(OUT_DIR, f"us_tts_{idx}.wav")
        asyncio.run(edge_tts.Communicate(text, voice).save(mp3_path))
        audio, sr = sf.read(mp3_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        sf.write(wav_path, audio, sr, subtype="PCM_16")
        os.remove(mp3_path)
        score = analyze_audio(wav_path)
        rows.append((f"us_tts_{idx}.wav", voice, score, score >= SPOOF_THRESHOLD))

    print()
    print("US-English control -- AASIST on standard-accent neural TTS")
    header = "{:<14} {:<20} {:>12} {:>16}".format("filename", "voice", "spoof_score", "flagged SPOOF?")
    print(header)
    print("-" * len(header))
    for name, voice, score, caught in rows:
        print("{:<14} {:<20} {:>12.4f} {:>16}".format(name, voice, score, "YES" if caught else "NO  <-- MISS"))
    print("-" * len(header))
    caught_n = sum(1 for *_, c in rows if c)
    avg = sum(s for *_, s, _ in rows) / len(rows)
    print()
    print(f"Detected as spoof : {caught_n}/{len(rows)}")
    print(f"Detection rate    : {100.0 * caught_n / len(rows):.1f}%  (standard US-English synthetic voices)")
    print(f"Mean spoof_score  : {avg:.4f}")


if __name__ == "__main__":
    main()
