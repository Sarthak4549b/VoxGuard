"""
generate_samples.py -- VoxGuard "honest limitations" test-data generator.

Generates 5 synthetic Indian-accented / Hinglish scam-style voice clips and
saves them to:

    samples/indian_tts_1.wav ... samples/indian_tts_5.wav

TTS engine
----------
Uses Microsoft **edge-tts** (free, no API key, no GPU). Unlike Coqui's public
zoo -- which has no Indian-accent English model -- edge-tts ships native
Indian-English neural voices (en-IN-NeerjaNeural, en-IN-PrabhatNeural), so the
clips carry a genuine Indian accent rather than an American/British fallback.
edge-tts also installs cleanly on Python 3.13 (Coqui `TTS` caps at 3.11).

Every clip is still 100% synthetic (neural TTS) -- which is exactly what the
AASIST spoof detector is meant to flag.

Pipeline: edge-tts -> temp .mp3 -> decode with soundfile -> mono 16-bit PCM .wav

Usage:
    python generate_samples.py
"""

import asyncio
import os
import sys

import soundfile as sf

try:
    import edge_tts
except Exception as exc:  # noqa: BLE001
    print("ERROR: edge-tts not installed. Run:  pip install edge-tts")
    print("       ", repr(exc))
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "samples")

# Indian-English neural voices (male/female) alternated for speaker variety.
# Romanised Hindi ("Aapka account block ho jayega") is exactly how code-mixed
# Hinglish is written, and these voices read it with a natural Indian accent.
INDIAN_VOICES = ["en-IN-PrabhatNeural", "en-IN-NeerjaNeural"]

# (sentence, short tag) -- Hinglish / Indian-English scam-style prompts.
SENTENCES = [
    ("Hello sir, this is calling from your bank, please share your OTP number.", "english"),
    ("Aapka account block ho jayega, turant paisa transfer kariye.", "hinglish"),
    ("Namaste, main aapki madad karne ke liye call kar raha hoon.", "hinglish"),
    ("Your electricity bill is pending, pay immediately or connection will be cut.", "english"),
    ("Beta, mujhe urgent paise chahiye, abhi transfer karo.", "hinglish"),
]


async def _synth(text: str, voice: str, mp3_path: str) -> None:
    await edge_tts.Communicate(text, voice).save(mp3_path)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"TTS engine : edge-tts (Microsoft neural voices)")
    print(f"Voices     : {', '.join(INDIAN_VOICES)}  (native Indian-English accent)")
    print(f"Output dir : {OUT_DIR}")
    print()

    for idx, (text, tag) in enumerate(SENTENCES, start=1):
        voice = INDIAN_VOICES[(idx - 1) % len(INDIAN_VOICES)]
        wav_path = os.path.join(OUT_DIR, f"indian_tts_{idx}.wav")
        mp3_path = os.path.join(OUT_DIR, f"_tmp_{idx}.mp3")

        asyncio.run(_synth(text, voice, mp3_path))

        # decode mp3 -> mono float -> 16-bit PCM wav (AASIST resamples to 16k itself)
        audio, sr = sf.read(mp3_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        sf.write(wav_path, audio, sr, subtype="PCM_16")
        os.remove(mp3_path)

        dur = len(audio) / float(sr)
        print(f"[{idx}/5] {wav_path}")
        print(f"        voice={voice}  type={tag}  {dur:0.1f}s @ {sr} Hz")
        print(f"        text : {text}")

    print()
    print("=" * 72)
    print("DONE. 5 synthetic Indian-accent clips written to:", OUT_DIR)
    print("Engine     : edge-tts  (native Indian-English voices -- real accent,")
    print("             not a US/UK fallback).")
    print("All clips are neural TTS -> the spoof detector should flag every one.")
    print("=" * 72)


if __name__ == "__main__":
    main()
