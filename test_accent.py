"""
test_accent.py -- Evaluate the AASIST spoof detector on Indian-accent TTS clips.

Loads analyze_audio() from C:/VoxGuard/ml/model.py and runs it against the 5
synthetic clips produced by generate_samples.py (samples/indian_tts_1..5.wav).

Every clip is synthetic (TTS), so the model SHOULD return spoof_score >= 0.5
for all of them. We report how many it actually catches -- the detection rate
on Indian-accent / Hinglish synthetic speech.

Usage:
    python test_accent.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ML_DIR = r"C:\VoxGuard\ml"
if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

try:
    from model import analyze_audio
except Exception as exc:  # noqa: BLE001
    print("ERROR: could not import analyze_audio from", os.path.join(ML_DIR, "model.py"))
    print("       ", repr(exc))
    sys.exit(1)

SAMPLES_DIR = os.path.join(HERE, "samples")
SPOOF_THRESHOLD = 0.5
FILES = [f"indian_tts_{i}.wav" for i in range(1, 6)]


def main():
    rows = []
    missing = []
    for name in FILES:
        path = os.path.join(SAMPLES_DIR, name)
        if not os.path.isfile(path):
            missing.append(name)
            continue
        score = analyze_audio(path)
        caught = score >= SPOOF_THRESHOLD
        rows.append((name, score, caught))

    if missing:
        print("WARNING: missing sample files (run generate_samples.py first):")
        for m in missing:
            print("   -", m)
        print()

    if not rows:
        print("No samples to evaluate. Aborting.")
        sys.exit(1)

    # ---- results table -----------------------------------------------------
    print()
    print("VoxGuard -- AASIST spoof detection on Indian-accent TTS samples")
    print("threshold: spoof_score >= {:.2f}  ==>  flagged SPOOF".format(SPOOF_THRESHOLD))
    print()
    header = "{:<20} {:>12} {:>18}".format("filename", "spoof_score", "flagged SPOOF?")
    print(header)
    print("-" * len(header))
    for name, score, caught in rows:
        print("{:<20} {:>12.4f} {:>18}".format(name, score, "YES" if caught else "NO  <-- MISS"))
    print("-" * len(header))

    caught_n = sum(1 for _, _, c in rows if c)
    total_n = len(rows)
    rate = 100.0 * caught_n / total_n
    avg = sum(s for _, s, _ in rows) / total_n

    print()
    print(f"Detected as spoof : {caught_n}/{total_n}")
    print(f"Detection rate    : {rate:.1f}%  (on Indian-accent / Hinglish synthetic voices)")
    print(f"Mean spoof_score  : {avg:.4f}")
    print()
    if caught_n == total_n:
        print("All synthetic clips were flagged. No accent-related misses in this batch.")
    else:
        print(f"{total_n - caught_n} synthetic clip(s) slipped through -- see results_indian_accent.md.")


if __name__ == "__main__":
    main()
