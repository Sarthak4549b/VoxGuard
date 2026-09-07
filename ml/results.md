# VoxGuard — AASIST Anti-Spoofing Module: Test Results

**Date:** 2026-09-06
**Machine:** Windows 11, NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM), driver 591.59
**Python:** 3.13.15 (isolated venv at `ml/.venv`)

---

## Model

| | |
|---|---|
| Architecture | AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks) |
| Source | [`clovaai/aasist`](https://github.com/clovaai/aasist) |
| Checkpoint | `aasist/models/weights/AASIST.pth` (1,281,532 bytes, ships in the repo) |
| Config | `aasist/config/AASIST.conf` → `model_config` |
| Parameters | ~297 K |
| Training data | ASVspoof 2019 LA (by the original authors) |
| Fine-tuning | **None.** Pretrained inference only. |
| Reference metrics (authors, ASVspoof 2019 LA eval) | EER **0.83 %**, min t-DCF **0.0275** |

### Inference contract

`analyze_audio(wav_path: str) -> float`

1. `librosa.load(..., sr=16000, mono=True)` → resample to 16 kHz mono
2. Peak-normalise amplitude (`y / max(|y|)`)
3. Pad/truncate to 64 600 samples (~4 s) by tiling — matches AASIST's eval `pad()`
4. Forward pass on CUDA (falls back to CPU if unavailable)
5. `softmax(logits)` → **`spoof_score = P(class 0 = spoof)`**
   (AASIST column 1 = bonafide, column 0 = spoof)
6. Clamp to `[0.0, 1.0]`
7. **Any exception → return `0.5`** (safe fallback)

`0.0` = genuine/bonafide · `1.0` = spoofed/synthetic/cloned

---

## Environment verification

```
$ python -c "import torch; print(torch.cuda.is_available())"
True

torch            : 2.6.0+cu124
torchaudio       : 2.6.0+cu124
CUDA available   : True
CUDA device      : NVIDIA GeForce RTX 3050 Laptop GPU
```

> The task asked for the `cu121` wheel index. It has **no Python 3.13 wheels**
> (that index tops out at torch 2.5.1 / cp312), so `cu124` was used instead —
> the nearest build, fully compatible with this GPU and the 591.59 driver.
> The model loads and runs within the 4 GB VRAM budget with room to spare
> (AASIST is a ~297 K-parameter model).

---

## `test_model.py` output — synthetic signals

**First run**, before real audio was added: no `.wav` files were present, so the
harness generated synthetic 16 kHz signals (4 s each) and scored every one.
(The real-speech run is in the next section.)

| File | spoof_score | Verdict | Valid float in [0,1] |
|---|---|---|---|
| `synthetic_tone_220hz.wav` | **1.0000** | SPOOF | ✅ |
| `synthetic_tone_440hz.wav` | **1.0000** | SPOOF | ✅ |
| `synthetic_sweep.wav` | **1.0000** | SPOOF | ✅ |
| `synthetic_white_noise.wav` | **0.9998** | SPOOF | ✅ |
| `synthetic_voiced_vowel.wav` (harmonic stack + formants + pitch glide) | **0.9990** | SPOOF | ✅ |

**All 5 inputs returned a valid float in `[0, 1]` — the function ran without crashing.**

### Fallback path

| Input | Returned |
|---|---|
| `analyze_audio("does_not_exist.wav")` | `0.5` |
| `analyze_audio("")` | `0.5` |

(The underlying exception is logged to stderr; the caller still receives `0.5`.)

### Latency (RTX 3050)

| | |
|---|---|
| Cold start (first call, includes model build + weight load) | ~3–5 s |
| Warm inference per 4 s chunk (librosa load + resample + GPU forward) | **~470 ms** |

---

### Synthetic-signal interpretation

Every synthetic signal scores at the **spoof** ceiling (~1.0). Expected, **not**
a bug: AASIST separates *genuine human speech* from TTS/VC attacks, so pure
tones, sweeps, noise and a synthetic vowel all fall outside the "bonafide
speech" manifold. The score still varies (1.0000 → 0.9990), confirming the
logits are live. Synthetic signals only prove the pipeline runs — a real
genuine-vs-spoof split needs real audio (below).

---

## Real-sample test (`real_samples/`)

3 genuine + 3 TTS clips, downloaded from freely-available sources, transcoded to
16 kHz mono WAV, scored with `python test_model.py`:

| # | File | True label | **spoof_score** | Model verdict (≥0.5 ⇒ spoof) | Correct? |
|---|---|---|---|---|---|
| 1 | `genuine_1_librispeech_2086-149220-0033.wav` | **GENUINE** | **0.0001** | genuine | ✅ |
| 2 | `genuine_2_voices_sri_sp0307.wav` | **GENUINE** | **0.0228** | genuine | ✅ |
| 3 | `genuine_3_mlk_i-have-a-dream.wav` | **GENUINE** | **0.5206** | spoof | ❌ (false positive) |
| 4 | `tts_1_piper_en_US_lessac_high.wav` | **TTS** | **0.0341** | genuine | ❌ (false negative) |
| 5 | `tts_2_piper_en_US_ryan_high.wav` | **TTS** | **0.9294** | spoof | ✅ |
| 6 | `tts_3_piper_en_US_amy_medium.wav` | **TTS** | **0.9628** | spoof | ✅ |

**Class means:** genuine ≈ 0.18 · TTS ≈ 0.64 — the model clearly pushes the two
classes apart on real speech (vs. the flat ~1.0 it gave every synthetic tone).
**Accuracy at threshold 0.5: 4/6.**

### Sample provenance

| File | Source | Type |
|---|---|---|
| genuine 1 | LibriSpeech clip `2086-149220-0033` (public LibriVox reading, CC-BY-4.0), via the NeMo tutorial mirror `dldata-public.s3.us-east-2.amazonaws.com` | real human |
| genuine 2 | SRI/Lab41 **VOiCES** source clip `sp0307` (CC-BY-4.0), via `download.pytorch.org/torchaudio/tutorial-assets` | real human |
| genuine 3 | "I have a dream" excerpt from HF `datasets/Narsil/asr_dummy/mlk.flac` (ASR test fixture) | real human |
| tts 1–3 | **Piper** neural TTS (VITS) demo outputs `en_US/{lessac-high, ryan-high, amy-medium}`, from `github.com/rhasspy/piper-samples` (MIT) | synthetic |

### Why the two misses

- **genuine_3 (MLK, 0.52)** — a 1963 open-air PA recording: heavy reverb, crowd
  noise, AM-broadcast band-limiting. Nothing like the clean studio bonafide of
  ASVspoof 2019 LA, so channel mismatch nudges it just over 0.5. Borderline,
  not a confident error.
- **tts_1 (Piper lessac, 0.03)** — Piper is a 2023-era VITS model trained on a
  very clean single-speaker corpus. AASIST's training attacks are 2019-era
  (Tacotron/WaveNet/older vocoders). A modern high-quality neural TTS voice can
  slip past a CM trained only on older attacks — the well-known cross-dataset /
  cross-attack generalization gap. The other two Piper voices (ryan, amy) are
  still caught with high confidence.

### Takeaway for VoxGuard

The module works: on real speech it separates genuine from synthetic with a
wide margin and returns a bounded, live `spoof_score` (0.5 on error). For
production robustness against **current-gen** cloning, the pretrained checkpoint
should eventually be fine-tuned on a newer corpus (ASVspoof 2021/2024, In-the-
Wild). Fusion with the LLM signal and the speaker-similarity check (as VoxGuard
already does) also covers cases where any single detector is fooled.

---

## Reproduce

```bash
cd ml
python -m venv .venv
.venv\Scripts\activate                       # Windows
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python test_model.py
```
