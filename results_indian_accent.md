# VoxGuard — AASIST on Indian-Accent Synthetic Voices (Honest Limitations)

**Model under test:** pretrained AASIST checkpoint (`clovaai/aasist`), inference only,
no fine-tuning. Trained/evaluated on **ASVspoof 2019 LA** (2019-era vocoders & TTS).
**Entry point:** `ml/model.py::analyze_audio(wav) -> spoof_score` (0.0 genuine … 1.0 spoof).
**Decision threshold:** `spoof_score >= 0.5` → flagged **SPOOF**.

## How the test data was made

- `generate_samples.py` — 5 scam-style clips, Indian-English + Hinglish (romanised
  Hindi), synthesised with **Microsoft edge-tts** neural voices
  `en-IN-PrabhatNeural` (m) and `en-IN-NeerjaNeural` (f). These are *native
  Indian-English* voices — a real accent, not a US/UK fallback.
- Coqui TTS was the original plan but its `TTS` package caps at Python ≤ 3.11
  (this env is 3.13) and its public zoo has **no Indian-accent English model**
  anyway. edge-tts installs on 3.13, needs no GPU/API key, and gives a genuine
  Indian accent — so we used it instead.
- All 5 clips are 100 % neural TTS → the detector *should* flag every one.
- `test_accent_control_us.py` re-synthesises the **same 5 sentences** with
  US-English voices (`en-US-GuyNeural` / `en-US-AriaNeural`) as a control.

## Result — Indian-accent TTS

| filename           | voice                 | spoof_score | flagged SPOOF? |
|--------------------|-----------------------|------------:|:--------------:|
| indian_tts_1.wav   | en-IN-PrabhatNeural   |   0.9986    | YES            |
| indian_tts_2.wav   | en-IN-NeerjaNeural    |   0.0003    | **NO — miss**  |
| indian_tts_3.wav   | en-IN-PrabhatNeural   |   0.9992    | YES            |
| indian_tts_4.wav   | en-IN-NeerjaNeural    |   0.3758    | **NO — miss**  |
| indian_tts_5.wav   | en-IN-PrabhatNeural   |   0.9998    | YES            |

**Detection rate: 3 / 5 = 60 %.**  Mean spoof_score 0.6747.

## Control — same sentences, US-English TTS

| filename        | voice             | spoof_score | flagged SPOOF? |
|-----------------|-------------------|------------:|:--------------:|
| us_tts_1.wav    | en-US-GuyNeural   |   0.0086    | NO — miss      |
| us_tts_2.wav    | en-US-AriaNeural  |   0.0427    | NO — miss      |
| us_tts_3.wav    | en-US-GuyNeural   |   0.9645    | YES            |
| us_tts_4.wav    | en-US-AriaNeural  |   0.0254    | NO — miss      |
| us_tts_5.wav    | en-US-GuyNeural   |   0.0006    | NO — miss      |

**Detection rate: 1 / 5 = 20 %.**  Mean spoof_score 0.2084.

## Observations

1. **The accent did *not* reduce accuracy.** The Indian-accent set actually scored
   *higher* (60 % vs 20 %, mean 0.67 vs 0.21) than the identical sentences in
   US-English. The real weakness is upstream of accent: this pretrained AASIST
   checkpoint **does not generalise to 2024-era neural TTS** (Microsoft Edge
   voices). It was trained on ASVspoof 2019 LA; modern neural vocoders are
   out-of-distribution for it, Indian or not.

2. **Scores are polarised / bimodal.** Outputs cluster near 0.000 or near 0.999,
   with almost nothing in between (only `indian_tts_4` at 0.38 was genuinely
   borderline). When the model is wrong, it is *confidently* wrong — e.g.
   `indian_tts_2` at 0.0003. A confidence-based abstain band would not have
   caught these misses.

3. **The split tracks the TTS voice, not the language.** All three
   `en-IN-PrabhatNeural` clips were caught at > 0.998; both `en-IN-NeerjaNeural`
   clips were missed — and this cut across English (`indian_tts_4`) and Hinglish
   (`indian_tts_2`) alike. In the control, `en-US-GuyNeural` was caught 1/3 and
   `en-US-AriaNeural` 0/2. Detection here is driven by speaker timbre / vocoder
   artefacts, not by whether the text is Hindi-mixed.

4. **Small sample.** n = 5 per condition, 2 distinct voices each. Treat the
   percentages as directional, not precise. High per-clip variance means the CI
   on "60 %" is wide.

## Honest statement for the judges

> Our anti-spoof model is a **pretrained AASIST checkpoint used as-is**, trained on
> the 2019 ASVspoof dataset. On a small set of **Indian-accented / Hinglish
> synthetic voices** it caught **3 of 5 (60 %)**. Importantly, an identical
> US-English control scored **worse (1 of 5)** — so this is **not an accent bias**;
> it is a **generalisation gap to modern neural TTS**, which post-dates the
> model's training data. When the model errs it does so with high confidence, so
> a single-model score is not safe to rely on.
>
> **What we do about it in VoxGuard:** AASIST is one input to a fusion layer that
> also uses a Claude-based linguistic/scam-intent scorer and speaker
> verification, so a confident false-negative from AASIST alone does not clear a
> call. **Next steps:** fine-tune AASIST on in-the-wild + modern-TTS + Indian-accent
> data (e.g. ASVspoof 5, in-the-wild deepfake sets, self-generated edge-tts /
> Coqui / ElevenLabs samples), recalibrate the threshold on that data, and report
> per-accent detection rates with proper confidence intervals.

## Reproduce

```
pip install edge-tts
python generate_samples.py            # writes samples/indian_tts_1..5.wav
python test_accent.py                 # prints the Indian-accent table above
python test_accent_control_us.py      # prints the US-English control table
```

_Environment: Windows 11, Python 3.13, torch 2.6.0+cu124, AASIST weights at
`ml/aasist/models/weights/AASIST.pth`. Generated 2026-09-08._
