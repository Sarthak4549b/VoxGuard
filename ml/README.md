# VoxGuard — ML (AASIST anti-spoofing)

Pretrained [AASIST](https://github.com/clovaai/aasist) inference. One entry point:

```python
from model import analyze_audio
score = analyze_audio("clip.wav")   # 0.0 = genuine ... 1.0 = spoof; 0.5 on error
```

## Setup

```bash
cd ml
python -m venv .venv
.venv\Scripts\activate                        # Windows  (source .venv/bin/activate on *nix)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # -> True
```

> `cu124`, not `cu121`: the cu121 wheel index has no Python 3.13 build. cu124 is
> the nearest and works with the RTX 3050 + driver 591.59. Use
> `.../whl/cpu` for a CPU-only machine.

## Test

```bash
python test_model.py
```

Scores any `.wav` in `ml/`, `ml/real_samples/`, `ml/samples/`, `ml/audio/` or
`ml/test_audio/`. If none exist it synthesises signals so the inference path
always runs. `real_samples/` holds 3 genuine + 3 TTS clips — see `results.md`
for their scores and provenance.

## Files

| Path | Purpose |
|---|---|
| `model.py` | `analyze_audio()` — load → 16 kHz mono → AASIST → `spoof_score` |
| `test_model.py` | Smoke test / score dump |
| `aasist/` | Vendored `clovaai/aasist` (architecture in `models/AASIST.py`, weights in `models/weights/AASIST.pth`) |
| `requirements.txt` | Pinned deps (install torch separately from the CUDA index) |
| `results.md` | Test scores + environment notes |
