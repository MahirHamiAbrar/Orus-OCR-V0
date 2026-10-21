# Orus-OCR-0.1

A small, CPU-fast OCR for **Bangla, English and mixed Bangla+English** pages, handwriting first. For each page it returns the language (`bn` / `en` / `bn+en`), whether it is **handwritten** (Yes/No), and the **text with its structure** (line breaks, indentation, blank lines). Text inside photos and charts is ignored.

This repo is a complete, frozen snapshot of Orus-OCR-0.1: the code that produced the model, the released weights, the training recipe, the results and the documentation. Later releases (Orus-OCR-1.0-alpha and beyond) are developed separately and will be published under their own versions.

> Released under MIT (Non-Commercial). See `LICENSE`. Upstream notice: the released `det.onnx` comes from Indra OCR (`datalab-to/indra-ocr`, modified OpenRAIL-M: research and small-company use; commercial use not granted by the upstream). Training used research-only datasets (IAM, BanglaWriting, IMGUR5K, and others); consult each dataset's license before redistribution.

| | |
|---|---|
| Recognizer | 1.06M-parameter CTC line recognizer (Indra OCR 1.1 architecture, warm-started from Indra), 211 characters (Latin + Bengali block + typographic marks) |
| Pipeline size | ≈ 8.4 MB loaded (det 0.39 + rec 4.27 + LM 3.19 + picture net 0.16 + handwritten flag 0.40) |
| Speed (4 CPU threads) | 1.3–11.7 ms per line; 130–430 ms per page, within ~20% of Indra OCR 1.1 |
| Accuracy (line CER) | Bangla handwriting 6.9–16.2% (unseen source 25.8%); English handwriting 12.4–19.0%; printed 4.4–4.6%; prescriptions 6.9% |
| vs Indra OCR 1.1 | Bangla handwriting 6.9% vs 84.6% (BN-HTRd); English handwriting 12.4% vs 18.7% (IAM) |

## Quick start

```bash
python run.py page.jpg              # language, handwritten, text
python run.py page.jpg --json       # + lines with boxes and confidence, timings
python run.py --demo                # self-check (rendered mixed page; expects CER < 5%)
```

```python
import sys; sys.path.insert(0, "orus")
from ocr import OCR
ocr = OCR(release="models")
r = ocr("page.jpg")
print(r["language"], r["handwritten"]); print(r["text"])
for line in r["lines"]:          # bbox, text, conf, lang
    if line["conf"] < 0.8: ...   # unreliable line: conf >= 0.95 has 1-4% CER, < 0.6 has 27-87%
```

Requirements: Python 3.10, `onnxruntime`, `opencv-python`, `numpy`, `rapidfuzz`, `indra-ocr` (pip; provides preprocessing, detector decoding, deskew and the LM beam search), `torch` (only for the handwritten flag's tile code and for training). Inference runs fully on CPU through ONNX Runtime.

## Layout

| Path | What |
|---|---|
| `run.py` | inference CLI (uses `models/`) |
| `train.sh` | full training recipe as stages: `./train.sh data synth lm A B1 B3 distill C pic export bench` |
| `orus/` | frozen code: `ocr.py` (pipeline), `ocr_model.py` (network), `ocr_data.py` (datasets → line stores), `ocr_synth.py` (synthetic lines), `ocr_train.py` (training), `ocr_lm.py`, `ocr_pic.py`, `ocr_distill.py`, `ocr_bench.py`, `ocr_eval.py`, `ocr_errors.py`, `ocr_det.py`, plus `tiles.py`, `train.py`, `render_printed.py`, `build_data.py` from the main project |
| `models/` | released weights + `MD5SUMS` (never written by the scripts) |
| `docs/` | documentation, below |
| `results/` | benchmark / page-evaluation / error-analysis JSON for the released model, Indra and PaddleOCR-VL |

Not in this repo (development-only, intentionally excluded from the public release):

- `bench/` — frozen line test sets (12 sets, pickles). Held privately for now; not shipped with the model.
- `runs/` — intermediate training checkpoints (stages A, B1, B2, B3, C) and their training logs. Reproducible from `train.sh`.
- `data/` — symlink to the main project's raw datasets and line stores. Datasets are research-only and not cleared for redistribution.

## Documentation

1. [docs/01_research_method.md](docs/01_research_method.md) — problem, decisions, hypotheses, data and splits, protocol, metrics, baselines, what was tested and how.
2. [docs/02_technical.md](docs/02_technical.md) — architecture of every component, preprocessing, decoding, layout assembly, file formats, ONNX interface, sizes.
3. [docs/03_benchmarks.md](docs/03_benchmarks.md) — all numbers so far: line and page benchmarks, speed, sizes, comparison with Indra, PaddleOCR-VL and published state of the art; strengths and weaknesses.
4. [docs/04_training.md](docs/04_training.md) — training recipe: data preparation, synthetic data, stages A → B1 → B3 → C, distillation, picture net, LM, export, hardware and timings, how to reproduce.

Earlier working reports (kept as written): `docs/orus_report_v1.md`, `docs/orus_report_v2.md`, `docs/orus_weaknesses.md`, `docs/sota_ocr_wer.md`, `docs/indra_arch.md`.

## Links

- Code on GitHub: [github.com/MahirHamiAbrar/Orus-OCR-V0](https://github.com/MahirHamiAbrar/Orus-OCR-V0)
- Model weights on HuggingFace: [huggingface.co/mhabrar/Orus-OCR-V0](https://huggingface.co/mhabrar/Orus-OCR-V0)

## Licences

Research project. Training used research-only datasets (IAM, BanglaWriting, IMGUR5K, …). `models/det.onnx` and the warm start come from Indra OCR 1.1 (Datalab, modified OpenRAIL-M: research and small-company use). Not cleared for commercial use — see `LICENSE`.
