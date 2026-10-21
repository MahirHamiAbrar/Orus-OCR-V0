# Orus-OCR-0.1

Small, CPU-only OCR for **Bangla, English and mixed Bangla+English** pages, with handwriting as the first target.

For every page it gives you the **language** (`bn`, `en` or `bn+en`), whether the page is **handwritten** (`Yes` / `No`), and the **text with its structure** — line breaks, indentation and blank lines are preserved. Text sitting inside a photo or a chart is dropped before recognition, so it never reaches the output.

*Release date: 2026-10-21.*

## Quick start

The weights in this folder are used from the repo root, where `run.py` sits next to `models/`:

```bash
python run.py page.jpg              # language, handwritten, text
python run.py page.jpg --json       # + lines with boxes and confidence, timings
python run.py --demo                # self-check on a rendered mixed page
```

```python
import sys; sys.path.insert(0, "orus")
from ocr import OCR

ocr = OCR(release="models")     # any folder holding the release weights
r = ocr("page.jpg")

print(r["language"], r["handwritten"])
print(r["text"])

for line in r["lines"]:          # bbox, text, conf, lang
    if line["conf"] < 0.8:       # low confidence: do not trust this line
        ...
```

**Requirements:** Python 3.10, `onnxruntime`, `opencv-python`, `numpy` and `indra-ocr` (preprocessing, detector decoding, deskew and the language-model beam search). The handwritten flag also needs `torch` and `timm`; `rapidfuzz` and `Pillow` are only required for `--demo`. Everything runs on CPU through ONNX Runtime.

```bash
pip install onnxruntime opencv-python numpy indra-ocr torch timm rapidfuzz Pillow
```

## Files

| File | What | Params |
|---|---|---|
| `det.onnx` | line detector from Indra OCR 1.1, unchanged (datalab-to/indra-ocr, modified OpenRAIL-M: research use) | 90.6k |
| `orus_rec.onnx` | the recognizer, fp32 | 1.06M |
| `orus_rec.pt` | the same weights as a PyTorch checkpoint — for retraining only | |
| `orus_lm.npz` | order-5 character LM, applied to low-confidence lines | |
| `orus_pic.onnx` | picture-region net, drops text inside photos and charts | 38.4k |
| `v2_teff.onnx` + `v2_teff.json` | handwritten / printed page classifier and its threshold | 97.8k |

About 8.4 MB in total, and `MD5SUMS` has the checksum of every file.

INT8 is deliberately not shipped: it gives back 7–10 CER points on handwriting and is slower on CPU.

## Accuracy

Character error rate per line, on CPU:

| Set | CER | WER |
|---|---|---|
| BN-HTRd (Bangla handwriting) | 6.9 | 24 |
| BanglaWriting (Bangla handwriting) | 14.7 | 37 |
| IIIT Bangla handwritten words | 16.2 | 52 |
| Rupom (Bangla handwriting, unseen source) | 25.8 | 62 |
| IAM (English handwriting, clean) | 12.4 | 33 |
| GNHK (English handwriting, phone photos) | 13.6 | 42 |
| IMGUR5K (English words in the wild) | 19.0 | 39 |
| Doctor's prescriptions (words) | 6.9 | 23 |
| Mixed Bangla+English (stitched real words) | 16.6 | 41 |
| Printed, held-out fonts (synthetic) | 4.6 | 14 |
| Printed Bangla pages (Swapnil, page CER) | 4.4 | 18 |
| Printed receipts (CORD rows) | 20.5 | 55 |

On full pages, BN-HTRd comes out at 12.8% page CER against Indra OCR 1.1's 88.7%. Language ID reaches 96.7% on real mixed pages, the handwritten flag scores 93–100%, and nothing is read from pictures. With 4 CPU threads a line takes 1.3–11.7 ms and a page roughly 130–430 ms — within about 20% of Indra OCR 1.1.

## Where it struggles

Similar-letter confusions (o/a, r/n, ব/র, ল/ন), Bangla conjuncts, digits and punctuation (27–71% wrong inside those classes), receipts and dot-matrix print, and Bangla handwriting from sources it has never seen.

Every line carries a confidence value: at 0.95 and above the error rate sits at 1–4%, and below 0.8 the line is better treated as unreliable.

## Licence

Research project. Training used research-only datasets (IAM, BanglaWriting, IMGUR5K, …). `det.onnx` and the warm start come from Indra OCR 1.1 (Datalab, modified OpenRAIL-M: research and small-company use). Not cleared for commercial use.
