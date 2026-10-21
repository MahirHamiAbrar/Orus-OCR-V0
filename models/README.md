# Orus-OCR-0.1 (frozen release, 2026-10-21)

Small CPU OCR for Bangla, English and mixed Bangla+English pages, handwriting-first. Frozen so it stays usable for tasks that do not need the lower error rates targeted by Orus-OCR-1.0-alpha.

## Use

From the repo root:

```bash
python run.py page.jpg [--json]
```

## Files (`MD5SUMS`)

| File | What | Params |
|---|---|---|
| `det.onnx` | Indra OCR 1.1 line detector, unchanged (datalab-to/indra-ocr, modified OpenRAIL-M: research use) | 90.6k |
| `orus_rec.onnx` | Orus recognizer, model C (fp32; INT8 is not shipped: it loses 7–10 CER points on handwriting) | 1.06M |
| `orus_rec.pt` | same weights, PyTorch checkpoint (`ocr_model.RecNet(**ck["cfg"])`) | |
| `orus_lm.npz` | order-5 character LM (Indra npz format), used on low-confidence lines | |
| `orus_pic.onnx` | picture-region net (drops text inside photos/charts) | 38.4k |
| `v2_teff.onnx` + `v2_teff.json` | handwritten / printed page classifier + threshold | 97.8k |

Total ≈ 8.4 MB loaded (orus_rec.pt is only for retraining).

## Accuracy (line CER %, idle CPU, from docs/orus_report_v2.md)

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

Pages: BN-HTRd page CER 12.8% (Indra 1.1: 88.7%); language ID 96.7% on real mixed pages; handwritten flag 93–100%; 0 lines read from pictures. Speed (4 CPU threads): 1.3–11.7 ms per line, page ≈ 130–430 ms (≈ Indra 1.1).

## Known weaknesses (docs/orus_weaknesses.md)

Similar-letter confusions (o/a, r/n, ব/র, ল/ন), Bangla conjuncts, digits and punctuation (27–71% wrong inside those classes), receipts and dot-matrix print, unseen Bangla handwriting sources. Use the per-line `conf` in `lines`: lines with conf ≥ 0.95 have 1–4% CER; below 0.8 treat as unreliable.

## Code version

The frozen code that matches these weights is `../orus/` (snapshot of 2026-10-21). The main project's code keeps `OCR(release=...)` working too, but may change.
