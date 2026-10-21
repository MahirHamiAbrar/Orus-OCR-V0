# 3. Benchmarks, accuracy, strengths and weaknesses — Orus-OCR-0.1-alpha

All numbers come from `results/*.json`, produced by `orus/ocr_bench.py`, `orus/ocr_eval.py` and `orus/ocr_errors.py` on the frozen test sets in `bench/`. Protocol: `01_research_method.md` §1.5. Released model = stage **C**, fp32 ONNX, CPU, 4 threads, on an idle machine.

## 3.1 Line level: Orus vs Indra OCR 1.1 vs PaddleOCR-VL-1.6

PaddleOCR-VL ran on the GPU with 20 lines per set, so its numbers are coarse. CER above 100% means it wrote more wrong text than the reference holds: on Bangla handwriting it invents Arabic, Devanagari or Hangul text or loops.
**CER %** (lines per set: Orus C (released) 300, Indra 1.1 300, PaddleOCR-VL-1.6 20)

| set | Orus C (released) | Indra 1.1 | PaddleOCR-VL-1.6 |
|---|---|---|---|
| bnhtrd | 6.9 | 84.6 | 72.1 |
| banglawriting | 14.7 | 91.4 | 121.3 |
| iiit | 16.2 | 98.7 | 117.7 |
| rupom | 25.8 | 96.8 | 107.3 |
| gnhk | 13.6 | 20.7 | 18.9 |
| iam | 12.4 | 18.7 | 10.1 |
| imgur5k | 19.0 | 35.2 | 14.8 |
| prescription | 6.9 | 53.4 | 27.8 |
| mixed | 16.6 | 72.6 | 125.6 |
| font | 4.6 | 25.2 | 11.4 |
| bnprinted | 24.3 | 69.8 | – |
| cord_en | 20.5 | 21.9 | – |
| **mean BN hand** | 15.9 | 92.9 | 104.6 |
| **mean EN hand** | 13.0 | 32.0 | 17.9 |

**WER %** (lines per set: Orus C (released) 300, Indra 1.1 300, PaddleOCR-VL-1.6 20)

| set | Orus C (released) | Indra 1.1 | PaddleOCR-VL-1.6 |
|---|---|---|---|
| bnhtrd | 23.5 | 98.4 | 94.6 |
| banglawriting | 37.4 | 98.6 | 128.7 |
| iiit | 51.7 | 107.0 | 115.0 |
| rupom | 62.3 | 100.4 | 127.9 |
| gnhk | 42.1 | 53.7 | 44.9 |
| iam | 33.5 | 49.2 | 37.2 |
| imgur5k | 39.3 | 57.7 | 25.0 |
| prescription | 22.9 | 93.4 | 75.0 |
| mixed | 40.8 | 91.7 | 121.1 |
| font | 14.0 | 41.9 | 21.6 |
| bnprinted | 44.2 | 86.3 | – |
| cord_en | 55.5 | 59.7 | – |
| **mean BN hand** | 43.7 | 101.1 | 116.6 |
| **mean EN hand** | 34.5 | 63.5 | 45.5 |

**ms / line** (lines per set: Orus C (released) 300, Indra 1.1 300, PaddleOCR-VL-1.6 20)

| set | Orus C (released) | Indra 1.1 | PaddleOCR-VL-1.6 |
|---|---|---|---|
| bnhtrd | 6.5 | 25.9 | 3881.3 |
| banglawriting | 11.8 | 22.5 | 26230.0 |
| iiit | 4.5 | 10.7 | 3162.5 |
| rupom | 10.8 | 12.5 | 9078.2 |
| gnhk | 5.9 | 17.9 | 2236.8 |
| iam | 9.2 | 17.1 | 3477.1 |
| imgur5k | 1.8 | 8.0 | 545.9 |
| prescription | 0.9 | 8.2 | 782.0 |
| mixed | 8.7 | 22.4 | 5728.5 |
| font | 3.0 | 13.8 | 4000.1 |
| bnprinted | 7.3 | 20.6 | – |
| cord_en | 4.8 | 12.0 | – |

Bangla mean = bnhtrd, banglawriting, iiit, rupom. English mean = gnhk, iam, imgur5k, prescription.

`bnprinted` and `cord_en` have test-set problems that inflate the error (§3.8), so treat those two rows as upper bounds.

## 3.2 Training stages (ablation, line CER %)

A, B1, B2 and B3 were run in PyTorch on the GPU and C on CPU ONNX; this table compares accuracy only. Stage details are in `04_training.md`.

**CER %** (lines per set: A 300, B1 300, B2 scratch 2.77M 300, B3 300, C 300)

| set | A | B1 | B2 scratch 2.77M | B3 | C |
|---|---|---|---|---|---|
| bnhtrd | 7.4 | 6.9 | 9.6 | 7.0 | 6.9 |
| banglawriting | 16.5 | 15.1 | 21.2 | 14.7 | 14.7 |
| iiit | 39.3 | 16.9 | 21.6 | 16.4 | 16.2 |
| rupom | 29.1 | 26.5 | 33.0 | 25.6 | 25.8 |
| gnhk | 14.0 | 13.5 | 17.0 | 13.6 | 13.6 |
| iam | 13.7 | 12.8 | 18.3 | 12.8 | 12.4 |
| imgur5k | 21.7 | 21.4 | 24.8 | 19.9 | 19.0 |
| prescription | 7.2 | 7.2 | 8.6 | 7.0 | 6.9 |
| mixed | 32.8 | 16.4 | 21.9 | 16.6 | 16.6 |
| font | 5.2 | 4.6 | 6.8 | 4.2 | 4.6 |
| **mean BN hand** | 23.1 | 16.3 | 21.3 | 15.9 | 15.9 |
| **mean EN hand** | 14.2 | 13.8 | 17.2 | 13.4 | 13.0 |

- **A → B1:** adding 82.5k IIIT Bangla words and 200k stitched lines fixed IIIT (39.3 → 16.9) and mixed (32.8 → 16.4).
- **B2:** a 2.77M model from scratch is worse than the 1.06M Indra-initialised model on every set, after a similar number of steps.
- **B3 / C:** small further gains. Distillation in C helps the English word sets (IMGUR5K 19.9 → 19.0, IAM 12.8 → 12.4).

## 3.3 Quantization (line CER %, CPU)

| Set | fp32 (shipped) | dynamic INT8 |
|---|---|---|
| BN-HTRd | 6.9 | 7.1 |
| IIIT | 16.2 | 26.7 |
| GNHK | 13.6 | 22.6 |
| IAM | 12.4 | 21.5 |
| IMGUR5K | 19.0 | 28.4 |
| CPU ms/line | 1.3–11.7 | 3.7–20.6 |

INT8 is smaller (1.21 MB vs 4.27 MB) but both less accurate and slower: ONNX Runtime's `ConvInteger` is slow for depthwise convolutions. Not shipped.

## 3.4 Page level (30 pages per set, CPU, 4 threads, idle)

**bnhtrd** (30 pages)

| system | page CER % | CER w/ newlines % | line-count err % | lang acc % | handwritten acc % | ms p50 | ms p95 |
|---|---|---|---|---|---|---|---|
| Orus C fp32 | 12.8 | 12.9 | 3.5 | 96.7 | 100.0 | 345.3 | 452.9 |
| Indra 1.1 | 88.7 | 89.0 | 33.9 | 0.0 | – | 332.7 | 434.0 |

**banglawriting** (30 pages)

| system | lang acc % | handwritten acc % | ms p50 | ms p95 |
|---|---|---|---|---|
| Orus C fp32 | 100.0 | 100.0 | 279.3 | 356.6 |
| Indra 1.1 | 0.0 | – | 271.4 | 332.8 |

**gnhk** (30 pages)

| system | lang acc % | handwritten acc % | ms p50 | ms p95 |
|---|---|---|---|---|
| Orus C fp32 | 100.0 | 93.3 | 258.2 | 496.9 |
| Indra 1.1 | 100.0 | – | 211.4 | 380.6 |

**cmater_mixed** (30 pages)

| system | lang acc % | handwritten acc % | ms p50 | ms p95 |
|---|---|---|---|---|
| Orus C fp32 | 96.7 | 100.0 | 432.8 | 584.6 |
| Indra 1.1 | 43.3 | – | 443.0 | 590.0 |

**swapnil_bn_print** (30 pages)

| system | page CER % | CER w/ newlines % | line-count err % | lang acc % | handwritten acc % | ms p50 | ms p95 |
|---|---|---|---|---|---|---|---|
| Orus C fp32 | 4.4 | 4.4 | 1.9 | 90.0 | 93.3 | 277.0 | 346.4 |
| Indra 1.1 | 4.4 | 4.4 | 2.7 | 90.0 | – | 266.8 | 319.9 |

**cord_en_print** (30 pages)

| system | lang acc % | handwritten acc % | ms p50 | ms p95 |
|---|---|---|---|---|
| Orus C fp32 | 96.7 | 96.7 | 133.4 | 213.6 |
| Indra 1.1 | 100.0 | – | 162.8 | 252.0 |

**synth_pictures** (30 pages)

| system | ms p50 | ms p95 | lines on pictures |
|---|---|---|---|
| Orus C fp32 | 203.1 | 289.4 | 0 (of 402 lines) |
| Indra 1.1 | 204.2 | 327.7 | 34 (of 520 lines) |

- **Per-stage Orus page time (median):** detect 62–94 ms, recognize 29–265 ms, handwritten flag 37–141 ms.
- **Against Indra:** within about ±20% of Indra's page time on every set, while also computing the handwritten flag, which Indra does not.
- **Page CER vs line CER on BN-HTRd:** 12.8% vs 6.9%. The gap is detection (split, merged and missed lines; line-count error 3.5%) and reading order.
- **Swapnil language "misses":** the 3 misses (90%) are pages whose own ground truth contains English (`bn+en`), so corrected Orus language accuracy there is 100%.

## 3.5 Size

| | Params | Size on disk |
|---|---|---|
| Orus-OCR-0.1-alpha pipeline | ≈1.29M | ≈8.4 MB |
| Indra OCR 1.1 default set (det, rec, rec_world, script, order, orient, 2 LMs) | ≈3.5M | ≈41.9 MB |
| PaddleOCR-VL-1.6 / LightOnOCR-2-1B | 0.9B / 1B | ~2 GB (bf16) |

## 3.6 Against the published state of the art (`docs/sota_ocr_cer_wer.md`)

| Group | Orus-OCR-0.1-alpha | Best published (model, params) | Realistic target for a 1–10M CPU model |
|---|---|---|---|
| English handwriting, IAM lines | CER 12.4 / WER 33 | CER 2.26 / WER 4.81–6.55 (DRetHTR, 107M); small: CER 3.2 / WER 10.5 (Gated CRNN, 0.75M, with LM) | CER ≤ 5, WER ≤ 11–15 |
| Bangla handwriting | BN-HTRd lines CER 6.9 / WER 24; BN-HTRd pages CER 12.8; IIIT words CER 16.2 / WER 52 | BN-HTRd pages CER 6.19 / WER 14.2 (GraDeT-HTR, 87M); IIIT words CER 2.35 / WER 12.3 | CER ≤ 6, WER ≤ 18 |
| Printed English | font lines 4.6; receipts 20.5 | CER 0.1–0.4 on clean book lines | CER ≤ 1, WER ≤ 3 |
| Printed Bangla | pages 4.4 / WER 18 | CER 0.68 / WER 3.35 (clean rendered); real pages: Gemini 2.5 Flash CER 4.0 / WER 11.3 | CER ≤ 2, WER ≤ 8 |

On real BN-HTRd pages, Orus (12.8% page CER at 1.3M parameters) is better than Gemini 2.5 Flash (19.4%), but about twice the 87M GraDeT-HTR (6.19%).

## 3.7 Strengths

- **Bangla handwriting** at 1M parameters: 6.9% CER on BN-HTRd lines. Indra OCR 1.1 gets 84.6% and PaddleOCR-VL-1.6 72.1%.
- **Mixed Bangla+English** in one line (16.6% vs 72.6% Indra). Language ID is 96.7% on real mixed pages it has never seen (Indra's output: 43.3%).
- **Doctors' prescriptions**: 6.9% CER on drug-name words (Indra 53.4%, PaddleOCR-VL 27.8%).
- **Speed and size:** faster than Indra per line on CPU (1.3–11.7 vs 6–28 ms), same page time, 8.4 MB in total; about 300–2000× faster per line than the VLMs.
- **Pictures are ignored:** 0 of 402 output lines on photos and charts.
- **Handwritten flag:** 93–100% per set.
- **Usable confidence:** lines with conf ≥ 0.95 have 1–4% CER (0.8% on print). A caller can keep confident lines and route the rest elsewhere.
- **No hallucination:** a CTC recognizer cannot invent text that is not on the line, unlike the VLMs.

## 3.8 Weaknesses (from `ocr_errors.py`; detail in `docs/orus_weaknesses.md`)

1. **Look-alike letters** cause 52–64% of all edits:
   - English: `o/a`, `a/e`, `r/n`, `m/n`, `s/r`, `t/l`;
   - Bangla: `ব/র` (the dot under র), `ল/ন`, `ণ/ন`, `য/র`, `ম/স`.

   Likely causes: the 32 px line height, and no sequence model over the whole line.
2. **Bangla conjuncts** (hasanta `্`): 9–28% wrong within the class; `্` is often read as `া`.
3. **Digits and punctuation:** 27–71% wrong within those classes (GNHK digits 31%, BN-HTRd 37%, rupom 71%); there is little real handwritten digit and punctuation data.
4. **Skipped characters:** 26–39% of edits are deletions, especially in long, tight cursive lines (IAM 60+ characters: 24% CER).
5. **A few bad lines cause most of the error:** the worst 10% of lines cause 26–69% of all errors (bad crops, ambiguous ink, label noise).
6. **Unseen writers and sources:** rupom 25.8%, BanglaWriting 14.7%.
7. **Receipts and non-book print:** dot-matrix and thermal fonts, skewed photos (CORD 20.5%, about the same as Indra).
8. **WER is 2.5–3.5× CER** on handwriting, because one wrong character makes the whole word wrong.
9. **Page-level losses on top of line errors** come from the unchanged Indra detector.
10. **Test-set issues** that inflate two numbers:
    - `bnprinted`: many images hold 2–4 lines under a one-line label;
    - `cord_en`: rows from skewed photos also contain the neighbouring row.

## 3.9 Where 0.1-alpha is fit for use

- Bangla or mixed handwriting, where Indra and the VLMs fail.
- Searchable text and indexing.
- Language and handwritten triage.
- Prescription drug-name capture with human review.
- Anything where per-line confidence can route hard lines to a person or a bigger model.

Not fit for: unattended transcription that needs CER < 5% on handwriting, receipts, or digit-heavy content such as amounts, phone numbers or arithmetic. Those are the targets of Orus-OCR-1.0-alpha.
