# Orus-OCR-0.1-alpha — evaluation report v2 (after distillation, idle-machine timing)

Date: 2026-10-21. Same machine and test sets as [report v1](orus_report_v1.md), plus two printed line sets (bnprinted, cord_en). Weakness analysis: [orus_weaknesses.md](orus_weaknesses.md).

## 1. What changed since v1

| Step | Result |
|---|---|
| B3: B1 trained 30k more steps at lower LR | mean real val CER 15.0% → 14.7% |
| Distillation: PaddleOCR-VL-1.6 teacher on 1,200 English lines of unlabeled real pages (doctor notes, NoTeS-Bank, reddit notes); 503 lines kept where teacher and Orus agree within 30% CER | stage C = B3 + those lines (3% sampling weight), 12k steps: val 14.65% |
| Effect of distillation on test lines (C vs B3) | IMGUR5K 19.9 → 19.0, IAM 12.8 → 12.4, prescription 7.0 → 6.9; Bangla unchanged (teacher fails on Bangla, so no Bangla was distilled) |
| **INT8 dropped as default** | Dynamic INT8 (ConvInteger) costs 7–10 CER points on handwriting (IAM 12.4 → 21.5, GNHK 13.6 → 22.6, IIIT 16.2 → 26.7) and is slower on CPU. The v1 "99.6% argmax agreement" check used random inputs and missed it. The pipeline now uses the fp32 ONNX (4.27 MB). All v1 page numbers were INT8 and therefore pessimistic. |
| Timing | All numbers in this report were taken on an otherwise idle machine (no GPU/CPU jobs running). |

Distillation verdict: it works but the effect is small because only 503 lines passed the agreement filter and the teacher is useless for Bangla. A bigger effect needs a teacher that reads Bangla handwriting (none of the tested VLMs does) or many more English pages.

## 2. Line-level results (Orus C = fp32 ONNX on CPU, 4 threads; Indra on CPU, 4 threads; both on an idle machine)

**CER %** (lines per set: Orus C fp32 CPU 300, Orus B1 300, Indra 1.1 CPU 300, PaddleOCR-VL-1.6 20)

| set | Orus C fp32 CPU | Orus B1 | Indra 1.1 CPU | PaddleOCR-VL-1.6 |
|---|---|---|---|---|
| bnhtrd | 6.9 | 6.9 | 84.6 | 72.1 |
| banglawriting | 14.7 | 15.1 | 91.4 | 121.3 |
| iiit | 16.2 | 16.9 | 98.7 | 117.7 |
| rupom | 25.8 | 26.5 | 96.8 | 107.3 |
| gnhk | 13.6 | 13.5 | 20.7 | 18.9 |
| iam | 12.4 | 12.8 | 18.7 | 10.1 |
| imgur5k | 19.0 | 21.4 | 35.2 | 14.8 |
| prescription | 6.9 | 7.2 | 53.4 | 27.8 |
| mixed | 16.6 | 16.4 | 72.6 | 125.6 |
| font | 4.6 | 4.6 | 25.2 | 11.4 |
| bnprinted | 24.3 | – | 69.8 | – |
| cord_en | 20.5 | – | 21.9 | – |
| **mean BN hand** | 15.9 | 16.3 | 92.9 | 104.6 |
| **mean EN hand** | 13.0 | 13.8 | 32.0 | 17.9 |

**WER %** (lines per set: Orus C fp32 CPU 300, Orus B1 300, Indra 1.1 CPU 300, PaddleOCR-VL-1.6 20)

| set | Orus C fp32 CPU | Orus B1 | Indra 1.1 CPU | PaddleOCR-VL-1.6 |
|---|---|---|---|---|
| bnhtrd | 23.5 | 23.4 | 98.4 | 94.6 |
| banglawriting | 37.4 | 38.2 | 98.6 | 128.7 |
| iiit | 51.7 | 54.3 | 107.0 | 115.0 |
| rupom | 62.3 | 62.4 | 100.4 | 127.9 |
| gnhk | 42.1 | 42.0 | 53.7 | 44.9 |
| iam | 33.5 | 34.1 | 49.2 | 37.2 |
| imgur5k | 39.3 | 41.0 | 57.7 | 25.0 |
| prescription | 22.9 | 24.3 | 93.4 | 75.0 |
| mixed | 40.8 | 39.9 | 91.7 | 121.1 |
| font | 14.0 | 14.4 | 41.9 | 21.6 |
| bnprinted | 44.2 | – | 86.3 | – |
| cord_en | 55.5 | – | 59.7 | – |
| **mean BN hand** | 43.7 | 44.6 | 101.1 | 116.6 |
| **mean EN hand** | 34.5 | 35.3 | 63.5 | 45.5 |

**ms / line** (lines per set: Orus C fp32 CPU 300, Orus B1 300, Indra 1.1 CPU 300, PaddleOCR-VL-1.6 20)

| set | Orus C fp32 CPU | Orus B1 | Indra 1.1 CPU | PaddleOCR-VL-1.6 |
|---|---|---|---|---|
| bnhtrd | 6.5 | 6.4 | 25.9 | 3881.3 |
| banglawriting | 11.8 | 11.3 | 22.5 | 26230.0 |
| iiit | 4.5 | 4.6 | 10.7 | 3162.5 |
| rupom | 10.8 | 10.5 | 12.5 | 9078.2 |
| gnhk | 5.9 | 5.4 | 17.9 | 2236.8 |
| iam | 9.2 | 8.4 | 17.1 | 3477.1 |
| imgur5k | 1.8 | 1.7 | 8.0 | 545.9 |
| prescription | 0.9 | 0.6 | 8.2 | 782.0 |
| mixed | 8.7 | 8.1 | 22.4 | 5728.5 |
| font | 3.0 | 2.2 | 13.8 | 4000.1 |
| bnprinted | 7.3 | – | 20.6 | – |
| cord_en | 4.8 | – | 12.0 | – |

Notes: Orus B1 column is the v1 model (GPU run) for comparison of accuracy only; its ms column is GPU. PaddleOCR-VL: 20 lines per set, GPU, see v1 §4. `bnprinted` and `cord_en` have test-set problems (multi-line images under a one-line label; skewed receipt rows containing the neighbouring row) described in `orus_weaknesses.md` §4 — their numbers are upper bounds on the error, not final.

**CPU speed per line (fp32 ONNX, 4 threads, idle)**: Orus 1.3–11.7 ms vs Indra 6.0–28.2 ms on the same crops — Orus is faster on every set, while being far more accurate on Bangla and better on English handwriting.

## 3. Page-level results (CPU, 4 threads, idle; 30 pages per set)

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

Per-stage page time for Orus (median ms): detect 62–94, recognize 29–265, handwritten flag 37–141. Full-page medians: Orus vs Indra 345 vs 333 (BN-HTRd), 279 vs 271 (BanglaWriting), 258 vs 211 (GNHK), 433 vs 443 (CMATERdb), 277 vs 267 (Swapnil), 133 vs 163 (CORD), 203 vs 204 (picture pages) — within about ±20% of Indra on every set (GNHK +22%, CORD −18%) while also computing the handwritten flag. Indra does not compute the handwritten flag; without it Orus is at or below Indra's page time on most sets (e.g. BN-HTRd 345 − 76 ≈ 269 ms vs Indra 333 ms; CORD 133 − 37 ≈ 97 ms vs 163 ms). The handwritten flag is the largest fixed cost and can run on a smaller thumbnail if needed.

## 4. Summary vs the request "CER < 5%, WER < 3% for printed and handwritten BN/EN"

| Group | Best Orus CER now | Target met? |
|---|---|---|
| Printed Bangla (pages, Swapnil) | 4.4% (WER 18.2%) | CER yes, WER no |
| Printed English | font set 4.6%; receipts 20.5% (test set partly broken) | partly |
| Handwritten Bangla | 6.9% (BN-HTRd) … 25.8% (unseen source) | no |
| Handwritten English | 12.4% (IAM) … 19.0% (IMGUR5K words) | no |

See `orus_weaknesses.md` for why and for the ranked fix list, and `sota_ocr_cer_wer.md` (research in progress) for what the best published systems reach on these benchmarks.

## 5. Model sizes (shipped configuration)

| Component | Params | Size |
|---|---|---|
| Indra det.onnx (unchanged) | 90.6k | 0.39 MB |
| Orus recognizer C, fp32 ONNX | 1.06M | 4.27 MB |
| Character LM | – | 3.19 MB |
| Picture net | 38.4k | 0.16 MB |
| Handwritten flag | 97.8k | 0.40 MB |
| **Total** | **≈1.29M** | **≈8.4 MB** (Indra default ≈ 41.9 MB) |
