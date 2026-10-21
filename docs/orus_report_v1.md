# Orus-OCR-0.1-alpha — evaluation report v1 (after training, before distillation)

Date: 2026-10-21. Machine: RTX 3060 12 GB, 6 CPU cores, 15.6 GB RAM. All numbers are produced by code in this repo (`ocr_bench.py`, `ocr_eval.py`, `ocr_pic.py`) and saved in `runs/*.json`; tables below are generated from those files by `scripts/report_tables.py`.

## 1. Summary

- **Orus-OCR-0.1-alpha** = Indra OCR 1.1 line detector (unchanged) + our own **1.06M-parameter BN+EN CTC recognizer** (Indra architecture, warm-started from Indra's `rec.onnx`, one 211-character Bangla+Latin vocabulary) + 5-gram character LM (gated beam search) + a new 38k-parameter picture-region net + the existing 98k-parameter handwritten/printed tile classifier.
- On **Bangla handwriting** Orus reaches **16.3% mean line CER** against **92.9% for Indra 1.1** and **104.6% for PaddleOCR-VL-1.6** (0.9B VLM, which invents text in other scripts on Bangla handwriting). On BN-HTRd lines: **6.9% vs 84.6% (Indra) vs 72.1% (PaddleOCR-VL)**.
- On **English handwriting** Orus reaches **13.8% mean line CER** against **32.0% (Indra)** and **17.9% (PaddleOCR-VL)**. PaddleOCR-VL is still better on clean IAM (10.1 vs 12.8) and noisy IMGUR5K (14.8 vs 21.4) — the main remaining gap, and the target of distillation.
- **Mixed BN+EN lines** (real handwritten words of both scripts in one line): **16.4% vs 72.6% (Indra) vs 125.6% (PaddleOCR-VL)**.
- **Doctor's prescriptions** (BN/EN drug names, word level): **7.2% vs 53.4% (Indra) vs 27.8% (PaddleOCR-VL)**.
- **Pages**: BN-HTRd page CER **14.0% vs 88.7%** (Indra); language ID on real mixed BN+EN pages **96.7% vs 43.3%**; handwritten Yes/No 93–100% per set; **0 of 402** output lines on pasted pictures (Indra: 34 of 520).
- **Size**: recognizer 4.27 MB fp32 / **1.21 MB INT8** (99.6% argmax agreement with PyTorch). Whole Orus pipeline (det + rec INT8 + LM + picture net + hand flag) ≈ **5.4 MB**, vs Indra's default Latin+world set ≈ 41 MB.
- **Speed**: recognition is 1.6–11 ms per line on GPU (Indra on CPU: 10–36 ms). Page latency on CPU in this run was measured while GPU training jobs were using the CPU, so it is reported with that caveat (§6); a clean idle re-measurement goes into report v2.
- **Size sweep / from-scratch todo**: a 2.77M-parameter model trained from scratch (45k steps) reached 19.6% mean val CER vs 15.0% for the 1M Indra-initialised model after A+B1 (70k steps) — at this budget the warm start wins on every set.

## 2. What was built

| Part | File | Notes |
|---|---|---|
| Recognizer | `ocr_model.py`, `ocr_train.py` | Exact PyTorch port of Indra `rec.onnx` (max \|Δlogp\| 1.8e-5 vs ONNX), head re-mapped to the Orus vocab; width-bucketed CTC, GPU augmentation, bf16 AMP. |
| Data | `ocr_data.py`, `ocr_synth.py` | 10 real sources unified to pages JSONL + mmap line stores; word-baseline deskew for slanted handwriting; synthetic lines from 600+ fonts (408 Bangla) and stitched real word crops (BN/EN/mixed). |
| LM | `ocr_lm.py` | Order-5 character LM (3.2 MB) in Indra's npz format, BN Wikipedia + wikitext + generated mixed / prescription / list text; used only on low-confidence lines. |
| Pipeline | `ocr.py` | Deskew → detect → drop lines on pictures → recognize → assemble rows, indentation and blank lines → language + handwritten flag → JSON. ONNX export + INT8. |
| Picture net | `ocr_pic.py` | Indra det.onnx v1.1 has **no** picture channel, so text inside photos was being read; new 38k-param net on a 384 px thumbnail. |
| Benchmarks | `ocr_bench.py`, `ocr_eval.py` | Line-level (10 test sets) and page-level (7 page sets). |

### Training stages

| Stage | Init | Data | Steps | Mean real val CER* |
|---|---|---|---|---|
| A | Indra `rec.onnx` | real + 1 synthetic font shard | 30k | 16.0% (7 val sets) |
| B1 | A | all real incl. IIIT (82.5k BN words) + 250k font lines + 200k stitched lines | 40k | **15.0%** (8 val sets incl. IIIT) |
| B2 | scratch, 2.77M params (c192/e384/depth 12) | same as B1 | 45k | 19.6% |
| B3 | B1, lower LR | same as B1 | 30k | (running at the time of writing; in report v2) |

\* greedy decoding, no LM, validation splits (writer/page-disjoint from test).

Data volume used by B1: 246.6k rendered font lines, 200k stitched lines, 164k IMGUR5K words, 82.5k IIIT words, 25.5k bn_synth words, 11k BN-HTRd lines, 6.4k IAM lines, 6k GNHK lines, 3.1k prescription words, 2k BanglaWriting lines, 1k printed BN lines.

## 3. Test sets

Line level (`ocr_bench.py make`, 300 random test lines per set, seed 0; rupom has 221):

| Set | What | Source split |
|---|---|---|
| bnhtrd | Bangla handwriting lines | BN-HTRd test pages |
| banglawriting | Bangla handwriting lines (word boxes → lines) | BanglaWriting test writers |
| iiit | Bangla handwritten words | IIIT-INDIC-HW-WORDS bn test |
| rupom | Bangla handwriting lines from another source (rotated scans, cleaned + deskewed for all systems) | bn_hw_rupom (never trained on) |
| gnhk | noisy English handwriting (phone photos) | GNHK test |
| iam | clean English handwriting lines | IAM test |
| imgur5k | noisy English handwritten words in the wild | IMGUR5K test |
| prescription | doctor's handwritten drug names | Prescription BD test |
| mixed | BN+EN lines stitched from real **test-split** handwritten word crops | synthetic composition, real ink |
| font | rendered lines in **held-out** fonts | synthetic |

Every system gets the same crop (line with page context); VLMs get the plain line image.

## 4. Line-level results

**CER %** (lines per set: Orus B1 (1M) 300, Orus A (1M) 300, Orus B2 (3M scratch) 300, Indra 1.1 300, PaddleOCR-VL-1.6 20)

| set | Orus B1 (1M) | Orus A (1M) | Orus B2 (3M scratch) | Indra 1.1 | PaddleOCR-VL-1.6 |
|---|---|---|---|---|---|
| bnhtrd | 6.9 | 7.4 | 9.6 | 84.6 | 72.1 |
| banglawriting | 15.1 | 16.5 | 21.2 | 91.4 | 121.3 |
| iiit | 16.9 | 39.3 | 21.6 | 98.7 | 117.7 |
| rupom | 26.5 | 29.1 | 33.0 | 96.8 | 107.3 |
| gnhk | 13.5 | 14.0 | 17.0 | 20.7 | 18.9 |
| iam | 12.8 | 13.7 | 18.3 | 18.7 | 10.1 |
| imgur5k | 21.4 | 21.7 | 24.8 | 35.2 | 14.8 |
| prescription | 7.2 | 7.2 | 8.6 | 53.4 | 27.8 |
| mixed | 16.4 | 32.8 | 21.9 | 72.6 | 125.6 |
| font | 4.6 | 5.2 | 6.8 | 25.2 | 11.4 |
| **mean BN hand** | 16.3 | 23.1 | 21.3 | 92.9 | 104.6 |
| **mean EN hand** | 13.8 | 14.2 | 17.2 | 32.0 | 17.9 |

**WER %** (lines per set: Orus B1 (1M) 300, Orus A (1M) 300, Orus B2 (3M scratch) 300, Indra 1.1 300, PaddleOCR-VL-1.6 20)

| set | Orus B1 (1M) | Orus A (1M) | Orus B2 (3M scratch) | Indra 1.1 | PaddleOCR-VL-1.6 |
|---|---|---|---|---|---|
| bnhtrd | 23.4 | 25.4 | 28.1 | 98.4 | 94.6 |
| banglawriting | 38.2 | 39.4 | 46.0 | 98.6 | 128.7 |
| iiit | 54.3 | 84.3 | 59.0 | 107.0 | 115.0 |
| rupom | 62.4 | 65.7 | 68.6 | 100.4 | 127.9 |
| gnhk | 42.0 | 42.6 | 47.8 | 53.7 | 44.9 |
| iam | 34.1 | 35.0 | 43.4 | 49.2 | 37.2 |
| imgur5k | 41.0 | 44.0 | 47.3 | 57.7 | 25.0 |
| prescription | 24.3 | 22.6 | 29.2 | 93.4 | 75.0 |
| mixed | 39.9 | 63.0 | 46.7 | 91.7 | 121.1 |
| font | 14.4 | 15.2 | 18.7 | 41.9 | 21.6 |
| **mean BN hand** | 44.6 | 53.7 | 50.4 | 101.1 | 116.6 |
| **mean EN hand** | 35.3 | 36.1 | 41.9 | 63.5 | 45.5 |

**ms / line** (lines per set: Orus B1 (1M) 300, Orus A (1M) 300, Orus B2 (3M scratch) 300, Indra 1.1 300, PaddleOCR-VL-1.6 20)

| set | Orus B1 (1M) | Orus A (1M) | Orus B2 (3M scratch) | Indra 1.1 | PaddleOCR-VL-1.6 |
|---|---|---|---|---|---|
| bnhtrd | 6.4 | 7.2 | 13.0 | 36.3 | 3881.3 |
| banglawriting | 11.3 | 12.4 | 14.3 | 27.6 | 26230.0 |
| iiit | 4.6 | 5.3 | 5.2 | 12.9 | 3162.5 |
| rupom | 10.5 | 10.8 | 11.4 | 18.6 | 9078.2 |
| gnhk | 5.4 | 5.7 | 7.0 | 21.1 | 2236.8 |
| iam | 8.4 | 8.8 | 11.6 | 20.7 | 3477.1 |
| imgur5k | 1.7 | 1.7 | 1.8 | 10.1 | 545.9 |
| prescription | 0.6 | 0.6 | 0.7 | 9.7 | 782.0 |
| mixed | 8.1 | 9.2 | 9.5 | 29.6 | 5728.5 |
| font | 2.2 | 2.5 | 3.7 | 19.2 | 4000.1 |

Notes:
- "mean BN hand" = mean of bnhtrd, banglawriting, iiit, rupom; "mean EN hand" = mean of gnhk, iam, imgur5k, prescription.
- CER above 100% means the system wrote more wrong characters than the reference has (insertions): PaddleOCR-VL loops or writes Arabic / Devanagari / Hangul text on Bangla handwriting (e.g. BN-HTRd line *জরুরি পদক্ষেপ নিতে…* → Arabic sentence; another → `CO 쌍쌍쌍 참쓴`). It is accurate on English (`SODIUM` → `SODIUM`, prescription `Omon` for `Odmon`).
- PaddleOCR-VL was run on 20 lines per set (≈ 4–26 s per Bangla line on the shared GPU), so its numbers are a coarse estimate; Orus and Indra use 300 lines per set.
- Orus ms/line here are PyTorch on the GPU, batch of all lines in a set (throughput). Indra is ONNX Runtime on 4 CPU threads, one line per call. CPU-vs-CPU numbers for the same pipeline are in §6.
- **LightOnOCR-2-1B** could not be evaluated on line crops: with transformers 5.20's native implementation it returns an end-of-sequence token immediately for every line image (tried: no text prompt, bf16 pixels, line pasted on a page-sized canvas). It is a page-level model; its line numbers (`runs/bench_lighton.json`, CER 140–2900%) are therefore **not** reported as a result. A page-level LightOn comparison is left for later.

### Model progression

- A → B1: adding IIIT (82.5k Bangla handwritten words) and stitched mixed lines cut IIIT CER 39.3 → 16.9 and mixed 32.8 → 16.4, and improved every other set slightly.
- B2 (2.77M, scratch) is worse than B1 (1.06M, Indra init) on every set after a similar number of total steps; more steps would narrow the gap (still improving at the end, val 19.99 → 19.56 over the last 5k steps), but the 1M model is also 2.6× cheaper at inference. The 1M Indra-initialised model is kept.

## 5. Page-level results (`ocr_eval.py`, 30 pages per set, CPU, 4 threads)

Orus = full pipeline with the B1 recognizer INT8 ONNX. Indra = `indra_ocr.OrtOCR` full pipeline with its default models. Page CER compares text with all whitespace collapsed; "CER w/ newlines" keeps line breaks (reading order + line splitting); line-count error = |#lines out − #lines GT| / #lines GT.

Page text ground truth is reliable only for BN-HTRd (transcription per line from the dataset's xlsx) and Swapnil (printed BN pages with newlines). GNHK page GT in our JSONL skips lines with `%…%` / `*` tokens and BanglaWriting page lines are re-grouped from word boxes, so those sets report language / handwritten / latency only; their accuracy is covered at line level.

**bnhtrd** (30 pages)

| system | page CER % | CER w/ newlines % | line-count err % | lang acc % | handwritten acc % | ms p50 | ms p95 |
|---|---|---|---|---|---|---|---|
| Orus B1 INT8 | 14.0 | 14.0 | 3.6 | 96.7 | 100.0 | 696.1 | 900.0 |
| Indra 1.1 | 88.7 | 89.0 | 33.9 | 0.0 | – | 532.1 | 684.9 |

**banglawriting** (30 pages)

| system | lang acc % | handwritten acc % | ms p50 | ms p95 |
|---|---|---|---|---|
| Orus B1 INT8 | 100.0 | 100.0 | 553.3 | 750.8 |
| Indra 1.1 | 0.0 | – | 457.4 | 561.3 |

**gnhk** (30 pages)

| system | lang acc % | handwritten acc % | ms p50 | ms p95 |
|---|---|---|---|---|
| Orus B1 INT8 | 100.0 | 93.3 | 441.4 | 910.1 |
| Indra 1.1 | 100.0 | – | 364.3 | 533.6 |

**cmater_mixed** (30 pages)

| system | lang acc % | handwritten acc % | ms p50 | ms p95 |
|---|---|---|---|---|
| Orus B1 INT8 | 96.7 | 100.0 | 868.3 | 1500.4 |
| Indra 1.1 | 43.3 | – | 673.8 | 848.6 |

**swapnil_bn_print** (30 pages)

| system | page CER % | CER w/ newlines % | line-count err % | lang acc % | handwritten acc % | ms p50 | ms p95 |
|---|---|---|---|---|---|---|---|
| Orus B1 INT8 | 4.5 | 4.5 | 2.1 | 90.0 | 93.3 | 746.9 | 937.4 |
| Indra 1.1 | 4.4 | 4.4 | 2.7 | 90.0 | – | 420.0 | 540.2 |

**cord_en_print** (30 pages)

| system | lang acc % | handwritten acc % | ms p50 | ms p95 |
|---|---|---|---|---|
| Orus B1 INT8 | 96.7 | 96.7 | 265.0 | 432.0 |
| Indra 1.1 | 100.0 | – | 240.6 | 354.5 |

**synth_pictures** (30 pages)

| system | ms p50 | ms p95 | lines on pictures |
|---|---|---|---|
| Orus B1 INT8 | 419.8 | 790.4 | 0 (of 402 lines) |
| Indra 1.1 | 324.3 | 449.1 | 34 (of 520 lines) |

Reading the page tables:
- **BN-HTRd pages**: 14.0% page CER (line CER 6.9%; the gap is detection: split / merged lines and lines at page edges). Indra reads Bangla handwriting as Latin / Devanagari garbage (88.7%) and its line count is off by 34%.
- **Swapnil printed Bangla**: Orus 4.5% ≈ Indra 4.4%. Indra's `rec_world` handles *printed* Bangla well; Orus matches it while being trained mostly on handwriting.
- **Language ID** (`bn` / `en` / `bn+en` from the output characters): 90–100% on single-script sets, **96.7% on real mixed BN+EN pages (CMATERdb, never trained on)** vs 43.3% for Indra's output. The 3 Swapnil "misses" (90%) are pages whose ground-truth text itself contains English words (the same rule on the GT says `bn+en`), so Orus is right there and the expected label is wrong; corrected Orus accuracy is 100% on this set.
- **Handwritten Yes/No** (existing 98k-param tile classifier on the whole page): 100% on BN-HTRd, BanglaWriting, CMATERdb; 93.3% on GNHK; 93.3% / 96.7% "No" on printed Swapnil / CORD.
- **Images are ignored**: on 30 synthetic pages with 1–3 pasted photos/charts, **0 of 402** Orus output lines lie on a picture; Indra outputs 34 of 520 lines from inside pictures. Picture net on held-out pages: mask IoU 0.89, false picture on 8.3% of picture-free pages (which can hide real text there; the min-area rule limits this to blocks ≥ 1% of the page).

## 6. Speed

| What | Orus | Indra 1.1 |
|---|---|---|
| Line recognition, GPU, batched (ms/line) | 0.6–11.3 | – |
| Line recognition, CPU 4 threads (ms/line) | in v2 (idle machine) | 9.7–36.3 (one line per call) |
| Page p50 / p95, CPU 4 threads, BN-HTRd | 696 / 900 ms* | 532 / 685 ms* |
| Page p50, CORD printed receipts | 265 ms* | 241 ms* |
| Page p50, CMATERdb mixed | 868 ms* | 674 ms* |

\* measured while three GPU jobs (training, VLM baseline, teacher) were running and using CPU cores; both systems ran under the same load, but absolute numbers are inflated. Orus page time includes the handwritten flag (~20–90 ms) and picture net (~3 ms), which Indra does not compute. Clean idle numbers, with a per-stage breakdown and INT8 vs fp32, are in report v2.

## 7. Model sizes

| Component | Params | File |
|---|---|---|
| Line detector (Indra det.onnx, unchanged) | 90.6k | 0.39 MB |
| **Orus recognizer B1** | **1.06M** | 4.27 MB fp32 / **1.21 MB INT8** |
| Orus recognizer B2 (scratch, not used) | 2.77M | 11.1 MB / 2.96 MB INT8 |
| Character LM (order 5) | – | 3.19 MB |
| Picture net | 38.4k | 0.16 MB |
| Handwritten tile classifier | 97.8k | 0.40 MB |
| **Orus pipeline total (INT8)** | **≈1.29M** | **≈5.4 MB** |
| Indra 1.1 default (det, rec, rec_world, script, order, orient, lm, lm_world) | ≈3.5M | ≈41.9 MB |
| PaddleOCR-VL-1.6 / LightOnOCR-2-1B | 0.9B / 1B | ~2 GB bf16 |

## 8. Things that did not work (and what was done instead)

- **Detector fine-tuning** (`ocr_det.py`): after 1k steps validation F1 dropped (BanglaWriting precision 0.20 → 0.015). The GT line boxes built from word boxes are much taller than the ink, so the targets disagree with Indra's tight ink boxes. Indra's detector already finds these lines well (visual check), so it is used unchanged. Fix for later: tighten GT boxes to the ink before training.
- **Picture channel**: Indra det.onnx v1.1 outputs only 3 channels (core, top, bottom), so the picture net was added.
- **LightOnOCR-2-1B** on lines: see §4.
- **PaddleOCR-VL as a Bangla teacher**: unusable (invents text), so distillation is English-only (§9).
- **Night run**: the machine ran out of RAM at 02:57 (two trainings + another session's training + a benchmark), killing everything for ~6 h. All jobs now run in memory-capped systemd scopes, one heavy job at a time.

## 9. Distillation — decision and status

Needed: yes, for English. PaddleOCR-VL beats Orus on IAM and IMGUR5K, and Orus has seen few noisy English *pages* (doctor's notes, student notes). Not useful for Bangla (teacher fails).

Procedure (`ocr_distill.py`):
1. Indra det on 392 unlabeled real pages (medical_hw 97 readable images, NoTeS-Bank 120, reddit_hw 75, CMATERdb BE001–100 — BE101–150 stay for evaluation) → 9,879 line crops.
2. Teacher PaddleOCR-VL-1.6 on 1,200 English-page lines.
3. Keep a line only if Orus (B1, greedy) and the teacher agree within 30% CER, with the teacher's text as label: **503 kept** (reddit 144/262, NoTeS-Bank 210/557, medical 149/381) → 462 train / 41 val, page-disjoint.
4. Stage C: continue from B3 with the distilled lines added at 3% sampling weight, 12k steps; then re-run all benchmarks → report v2.

## 10. Limitations

- Mixed BN+EN line test set is stitched from real word crops, not natural mixed handwriting; real mixed pages (CMATERdb) have no transcription, so only language ID is measured there. The user's 150-page labelled set (Phase 8) will close this.
- Page CER only on two page sets with full transcripts.
- PaddleOCR-VL line numbers use 20 lines per set.
- No GPU numbers for Indra (ONNX CPU only) and no CPU line numbers for Orus in this report (v2).
