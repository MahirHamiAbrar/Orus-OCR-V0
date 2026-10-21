# 1. Research method — Orus-OCR-0.1-alpha

## 1.1 Problem

Read photos and scans of pages in **Bangla, English, or both mixed in one line**, with **handwriting** as the main case: noisy phone photos, clean scans, and doctors' prescriptions. For each page, output:
- the language: `bn`, `en` or `bn+en`;
- whether it is handwritten (Yes/No);
- the text with its structure: numbering, spacing, line breaks, blank lines.

Pictures inside a page are ignored. The model must be small and as fast as Indra OCR 1.1 on a CPU (about 230 ms per page on 4 cores), with quality first. Hardware: one RTX 3060 (12 GB), 6 CPU cores, 15.6 GB RAM.

## 1.2 Decisions taken up front (with the user, 2026-10-20/21)

| Decision | Reason |
|---|---|
| Use the **Indra OCR 1.1 architecture**, and warm-start from its weights | It is the only published option that meets the CPU speed target (conv CTC recognizer ~1.25M params, LineNet detector ~90k params). Every large VLM (0.25–1B) is 10–200× slower on CPU. |
| **One** Bangla+English recognizer, not per-script models | Indra routes each line to a per-script recognizer, which breaks on lines that mix Bangla and English. |
| Research project: **all datasets allowed**, including non-commercial ones | Bangla handwriting data is scarce; leaving out non-commercial sets would remove most of it. |
| VLMs used only as **baselines and teachers** | Too slow and too large to be the product. |
| Same code for **scratch and warm start** (`--init indra|scratch`) | So the gap between them can be measured. |

## 1.3 Hypotheses tested

| # | Hypothesis | Test | Outcome |
|---|---|---|---|
| H1 | Indra's Latin recognizer transfers to Bangla handwriting once its head is remapped to a joint vocab | Stage A from Indra vs a smoke run | Yes: BN-HTRd val CER 28.6% after 600 steps, 18.1% after 15k, 15.2% after 30k |
| H2 | More Bangla handwriting (IIIT words) plus stitched mixed lines fix the weak sets | A → B1 | Yes: IIIT test CER 39.3 → 16.9, mixed 32.8 → 16.4, every other set slightly better |
| H3 | A bigger model from scratch beats the 1M warm start at equal time | B2 (2.77M, scratch, 45k steps) vs B1 | No: worse on every set (mean val CER 19.6% vs 15.0%) |
| H4 | Longer training at a lower learning rate still helps | B1 → B3 (+30k steps at lr 4e-4) | Slightly: val 15.0 → 14.7 |
| H5 | Distillation from a VLM teacher on unlabeled real pages helps | B3 → C with 503 teacher-agreed lines | English only, small: IMGUR5K 19.9 → 19.0, IAM 12.8 → 12.4; Bangla unchanged (the teacher cannot read Bangla handwriting) |
| H6 | Fine-tuning Indra's detector on our line boxes improves page reading | 1k-step fine-tune, val F1 | No: F1 fell, because word-union GT boxes are taller than the ink. Indra's detector kept unchanged. |
| H7 | Text inside pictures can be suppressed with a tiny separate net | Picture-region net | Yes: 0 of 402 output lines on pasted pictures (Indra: 34 of 520) |
| H8 | Dynamic INT8 quantization is free | INT8 vs fp32 on real test lines | No: costs 7–10 CER points on handwriting and is slower on CPU. fp32 shipped. |

## 1.4 Data

All sources are unified into one page format (`data/ocr/pages/<src>.jsonl`: image, line boxes or word boxes, text, split, licence). Then each line is cut with Indra's crop rule into a 32 px line ink map, with the lines above and below kept as context (96 px tall). The crops are stored as memory-mapped stores.

| Source | Kind | Train / val / test lines | Split rule |
|---|---|---|---|
| BN-HTRd v4 | Bangla handwriting, 786 pages | 10,997 / 798 / 2,472 | by writer |
| BanglaWriting | Bangla handwriting | 1,978 / 212 / 315 | by writer |
| IIIT-INDIC-HW-WORDS (bn) | Bangla handwritten words | 82,554 / 12,947 / 17,574 | official split |
| GNHK | English handwriting, phone photos | 5,979 / 698 / 2,090 | official split |
| IAM lines | English handwriting, clean | 6,439 / 975 / 2,870 | official split |
| IMGUR5K | English handwritten words in the wild | 164,049 / 19,355 / 21,634 | official split |
| Prescription BD | doctors' handwritten drug names | 3,120 / 780 / 780 | by image |
| bn_synth_words | printed Bangla words | 25,455 / 795 / 748 | hashed |
| bn_printed_val | printed Bangla lines | 1,036 / 128 / 216 | hashed |
| bn_hw_rupom | Bangla handwriting, another source | test only (221) | **never trained on** |
| Synthetic font lines | 629 fonts (408 Bangla), 3 stacked lines, degradations | 246,567 / 2,961 / 4,931 | 15% of fonts held out for test |
| Stitched lines | real handwritten word crops stitched into BN / EN / mixed lines | 199,995 / 3,000 / 5,000 | words from the matching split only |
| Distill | teacher-agreed lines from unlabeled English notes and prescriptions | 462 / 41 | by page |

Line deskew: for sources with word boxes, the line is cut along a baseline fitted over the word boxes (rotated crop). Axis-aligned boxes of slanted handwriting otherwise squash the text, which cut BanglaWriting's yield from 1,981 to 303 lines before this fix. A line is kept only if CTC can emit it: label length plus repeats must be at most the number of time steps (width / 4).

## 1.5 Evaluation protocol

**Line level** (`orus/ocr_bench.py`, frozen sets in `bench/`): 300 random test lines per set (seed 0; rupom 221, bnprinted 236).
- Each line is stored at source resolution with its page context, so every system gets the same pixels. VLMs get the plain line crop.
- Sets: bnhtrd, banglawriting, iiit, rupom (Bangla handwriting); gnhk, iam, imgur5k (English handwriting); prescription; mixed (stitched from test-split words); font (rendered in held-out fonts); bnprinted, cord_en (real printed).

**Page level** (`orus/ocr_eval.py`): 30 pages per set, CPU, 4 threads, machine otherwise idle.
- Page CER is computed only where full page transcripts exist (BN-HTRd, Swapnil printed Bangla).
- Language and handwritten accuracy on all sets, including CMATERdb mixed Bangla+English pages BE101–150 (never trained on).
- Lines read on pictures: 30 synthetic pages with pasted photos and charts.

**Metrics.** Text is normalised with NFC, ZWJ/ZWNJ removed, and whitespace collapsed (`ocr_model.canon`).
- **CER:** Levenshtein distance between the character sequences, divided by the reference length.
- **WER:** the same over whitespace-separated tokens.
- **Normalised WER** (error analysis only) also ignores case and punctuation.

**Baselines.**
- Indra OCR 1.1 full pipeline (ONNX, CPU, its own script routing to `rec_world` for Bangla).
- PaddleOCR-VL-1.6 (0.9B, GPU, prompt "OCR:", 20 lines per set because Bangla lines take 4–26 s).
- LightOnOCR-2-1B was tried but returns an empty answer on line crops, so it is excluded.

**Error analysis** (`orus/ocr_errors.py`):
- edit types (substitution, insertion, deletion);
- the character class each error hits: space, punctuation, digit, Latin upper/lower, Bangla consonant, vowel sign, hasanta, other marks;
- the most frequent confusions;
- CER by line length and by model confidence;
- the share of all errors caused by the worst 10% of lines.

**Published state of the art** for each group was collected from primary sources, every number with a URL: `docs/sota_ocr_cer_wer.md`.

## 1.6 Model selection

Every training stage evaluates greedy CTC (no LM) on up to 1,500 validation lines per source every 3–5k steps. It keeps the checkpoint with the lowest **mean CER over the real validation sets** (synthetic sets are reported but not used for selection). Test sets are used only for the final numbers.

## 1.7 Threats to validity

- The **mixed test set** is stitched from real word crops, not natural mixed handwriting. Real mixed pages (CMATERdb) have no transcripts, so only language ID is measured on them.
- **Page CER** covers two page sets only. GNHK page ground truth in our format skips lines with `%…%` tokens, and BanglaWriting page lines are regrouped from word boxes.
- **Test-set noise**:
  - `bnprinted`: many images hold 2–4 lines under a one-line label;
  - `cord_en`: rows from skewed photos also cover the neighbouring row.

  Their numbers overstate the error.
- **PaddleOCR-VL numbers** use 20 lines per set: a coarse estimate.
- **Speed** was measured on a shared workstation with no other jobs running. Absolute numbers vary with the CPU; the Orus-vs-Indra ratios are the meaningful part.
- **Published results** are not strictly comparable. IAM has two splits, normalisation differs between papers, and Bangla CER per grapheme differs from per codepoint.
