# Orus-OCR-0.1-alpha — weakness analysis (model C, 2026-10-21)

Source: `python ocr_errors.py collect runs/orus_rec_c.pt` then `analyze` → `runs/preds_orus_rec_c.json`, `runs/errors_orus_rec_c.json`. Line test sets as in `docs/orus_report_v1.md` §3 (300 lines per set; rupom 221; bnprinted 236).

## 1. Where Orus stands against the requested target (CER < 5%, WER < 3%)

| Group | Set | CER (LM) | WER | WER, case+punct ignored |
|---|---|---|---|---|
| Printed BN | Swapnil pages (page level) | 4.4% | 18.2% | – |
| Printed BN | bnprinted lines | 24.3% ⚠ test set broken, see §4 | 44.3% | 41.0% |
| Printed EN | CORD receipt rows | 20.5% ⚠ partly test set, see §4 | 55.4% | 37.6% |
| Printed (synthetic, held-out fonts) | font | 4.6% | 14.0% | 12.5% |
| Handwritten BN | BN-HTRd | 6.9% | 23.6% | 24.7% |
| Handwritten BN | BanglaWriting | 14.7% | 37.3% | 37.6% |
| Handwritten BN | IIIT (words) | 16.4% | 52.0% | 51.9% |
| Handwritten BN | rupom (unseen source) | 25.8% | 62.2% | 52.4% |
| Handwritten EN | IAM | 12.4% | 33.4% | 28.3% |
| Handwritten EN | GNHK | 13.6% | 42.2% | 33.5% |
| Handwritten EN | IMGUR5K (words) | 19.0% | 39.3% | 35.0% |
| Doctor | prescription (words) | 6.9% | 23.0% | 22.3% |
| Mixed | stitched BN+EN | 16.7% | 40.9% | 39.6% |

Only printed text is near the CER target today. WER is 2.5–3.5× CER on every handwritten set (one wrong character makes a wrong word), so WER < 3% would need CER ≈ 1%.

## 2. Weaknesses, ranked by how much error they cause

### W1. Glyph-level confusions of similar shapes — the bulk of the error (52–64% of all edits are substitutions)
- English: `o↔a`, `a↔e`, `r↔n`, `m↔n`, `s↔r`, `t↔l` (IAM top 6 = 113 errors on 300 lines).
- Bangla: `ব↔র` (differ only by the dot under র), `ল↔ন`, `ণ↔ন`, `য↔র`, `ম↔স`, `ক↔ব`; on rupom also `ো↔ে`, `আ↔া`.
- Consonants carry 46–62% of all Bangla errors; per-class error rate of consonants is 7% (BN-HTRd) to 28% (rupom).
- Likely causes: (a) **32 px line height**, inherited from Indra (designed for Latin print): the র dot, the ণ/ন loop and Bangla vowel-sign details are only 1–3 px; (b) **no sequence model** — the recognizer is a pure conv stack with a limited receptive field, so it cannot use word context to separate `o/a` or `ব/র`; the gated character LM only touches unsure lines.

### W2. Conjuncts (hasanta `্`) and rare marks
- Error rate inside the class: hasanta 9% (BN-HTRd), 18% (BanglaWriting), 24% (IIIT), 28% (rupom); "other" marks (ঁ ং ঃ ় ৎ) 6–29%.
- `্ → া` is a top confusion on font and rupom: the conjunct is read as a separate vowel sign.
- Cause: a CTC over **codepoints** has to emit 3 codepoints for one visual conjunct inside ~2–3 timesteps (stride 4 px at 32 px height); conjunct glyph shapes are also the rarest in data.

### W3. Digits and punctuation
- Inside-class error: digits 30–71% (GNHK 31%, BN-HTRd 37%, rupom 71%), punctuation 27–43%.
- Comma/period and `,`/`।` confusions; capital vs lower case (`c→C`, `O→o`) on GNHK/CORD.
- Cause: very little real handwritten digit/punctuation data; synthetic text is mostly words.

### W4. Deletions (skipped characters): 26–39% of edits
- Highest on IAM (38.5%) and font (38.6%); long lines are worse on IAM (24% CER at 60+ chars vs 11% at 30–60).
- Cause: CTC with 4 px per step on 32 px text gives ~0.5–1 steps per narrow character for tight cursive; repeated letters need a blank in between.

### W5. A minority of lines cause most of the error
- The worst 10% of lines cause 26–69% of all errors (prescription 69%, font 67%, IMGUR5K 52%).
- These are crop failures (wrong box, neighbour line inside the crop, rotated text), unreadable/ambiguous ink, and label noise.
- **Confidence separates them well**: lines with confidence ≥ 0.95 have CER 2–4% (BN-HTRd 2.4%, IAM 2.5%, IMGUR5K 2.2%, prescription 3.2%, font 0.8%); lines under 0.6 have CER 27–87%. A reject / fallback rule ("send lines under 0.8 to a slower reader") is available today.

### W6. Domain shift to unseen writers / sources
- rupom (never trained on, other scanner/writers): 25.8% vs 6.9% on BN-HTRd test writers. BanglaWriting (2k training lines only): 14.7%.
- Real Bangla handwriting lines in training: ~13k lines + 82k IIIT words — small.

### W7. Printed text that is not clean book print
- CORD thermal / dot-matrix receipts: 20.5% (Indra 21.9%). Dot-matrix glyphs, skewed photos, and wide gaps between item and price are absent from the synthetic printed data.
- Printed Bangla pages are fine (Swapnil 4.4%), so the recognizer itself reads print; receipts and skew are the gap.

### W8. Quantization
- Dynamic INT8 (ConvInteger) costs 7–10 CER points on handwriting (IAM 12.4 → 21.5) and is slower on CPU; fp32 is the shipped default now. Static (calibrated) INT8 or fp16 needs to be tried before any size reduction.

### W9. Page-level losses on top of line errors
- BN-HTRd: line CER 6.9% → page CER 12.8%. Extra loss comes from detection (split / merged / missed lines; line-count error 3.5%) and reading order.
- The Indra detector was not fine-tuned (§8 of report v1).

### W10. Language model is weak
- Gated character 5-gram helps 0.5–2.4 CER points (greedy → LM). No word lexicon, no word-level LM, and it only runs on lines with confidence 0.3–0.9.

## 3. What is *not* a big problem
- Word segmentation (spaces): 1–8% inside-class error, 4–12% of edits.
- Bangla vowel signs (kar) on BN-HTRd: 3.7% inside-class.
- Mixed-script lines: per-script errors in mixed lines look like the single-script ones; the script switch itself is not where errors come from.
- Images inside pages: 0 lines read from pictures.

## 4. Test-set problems found (fixed in the benchmark before any new numbers)
- `bnprinted`: many "line" images from the source dataset contain 2–4 printed lines while the label is one line (e.g. a 62×651 crop with three lines; a crop with a rotated page number). The worst 10% of its lines carry 63% of its errors and 75% of its edits are deletions. → filter multi-line images out of the set (and out of training).
- `cord_en`: rows come from row_id groups of photographed receipts; on skewed photos the axis-aligned row box contains the neighbouring row (e.g. *Total 40,000* box also covers *Pay Cash 100,000*); CORD labels also drop dot leaders (`Total........`). → deskew rows with the word quads (as done for handwriting) before scoring.

## 5. Highest-leverage fixes (ordered by expected CER gain per hour)

| # | Change | Targets | Expected effect | Cost |
|---|---|---|---|---|
| 1 | Taller input: 48 px line height (keep 4 px stride → 1.5× more steps per character) | W1, W2, W4 | largest single gain expected on Bangla diacritics, conjuncts and deletions | retrain (warm start from C by resizing the stem), ~1.3–1.5× CPU cost |
| 2 | Add a light sequence layer (2-layer BiLSTM 256 or 2–4 transformer layers) on top of the conv features | W1, W4, W10 | standard CRNN/SVTR gain on handwriting; uses word context | +1–3M params, CPU cost to be measured |
| 3 | More real handwriting: all of IAM (we use 6.4k lines), RIMES/Bentham, more BN data, synthetic handwriting with more Bangla handwriting fonts + digits/punctuation-heavy text | W3, W6 | broad | data work |
| 4 | Word-level LM / lexicon beam on every line (BN + EN dictionaries) | W1, W10 | typically largest WER drop; small CER drop | CPU cost of beam |
| 5 | Confidence-gated fallback (lines < 0.8 to a stronger model, e.g. a larger Orus or a VLM for EN) | W5 | caps error on the hard 20–40% of lines | latency only on those lines |
| 6 | Dot-matrix / receipt fonts and skew in synthetic printed data; deskew per line | W7 | receipts | data work |
| 7 | Detector fine-tune with ink-tight GT boxes | W9 | page CER toward line CER | ~2 h |
| 8 | Static INT8 / fp16 export | W8 | size without accuracy loss | small |
