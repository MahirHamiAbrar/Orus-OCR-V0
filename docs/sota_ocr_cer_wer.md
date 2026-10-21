# State of the art CER / WER: printed and handwritten English and Bangla

Research date: 2026-10-21. "Verified" means the number was read from the primary paper or the official leaderboard text (PDF converted with `pdftotext` and the table read directly). "Secondary" means the number comes from another paper's comparison table. "Unverified" means it comes only from a blog, repo or search snippet.

Notes on comparability:
- **IAM splits:** The "RWTH/Aachen" split (6,482 train / 976 val / 2,915 test lines) is used by TrOCR, DTrOCR, HTR-VT and DRetHTR. The older official split (6,161 / 940 / 1,861) is a different split. Numbers from the two splits are not strictly comparable. Some papers (for example Agbeti-Messan 2026) label the 6,161 split "Aachen" as well.
- **Normalization:** CER can be case-sensitive or case-insensitive, and punctuation may be kept or stripped. For Bangla it can be counted per Unicode codepoint or per grapheme cluster. On Tesseract Bengali, codepoint CER is 6.36% and grapheme CER is 4.59% for the same output (grapheme-kit, arXiv 2607.22456).
- **Units:** "%" is used throughout. ANLS/NED is a normalized edit distance (lower is better). It behaves like CER but is not identical to it.

---

## 1. Printed English

| Benchmark | Best CER | Best WER | Model | Params | Level / LM | Year | Source | Verified? |
|---|---|---|---|---|---|---|---|---|
| UW3 (modern English book pages, 72.8k train lines) | **0.114%** (5-model vote); 0.155% single | not reported | Calamari (2 conv + 1×LSTM-200, CTC) | not stated (tiny, under 1M by architecture) | line, no LM | 2018 | Wick et al., *Calamari*, https://arxiv.org/abs/1807.02004 (Table 2) | yes |
| UW3, same table | 0.397% | not reported | Tesseract 4 (same net trained on UW3) | same | line, no LM | 2018 | same | yes |
| UW3, same table | 0.870% | not reported | OCRopy LSTM-200 | tiny | line, no LM | 2018 | same | yes |
| UW-III (different split, 1,020 lines) | 0.6% | n/a | Breuel et al. 2013 OCRopus LSTM | tiny | line | 2013 | quoted in https://arxiv.org/abs/1712.05586 | secondary |
| OmniDocBench v1.0 text blocks, EN | 0.020 NED | n/a | GPT-4o | n/a | block-level | 2024/25 | https://github.com/opendatalab/OmniDocBench (component OCR table) | yes (README) |
| OmniDocBench v1.0 text blocks, EN | 0.096 NED | n/a | Tesseract-OCR | n/a | block-level | 2024/25 | same | yes (README) |
| OmniDocBench v1.6 full-page TextEdit (EN+ZH mixed) | 0.036 NED | n/a | MinerU2.5-Pro (PaddleOCR-VL 0.040, dots.ocr 0.048, Gemini 3 Pro 0.064, Mistral OCR 0.097, olmOCR 0.139) | 0.9B to multi-B VLMs | page, includes layout/reading order | 2026 | same | yes (README) |
| SROIE Task 2 (receipts) | n/a (word-level F1 98.37) | ≈1.6% word error (from F1) | DTrOCR | 105M | word | 2024 | Fujitake, *DTrOCR*, WACV 2024, https://arxiv.org/abs/2308.15996 (Table 2) | yes (F1 only) |
| olmOCR-bench | n/a (pass-rate metric, not CER) | n/a | Chandra 2: 85.8–85.9% | ~4B | page unit tests | 2025/26 | HF model card / codesota | unverified, not CER |

Verdict: on clean printed English lines, even tiny CNN-LSTM-CTC models reach CER of about 0.1–0.4%. A clean-print WER is not reported in these papers. With CER of about 0.2%, WER is about 1% by arithmetic, but that figure is inferred, not published. Real mixed documents (OmniDocBench) are worse: about 2–10% NED, depending on the system.

## 2. Handwritten English

| Benchmark | Best CER | Best WER | Model | Params | Level / LM | Year | Source | Verified? |
|---|---|---|---|---|---|---|---|---|
| IAM (Aachen split) | **2.26%** | **6.55%** (char tokens); BPE variant 2.36 / **4.81%** | DRetHTR-base (decoder-only RetNet + EfficientNetV2) | 107M | line, no external LM, synthetic pretraining | 2026 | Kim et al., *DRetHTR*, https://arxiv.org/abs/2602.17387 (Tables 8, 10, 11) | yes |
| IAM (Aachen) | 2.38% | not reported | DTrOCR (GPT-2 decoder only) | 105M | line, no LM, ~2B synthetic lines | 2024 | https://arxiv.org/abs/2308.15996 (Table 3) | yes |
| IAM (Aachen) | 2.75% | not reported | Diaz et al. Self-Attn/CTC + LM | not tabulated (~1e7 scale per Fig. 6) | line, with LM, internal data | 2021 | *Rethinking Text Line Recognition Models*, https://arxiv.org/abs/2104.07787 (Table 1) | yes |
| IAM (Aachen) | 2.89 / 3.42 / 4.22% | not reported | TrOCR large / base / small | 558M / 334M / 62M | line, no LM, 684M synthetic lines | 2021/23 | https://arxiv.org/abs/2109.10282 (Tables 4, 5) | yes |
| IAM | 3.2% | 10.5% | Bluche & Messina gated CRNN-CTC | **0.75M** | line, 50k lexicon + LM | 2017 | ICDAR 2017; values from TrOCR / Diaz / DRetHTR tables | secondary |
| IAM | 3.5% | 9.3% | Voigtlaender MDLSTM-CTC | 2.6M (survey) | line, with LM | 2016 | ICFHR 2016; via arXiv 2606.23524 Table 7 and survey arXiv 2502.08417 | secondary |
| IAM (Aachen) | 4.62% | 15.89% | Retsinas et al. "best practices" CNN-BiLSTM-CTC | not stated (compact ResNet + BiLSTM) | line, lexicon-free greedy | 2024 | https://arxiv.org/abs/2404.11339 (Table 3) | yes |
| IAM | 4.7% | 14.9% | HTR-VT (ViT + CTC) | 53.5M | line, no LM | 2024 | https://arxiv.org/abs/2409.08573 (Table 4) | yes |
| IAM | 5.0% (4.45% in DRetHTR table) | 16.3% | VAN | **2.7M** | line, no LM | 2023 | HTR-VT Table 4; DRetHTR Table 11 | secondary |
| IAM | 5.8% | 18.4% | Puigcerver CNN-BLSTM-CTC | **9.3M** | line, no LM (4.4% CER with LM) | 2017 | HTR-VT Table 4; survey Table II | secondary |
| IAM (116 pages, Aachen test) | 1.71% | 3.34% | GPT-4o-mini (Claude 3.5 Sonnet 1.75 / 3.55; GPT-4o 1.75 / 3.59; Qwen2-VL-7B 2.30 / 4.20) | API / 7B | **page-level**, line breaks and punctuation stripped before scoring | 2025 | Crosilla et al., *Benchmarking LLMs for HTR*, https://arxiv.org/abs/2503.15195 (Table IV) | yes (different protocol) |
| RIMES (French) | **1.81%** | not reported | DRetHTR-base | 107M | line, no LM | 2026 | arXiv 2602.17387 Table 9 | yes |
| RIMES | 2.75% | 9.93% | Retsinas best practices | compact | line, lexicon-free | 2024 | arXiv 2404.11339 | yes |
| RIMES (100 pages) | 1.69% | 3.76% | GPT-4o | API | page-level, normalized | 2025 | arXiv 2503.15195 Table V | yes |
| Bentham | **3.46%** | not reported | DRetHTR-base | 107M | line | 2026 | arXiv 2602.17387 Table 9 | yes |
| READ-2016 (German) | 3.59% (Coquenet, cited); 3.9% / 16.5% (HTR-VT); 4.21% (DRetHTR) | 16.5% | DAN/VAN family; HTR-VT | 53.5M (HTR-VT) | line | 2023–26 | arXiv 2409.08573 Table 3; arXiv 2602.17387 Table 9 | yes / secondary |
| LAM (Italian) | 2.8% | 7.4% | HTR-VT | 53.5M | line | 2024 | arXiv 2409.08573 Table 2 | yes |
| GNHK, IMGUR5K | not found | not found | – | – | – | – | no peer-reviewed CER/WER table located | not found |

Commercial APIs on IAM line-level CER (Google Cloud Vision, Azure Read, AWS Textract): no peer-reviewed number found. Mistral OCR, Chandra and PaddleOCR-VL on IAM: not found (only olmOCR-bench and OmniDocBench numbers exist).

**WER/CER ratio on handwriting (line level, no LM):** about 2.9–3.6×. Examples: DRetHTR 6.55/2.26 = 2.9; HTR-VT 14.9/4.7 = 3.2; Retsinas 15.89/4.62 = 3.4; Puigcerver 18.4/5.8 = 3.2; RIMES 9.93/2.75 = 3.6. With an LM the ratio is about 2.7–3.3× (Voigtlaender 9.3/3.5; Bluche 10.5/3.2). For LLM page-level output with punctuation stripped it is about 2×.

## 3. Printed Bangla (Bengali)

| Benchmark | Best CER | Best WER | Model | Params | Level / LM | Year | Source | Verified? |
|---|---|---|---|---|---|---|---|---|
| NLTR printed test (20 Bengali fonts, Bengali + English) | **0.68%** (CA 99.32%) | **3.35%** (WA 96.65%) | 1-layer BLSTM-128 + CTC | very small (single 128-unit BLSTM, about 0.2–0.5M by architecture; not stated) | line, no LM | 2019 | Paul & Chaudhuri, https://arxiv.org/abs/1908.08674 (Table I) | yes |
| same test | 1.46% | 7.14% | Google Drive OCR | n/a | page | 2019 | same | yes |
| same test | 8.21% | 23.69% | Tesseract 4.0 LSTM | n/a | page | 2019 | same | yes |
| FLORES+ sentences rendered in Noto (clean synthetic) | 6.36% codepoint / 4.59% grapheme | n/a | Tesseract | n/a | sentence | 2026 | grapheme-kit, https://arxiv.org/abs/2607.22456 (Table 4) | yes |
| IndicVisionBench-OCR, Bengali (real printed pages) | 4.04 char-ANLS | 11.30 word-ANLS | Gemini-2.5 Flash (Chitrapathak-2: 5.47 / 14.51; Surya: 12.61 / 28.76; GPT-4o: 32.68 / 55.51) | API; Chitrapathak ~3B | page | 2026 | Faraz et al., https://arxiv.org/abs/2602.16430 (Table 2) | yes (ANLS, not CER) |
| Apurba gold-standard docs (computer-composed) | 9.94% (Levenshtein acc. 90.06%) | n/a | Rabby et al. Bengali OCR (segmentation + char CNN) | not stated | page | 2024 | WACVW 2024, https://openaccess.thecvf.com/content/WACV2024W/WVLL/papers/Rabby_Enhancement_of_Bengali_OCR_by_Specialized_Models_and_Advanced_Techniques_WACVW_2024_paper.pdf (Tables 2–3) | yes |
| same, Tesseract | 26.51% (acc. 73.49%) | n/a | Tesseract | – | page | 2024 | same | yes |
| BornoDrishti (EACL 2026 Industry) | **no CER/WER** (printed word Top-1 accuracy 94.7%) | – | ViT CLIP-style encoder (word retrieval) | ~86M | word, closed vocabulary | 2026 | https://aclanthology.org/2026.eacl-industry.20.pdf (Tables 1–3) | yes (accuracy only) |

## 4. Handwritten Bangla (Bengali)

| Benchmark | Best CER | Best WER | Model | Params | Level / LM | Year | Source | Verified? |
|---|---|---|---|---|---|---|---|---|
| IIIT-INDIC-HW-WORDS Bengali (113K words, 24 writers) | **2.35%** | **12.34%** | Mondal & Jawahar, CRNN + attention + FastText semantic supervision | not stated | isolated word, no lexicon decoding | 2022 | https://arxiv.org/abs/2212.07776 (Table 3) | yes |
| same | 4.85% (OOV test 3.71) | 14.77% (OOV 16.65) | Gongidi & Jawahar baseline TPS-ResNet-BLSTM-CTC | 17.15M | word | 2021 | ICDAR 2021, https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/ConferencePapers/2021/iiit-indic-hw-words.pdf (Tables 3–4) | yes |
| BN-HTRd (805 full pages) | **6.19%** (4.58% if detection failures filtered) | **14.20%** (12.59%) | GraDeT-HTR word model, BnGraphemizer, decoder-only | 87M | **page-level end-to-end** (YOLO word detection + word recognizer) | 2025 | EMNLP 2025 Demos, https://arxiv.org/abs/2509.18081 (Tables 1, 3, 4) | yes |
| BN-HTRd, same | 26.17–38.34% | 39–52% | GraDeT line-level variants | 87–162M | line models, page eval | 2025 | same | yes |
| BN-HTRd, same | 19.39% | 32.49% | Gemini 2.5 Flash | API | page | 2025 | same, Table 3 | yes |
| Bongabdo (111 pages) | 8.68% | 23.56% | GraDeT-HTR | 87M | page | 2025 | same | yes |
| BanglaWriting → BN-HTRd (cross-dataset) | 18.04% | 33.20% | GraDeT-HTR | 87M | word | 2025 | same, Table 5 | yes |
| BN-HTRd → BanglaWriting (cross-dataset) | 11.12% (CRNN-CT) | 24.79% (GraDeT) | – | – | word | 2023/25 | same, Table 5 (CRNN rows cited from Basher et al. 2023) | yes / secondary |
| BN-HTRd → BanglaWriting | CRR 82.82% (≈17% CER) | WRR 59.09% | LILA-BOTI KD CRNN | not stated | word | 2022 | https://arxiv.org/abs/2205.11420 (Table I) | yes |
| BN-HTRd word test (6,614 words) | 2.52% | 7.79% | TrOCR + grapheme tokenizer (BUET thesis "PATHOK") | not stated | word | 2025? | https://github.com/kay33d/pathok-bangla-ocr | **unverified** (repo only) |
| BornoDrishti handwritten | no CER/WER (Top-1 word accuracy 68.9%; Rabby 2024 CNN 86.84%) | – | ViT encoder | ~86M | word retrieval | 2026 | EACL 2026 Industry, Table 3 | yes (accuracy only) |
| BN-HTRd line level, CER/WER of a dedicated line recognizer | not found | not found | – | – | – | – | – | not found |

Bangla WER/CER ratio (word level): 12.34/2.35 = 5.3 (IIIT best); 14.77/4.85 = 3.0 (baseline); 7.79/2.52 = 3.1 (PATHOK, unverified). For printed Bangla, Paul 2019 gives 3.35/0.68 = 4.9 and Google Drive gives 7.14/1.46 = 4.9. Bangla words are long in codepoints (conjuncts, vowel signs), so the WER/CER ratio is higher than in English.

---

## What is realistic for a ~1–10M-parameter CPU model

Evidence for small models (all CTC or CRNN-style, CPU-friendly):
- **Printed English:** Calamari or Tesseract-size CNN-LSTM models (well under 1M) reach CER of 0.11–0.40% on clean UW3 lines (arXiv 1807.02004). A 1–10M model is more than enough.
- **Printed Bangla:** A single BLSTM-128 layer (well under 1M) reaches CER 0.68% and WER 3.35% on 20 clean fonts (arXiv 1908.08674). Real scanned books and newspapers are much harder: even Gemini-2.5 scores about 4% char-ANLS and 11% word-ANLS on IndicVisionBench (arXiv 2602.16430).
- **Handwritten English (IAM line):**
  - Gated CRNN, 0.75M: CER 3.2% / WER 10.5% with lexicon and LM.
  - VAN, 2.7M: CER about 4.5–5.0%, WER 16.3%.
  - Puigcerver CRNN, 9.3M: CER 5.8% / WER 18.4% without LM, CER 4.4% with LM.
  - Retsinas compact CRNN: CER 4.62% / WER 15.89%.
  - Realistic target without LM: **CER 4–6%, WER 14–18%**. With an n-gram LM or lexicon: **CER 3–4.5%, WER 9–12%**.
  - Getting below 3% CER so far needed 100M+ parameters and hundreds of millions of synthetic lines (TrOCR, DTrOCR, DRetHTR).
- **Handwritten Bangla (word level):** The 17M CRNN baseline reaches CER 4.85% / WER 14.77% on IIIT-HW Bengali. The best published result is CER 2.35% / WER 12.34%. At page level with detection, the best is CER 6.2% / WER 14.2% (87M). Realistic target for a 1–10M model: **CER 4–8%, WER 13–25%**, depending on whether input is cropped words or whole lines and pages.

## Is CER < 5% and WER < 3% achievable, per group?

Arithmetic first: WER is about 3–5× CER on handwriting. **WER < 3% implies CER of roughly 0.6–1%**, so the WER target is the binding constraint and "CER < 5%" is redundant.

| Group | Verdict | Why |
|---|---|---|
| Printed English | **Yes** (clean scans) | CER 0.1–0.4% published for tiny models, so WER < 3% is plausible. Real mixed documents (OmniDocBench) are about 2–10% NED, so **partly** there. |
| Handwritten English | **CER yes, WER no** | CER < 5% is routine, including with 1–10M CRNNs. The best line-level WER in any published specialist model is 4.81% (DRetHTR, 107M, BPE). Small models get 10–18%. GPT-4o-class LLMs reach about 3.3–3.6% WER only at page level with punctuation stripped. |
| Printed Bangla | **Partly** | Clean synthetic-font lines: CER 0.68% and WER 3.35% (just above 3%) with a tiny BLSTM. Real printed pages: best system (Gemini-2.5) is at about 4% char / 11% word ANLS. CER < 5% is achievable, but WER < 3% only on clean, font-rendered text. |
| Handwritten Bangla | **No** for WER; **partly** for CER | Best published word-level result is CER 2.35% / WER 12.34%. Page level: CER 6.19% / WER 14.20%. No published system is anywhere near 3% WER. CER < 5% is achievable on cropped words with a good model. |

Suggested realistic targets for a small CPU model (cropped lines or words, greedy CTC plus an optional lexicon):

| Group | Target CER | Target WER |
|---|---|---|
| Printed EN | ≤ 1% | ≤ 3% |
| Printed BN | ≤ 2% | ≤ 8% |
| Handwritten EN | ≤ 5% | ≤ 15% (≤ 11% with LM) |
| Handwritten BN | ≤ 6% | ≤ 18% |

Report WER alongside CER, and state the split, case and punctuation normalization, and whether CER is counted per grapheme or per codepoint.
