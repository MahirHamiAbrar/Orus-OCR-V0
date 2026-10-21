# 2. Technical documentation — Orus-OCR-0.1-alpha

## 2.1 Pipeline

```
image (any size, colour or gray)
 ├─ gray; deskew: indra_ocr.pipeline.estimate_skew on a preview, rotate if |angle| ≥ 0.3°
 ├─ line detection: Indra det.onnx (LineNet, 90.6k params), long side 1024 (2048 when lines are tiny)
 ├─ picture regions: orus_pic.onnx (38.4k params) on a 384 px thumbnail → drop lines centred inside a picture
 ├─ line crops: indra_ocr.rec_preproc.crop_line(height=32, window=3) → [96, W] ink map per line
 ├─ recognition: orus_rec.onnx (1.06M params), width-bucketed batches, greedy CTC
 │     └─ lines with confidence 0.3–0.9: CTC prefix beam search (beam 8) + character 5-gram LM (orus_lm.npz)
 ├─ drop lines with confidence < 0.35 or empty text
 ├─ layout assembly: rows, reading order, indentation, blank lines → text
 ├─ language: share of Bengali vs Latin letters in the output
 └─ handwritten flag: v2_teff.onnx tile classifier (97.8k params) on the whole page
→ {"model", "language", "handwritten", "handwritten_score", "text", "lines": [{bbox, text, conf, lang}], "ms"}
```

Code: `orus/ocr.py` (`OCR`, `Recognizer`, `Detector`, `HandFlag`, `assemble`, `language`, `export`).

## 2.2 Recognizer (`orus/ocr_model.py: RecNet`)

An exact PyTorch port of Indra OCR 1.1 `rec.onnx`: maximum |Δ log-prob| against the ONNX graph is 1.8e-5. Only the vocabulary head is new.

**Input:** ink map `x [B, 1, 96, W]`, values 0–1 (1 = ink).
- The middle 32 rows hold the target line.
- The top and bottom 32 rows hold the page above and below it (Indra's "tall window").

`widths [B]` gives the valid pixels per line.

| Layer | Shape / settings |
|---|---|
| Stem | `Conv2d(1→128, kernel (32,4), stride (32,4))` on the middle 32 rows → `[B,128,T]`, **T = W/4** (one CTC step per 4 px) |
| Context branch (tall window) | on all 96 rows: `Conv2d(1→32, 4×4, s4)` → HardSwish → depthwise 3×3 → `Conv2d(32→64, 1×1)` → HardSwish → 4-head softmax over the 24 rows (score 1×1 conv + learned row prior `[1,4,24,1]`) → weighted row sum → `Conv1d(64→128)`; added to the stem output |
| Stem norm | BatchNorm1d(128) → HardSwish, masked to valid steps |
| 10 × Block | depthwise `Conv1d(128, k=9)` → expand `1×1 128→256` → HardSwish → masked squeeze-excite (`256→64→256`, mean over valid steps only) → project `1×1 256→128` → residual add → mask |
| Head | `Conv1d(128→212, 1×1)` → logits `[B, T, 212]` |
| Lengths | `clamp((widths + 3) // 4, max = T)` |

- **Parameters:** 1,058,616. Indra's `rec.onnx` has 1,250,566; the difference is the smaller vocabulary head.
- **Receptive field:** the depthwise k=9 convolutions over 10 blocks see roughly ±40 steps (±160 px). There is no recurrent or attention layer over the whole line; this is the main capacity limit (see `03_benchmarks.md`).

**Vocabulary** (211 characters + CTC blank at index 0):
- ASCII 0x20–0x7E (95);
- every assigned codepoint of the Bengali block U+0980–U+09FF;
- `‘ ’ “ ” – — • … ° × ÷ ± © ® £ € ₹ ¢ । ॥`.

**Label normalisation** (`canon`): NFC, ZWJ/ZWNJ removed, some look-alike characters mapped (`−‐‑` → `-`, `৷` → `।`), whitespace collapsed. Lines with a character outside the vocabulary are dropped, never altered.

**Warm start** (`load_indra`):
- Named weights are loaded by name.
- Anonymous BN-folded convolutions are matched by ONNX node name.
- Head rows are copied for every character Indra's vocabulary shares with ours; the rest start random.

## 2.3 Line crop (Indra's rule, `crop_line(height=32, window=3)`)

1. The line box is padded by 0.2× its height vertically and 0.3× horizontally.
2. It is scaled so the line is 32 px tall. The tall window adds the same height above and below, giving 96 px in total.
3. The crop is converted to an ink map (light-on-dark lines are inverted).

Maximum width is 4096 px at inference and 2048 px in training stores. Lines are batched by similar width, with the batch area capped.

## 2.4 Decoding and language model

- **Greedy CTC:** argmax per step, collapse repeats, drop blanks. Line confidence is the mean max-probability over the emitted steps.
- **Gated LM rescoring.** Only lines with confidence between 0.3 and 0.9 go through Indra's CTC prefix beam search:
  - beam 8, top-8 classes per step;
  - score = `log P_ctc + 0.3·log P_lm + 1.0·len`.
- **LM** (`orus/ocr_lm.py`, 3.19 MB): order-5 character n-gram in Indra's packed npz backoff format, with absolute discounting (D = 0.7).
  - Training text: 6M characters of Bangla Wikipedia, wikitext-103, and generated mixed / list / prescription / date / price text.
  - Sanity check: a real phrase must score better than its shuffled version.

## 2.5 Detector

Indra OCR 1.1 `det.onnx`, unchanged (90.6k params):
- **Input:** gray ink map, long side 1024, padded to a multiple of 32.
- **Output:** 3 channels at stride 2: a core-band logit (the middle 40% of the line height) and the distances to the line's top and bottom.
- **Decoding** (`det_post.decode`): connected components of the core band, each giving one box from its mean top/bottom distance.
- **Small text:** if the median line is under 12 px and there are at least 8 lines, detection runs again at a larger scale (target 16 px, up to 2048).

v1.1 has **no picture channel**, hence the separate picture net.

## 2.6 Picture-region net (`orus/ocr_pic.py`)

- **Input:** page thumbnail, 384 px long side, ink map, zero-padded to 384×384.
- **Layers:** `Conv-BN-HardSwish` ×6: 1→16 s2, 16→24 s2, 24→32 s2, then 32→32 at dilation 2, 4 and 8. Then `Conv 1×1 → 1` gives logits on a 48×48 grid (stride 8). 38,361 params.
- **Output:** cells above 0.5, connected components, kept if at least 1% of the page. Lines whose centre falls inside a picture box are dropped.
- **Training data:** 75% real train pages (BN-HTRd, GNHK, BanglaWriting) and 25% rendered pages, each with 0–3 pasted blocks (75% Imagenette photos, 30% of them faded; 25% printed charts). 4,000 pre-rendered thumbnails, mirror and contrast augmentation.
- **Held-out results:** mask IoU 0.89; 8.3% of picture-free pages get a false picture.

## 2.7 Layout assembly (`assemble`)

- Lines sorted by vertical centre. Lines whose centres are within 0.5 × median line height share a row and are joined left to right; the gap between them becomes 1–8 spaces.
- Indentation: `round((x_left − page_left) / (0.45·h) / 2)` spaces.
- A blank line is inserted when the vertical gap to the previous row exceeds 1.0 × line height.
- Single column only: multi-column pages would need XY-cut or Indra's `order.onnx` (not used).

## 2.8 Language and handwritten flag

- **Language:** count Bengali-block letters (b) and Latin letters (e).
  - `bn+en` if both scripts have at least 3 letters and a share of at least 5%;
  - otherwise the majority script;
  - `unknown` if there are none.
- **Handwritten:** the project's existing tile classifier `v2_teff.onnx` (97.8k params).
  - The page is normalised and cut into tiles; each tile gets a probability.
  - The page score is aggregated with `train.page_score`. "Yes" if the score is above 0.96 (`v2_teff.json`).

## 2.9 Files and ONNX interface

| File | Interface |
|---|---|
| `models/orus_rec.onnx` | inputs `x float32 [B,1,96,W]`, `widths int64 [B]`; outputs `logits float32 [B,T,212]`, `lengths int64 [B]`; opset 17; metadata `name` = Orus-OCR-0.1-alpha, `vocab` (JSON list), `cfg` (`{"c":128,"e":256,"depth":10,"rnn":0,"height":32}`), `val` (validation CER) |
| `models/orus_rec.pt` | `{"model": state_dict, "cfg": …, "chars": vocab, "val": …, "step": …}` → `RecNet(**cfg)` |
| `models/det.onnx` | Indra: input `x [1,1,H,W]`, output `[1,3,H/2,W/2]` |
| `models/orus_pic.onnx` | input `x [1,1,384,384]`, output `logit [1,1,48,48]` |
| `models/orus_lm.npz` | Indra `NgramLM` format: `order`, `vocab_size`, `bits` = 11, keys / values / history backoffs per order; token = class id + 1, BOS = 0 |
| `models/v2_teff.onnx` + `.json` | input `tiles [N,1,128,128]` normalised; threshold in json |

Store format for training data (`ocr_data.Store`, memory-mapped):
- `<name>.blob.npy`: uint8 PNG bytes, concatenated;
- `<name>.meta.npy`: offset, n, width, label offset, label length, hand flag;
- `<name>.lab.npy`: int16 class ids.

## 2.10 Size and speed

| Component | Params | Size | Median time per page (BN-HTRd, 4 threads, idle) |
|---|---|---|---|
| Detector | 90.6k | 0.39 MB | 77 ms |
| Recognizer (fp32) + gated LM | 1.06M | 4.27 MB + 3.19 MB | 176 ms (≈ 20 lines) |
| Picture net | 38.4k | 0.16 MB | ~3 ms |
| Handwritten flag | 97.8k | 0.40 MB | 76 ms |
| **Total** | ≈1.29M | ≈8.4 MB | **345 ms** (Indra 1.1: 333 ms) |

- **Per line on CPU:** 1.3–11.7 ms, against Indra's 6.0–28.2 ms on the same crops.
- **Dynamic INT8** (`ConvInteger`) is not shipped: it is slower on CPU and costs 7–10 CER points on handwriting.
- **Cold start:** the first call takes ~5 s, mostly loading torch for the handwritten flag's tile code.
