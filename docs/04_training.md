# 4. Training recipe — Orus-OCR-0.1

Everything below runs from the repo root with `./train.sh <stage> ...`. Each stage writes to `runs/`; `models/` holds the frozen release and is never written.

> **Reproducibility note (public release):** the original training run's `runs/` checkpoints, the `bench/` test sets, the `data/` directory (raw research-only datasets and line stores) and `scripts/dl_imgur5k.py` are **intentionally not** in this public repo. The recipe below is preserved as-is for historical reference; to actually re-run training, the artifacts above need to be available locally. Released weights in `models/` reproduce the **inference** path (`python run.py`) without any of them.

## 4.1 Environment and safety

- Python 3.10, PyTorch 2.x with CUDA, `onnxruntime`, `onnx`, `opencv-python`, `numpy`, `pyarrow`, `rapidfuzz`, `fonttools`, `Pillow` (RAQM layout for Bangla), `indra-ocr` (pip; the Indra 1.1 weights are fetched into the HF cache on first use), `transformers` ≥ 5 (only for the distillation teacher).
- Hardware used: RTX 3060 12 GB, 6 CPU cores, 15.6 GB RAM. The data loader is CPU-bound; the GPU is never the bottleneck at this model size.
- **Memory:**
  - Every heavy step runs in `systemd-run --user --scope -p MemoryMax=… -p MemorySwapMax=1G` (the `cap` helper in `train.sh`), so an out-of-memory kills the job and not the desktop session.
  - Training data are memory-mapped stores, never Python lists, so data-loader workers share the page cache.
  - Synthetic shards larger than ~50k lines at 32 px exceed a 4 GB cap; keep shards ≤ 50k.
- Put the line stores on an SSD. Random access to 8 GB of stores on a spinning disk made training I/O-bound.

## 4.2 Data preparation (`./train.sh data`)

1. **Raw datasets** go under `data/raw/` (see `docs/01_research_method.md` §1.4 for the list; download notes in the main project's `info/datasets.md`; `scripts/dl_imgur5k.py` fetches IMGUR5K with md5 checks).
2. `orus/ocr_data.py pages` converts every source into `data/ocr/pages/<src>.jsonl`.
   - Line boxes come from the dataset annotations.
   - For BanglaWriting and GNHK, lines are grouped from word boxes, and the word boxes are kept.
   - BN-HTRd text is read from its xlsx files with a plain zip+XML reader; word boxes come from its YOLO files.
3. `orus/ocr_data.py crops` cuts every line into a 32 px ink map with a 96 px tall window (Indra's `crop_line`), into stores `data/ocr/rec/<src>_<split>`.
   - Lines with word boxes are cut along a fitted baseline (`line_rect` + `rot_crop`): slope clipped to ±0.5, height = robust median of word heights × 1.1.
   - Lines with characters outside the vocabulary are dropped.
   - Lines CTC cannot emit are dropped (label length + repeats > width / 4).

## 4.3 Synthetic data (`./train.sh synth`)

- **Word bank** (`ocr_synth.py words`): real handwritten word crops at 48 px height into `data/ocr/words/{bn,en}_{train,test}`.
  - Bangla: 53.5k train / 11.1k test; English: 97.3k / 11.1k.
  - Sources: BN-HTRd, BanglaWriting, GNHK, IMGUR5K, prescriptions, following each source's own split. IIIT is listed in the code but was not downloaded yet when the released bank was built; rebuilding now would add it.
- **Font lines** (`lines font`): 3 stacked lines (context / target / context) rendered word by word, so spaces between scripts stay visible.
  - Fonts: 629 (the project's 118 print + 74 handwriting-style + 13 Bangla, plus Bangla fonts from `bn_fonts_extra`). 15% are held out by md5 for the `font` test set.
  - Text mix: Bangla 35%, English 30%, Bangla-led mixed 22%, English-led mixed 13%. Special lines: prescriptions, dates, ৳ prices, phone numbers, vitals, simple maths. Optional list prefixes (`1.`, `১.`, `ক)`, `•`).
  - Degradations: elastic warp, paper texture, ink colour, blur, noise, JPEG, then Indra's crop.
  - Released run: 5 train shards × 50k (246,567 kept), val 3,000, test 5,000.
- **Stitched lines** (`lines stitch`): 3–10 real word crops from the bank, from the same split only, placed on a line at height 28–56 px with gaps of 0.25–0.7 h and small vertical jitter.
  - Modes: Bangla 30%, English 25%, Bangla-led mixed 30%, English-led mixed 15%; context rows above and below.
  - Released run: 4 × 50k (199,995 kept), val 3,000, test 5,000.

## 4.4 Language model (`./train.sh lm`)

`ocr_lm.py build 5 6e6` builds a character 5-gram LM with absolute discounting (D = 0.7) into `runs/orus_lm.npz`:
- 45% Bangla Wikipedia, 30% wikitext-103, 25% generated mixed text (as in §4.3).
- Singletons are pruned at orders 4–5.
- Size: 3.2 MB.

## 4.5 Recognizer training (stages A → B1 → B3 → C)

Common settings (`orus/ocr_train.py rec`):

| Setting | Value |
|---|---|
| Model | `RecNet(c=128, e=256, depth=10)`, Indra-initialised (`--init indra`), 1.06M params |
| Loss | CTC, blank 0, `zero_infinity` |
| Optimiser | AdamW, weight decay 0.01, gradient clip 2.0 |
| Schedule | OneCycle: 5% warm-up, `div_factor` 20, `final_div_factor` 100 |
| Precision | bf16 autocast (CTC in fp32); `cudnn.benchmark` off (widths vary) |
| Batches | per round, draw 384 lines by source weight, sort by width, cut into 4 batches; ≤ 192 lines and ≤ 96×1024 px of summed width per batch; random width stretch 0.8–1.25 when CTC-feasible |
| GPU augmentation | rotation ±1.5°, scale ±6%, vertical shift ±4%, italic shear ±0.2, bolder 20% / thinner 15% strokes, Gaussian blur 30%, contrast 0.5–1.2, background lift (30%), noise ≤ 0.06 |
| Source weights (`SOURCE_W`) | synth_font 0.22, synth_stitch 0.18, bnhtrd 0.13, iiit 0.10, imgur5k 0.08, gnhk 0.07, iam 0.06, banglawriting 0.05, prescription 0.04, bnsynthwords 0.03, bnprinted 0.02, distill 0.03 (renormalised over the sources present) |
| Validation | greedy, no LM, ≤ 1,500 lines per source; best checkpoint = lowest mean CER of the real sets |

| Stage | Command (in `train.sh`) | Starts from | Steps | Peak LR | Workers | Time | Best val (mean real) |
|---|---|---|---|---|---|---|---|
| A | `./train.sh A` | Indra rec.onnx | 30,000 | 1.5e-3 | 3 | 46 min | 16.0% (7 sets) |
| B1 | `./train.sh B1` | A | 40,000 | 8e-4 | 3 | 31 min | 15.0% (8 sets, incl. IIIT) |
| B3 | `./train.sh B3` | B1 | 30,000 | 4e-4 | 2 | 44 min | 14.7% |
| C | `./train.sh C` | B3 (+ distill stores) | 12,000 | 3e-4 | 3 | 10 min | **14.65% → released** |
| B2 (ablation) | `./train.sh B2` | scratch, c192/e384/depth 12 (2.77M) | 45,000 | 1.5e-3 | 2 | 63 min | 19.6% |

Stage A of the released run started before IIIT and the stitched lines existed. It saw only the real sources and one font shard (49k lines). Running `./train.sh A` with all data present gives a slightly different A; B1 onward then sees the same data as the release.

## 4.6 Distillation (`./train.sh distill`)

1. `ocr_distill.py lines 120`: the Indra detector runs on deskewed unlabeled pages (medical_hw 97, NoTeS-Bank 120, reddit_hw 75, CMATERdb BE001–100), giving 9,879 line crops.
2. `ocr_distill.py teach paddlevl`: PaddleOCR-VL-1.6 (prompt "OCR:", greedy, ≤ 96 tokens, loaded straight to the GPU) reads 1,200 randomly sampled lines from the English pages. Bangla is skipped because the teacher invents text there.
3. `ocr_distill.py store runs/orus_rec_b1.pt 0.3`: a line is kept when Orus (greedy) and the teacher agree within 30% CER and the teacher's text is in the vocabulary and CTC-feasible; the label is the teacher's text.
   - 503 kept: 462 train / 41 val, split page-disjoint.
   - Stored as the `distill` source.

## 4.7 Picture net (`./train.sh pic`)

1. `ocr_pic.py data 4000` pre-renders 4,000 thumbnails (384 px) with masks at stride 8: 75% real train pages plus 25% rendered pages, with 0–3 pasted photos or charts.
2. `ocr_pic.py train 4000`: batch 32, AdamW at 3e-3, OneCycle, BCE with `pos_weight` 2, mirror and contrast augmentation. About 2.5 min on the GPU.

Held-out evaluation: mask IoU 0.89; false picture on 8.3% of pages that have none.

## 4.8 Export and evaluation

- `./train.sh export`: `ocr.py export runs/orus_rec_c.pt runs/orus_rec.onnx` writes fp32 ONNX (opset 17, dynamic batch and width) with metadata (name, vocab, cfg, val). It checks argmax agreement with PyTorch on random input. It also writes a dynamic INT8 copy; do not ship that one (see `03_benchmarks.md` §3.3) without testing it on real lines.
- `./train.sh bench` runs:
  - the line benchmark on the frozen `bench/` sets (rebuild them only with `REMAKE=1`);
  - Indra for comparison;
  - page evaluation;
  - error analysis.

  Results go to `runs/*.json`; `scripts/report_tables.py` turns them into markdown tables.
- **Making a new release:** copy `runs/orus_rec.onnx` (and `runs/orus_lm.npz` / `runs/orus_pic.onnx` if rebuilt) into a new folder laid out like `models/`, write a `v2_teff.json`, and regenerate `MD5SUMS`.

## 4.9 Reproducing the release from scratch (order and wall time on the machine above)

| Step | Time |
|---|---|
| `data` (10 sources) | ~25 min |
| `synth` (word bank + 9 shards of 50k) | ~1.5 h with 6 processes |
| `lm` | ~3 min |
| `A`, `B1`, `B3` | ~2 h |
| `distill` (teacher is the slow part: ~0.4 lines/s) | ~1 h |
| `C` | 10 min |
| `pic` | ~25 min (rendering) + 3 min |
| `export`, `bench` | ~15 min |

Total: about 6 hours of machine time.

Expected result: the numbers in `03_benchmarks.md`, within run-to-run noise of roughly ±0.3 CER points per set (training sampling is not seeded).
