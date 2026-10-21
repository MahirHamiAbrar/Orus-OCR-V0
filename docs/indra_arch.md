# Indra OCR 1.1: architecture, as read from the ONNX graphs

Source: `huggingface.co/datalab-to/indra-ocr`, tag `v1.1.0` (snapshot `bd02167f`), plus the inference code at `github.com/datalab-to/indra` (Apache 2.0). Weights are under a modified OpenRAIL-M licence, and research use is allowed. We read this with `onnx.load` and checked it with ONNX Runtime on 2026-10-21. Orus-OCR-0.1-alpha builds on it.

## Files
| File | Size | Params | Input | Output |
|---|---|---|---|---|
| `det.onnx` | 0.4 MB | **0.09M** | `x [1,1,H,W]` ink map, long side 1024, multiple of 32 | `out [1,3,H/2,W/2]`: core logit, top distance, bottom distance |
| `rec.onnx` (base: Latin + Latin handwriting) | 5.0 MB | 1.25M | `x [B,1,96,W]` tall window (`rec_window=3`), `widths [B]` | `ids, conf, lengths, blank, topv/topi [B,T,8]` |
| `rec_world.onnx` (Indic incl. Bengali, Thai, ...) | 5.6 MB | 1.40M | `x [B,1,32,W]`, `widths` | same |
| `rec_cjk.onnx` / `rec_rtl.onnx` | 10.2 / 4.8 MB | 2.54M / 1.19M | 32-px lines | same |
| `script.onnx` | 0.2 MB | 0.05M | 32-px line | `p [B,4]` over groups base/cjk/rtl/world |
| `order.onnx` | 3.0 MB | 0.73M | `feats [1,N,16]` per line, `mask` | pairwise order `a`, `cls [1,N,10]` block labels, `same` block, `tab` table relations |
| `orient.onnx` | 0.4 MB | 0.09M | 768×768 thumbnail | 5 classes (0/90/180/270/vertical) |
| `lm.npz`, `lm_world.npz`, `lm_cjk.npz` | 46 MB total | | token n-gram backoff tables | |

`meta.json`:
- `det_long_side=1024`, `rec_height=32`, `rec_window=3`;
- `lm_params` for world/cjk: alpha 0.2, beta 1.5, gate 0.95;
- `script_groups=[base,cjk,rtl,world]`.

**Vocab:**
- base `vocab`: 1,697 chars, **including all 96 Bengali codepoints and the 95 printable ASCII**;
- `vocab_world`: 2,935 chars, also with Bengali and ASCII.

So both recognizers can already emit mixed BN+EN text; **no head replacement is needed**.

## Recognizer (`rec_world.onnx`; `rec.onnx` is the same plus a context branch)
Pure convolution plus CTC, with no RNN or attention:
1. **Stem:** `Conv2d(1→128, kernel=(32,4), stride=(32,4))` patchifies the 32-px line into one row, **one time step per 4 px** (T = W/4). Then BatchNorm, HardSwish.
2. **10 residual 1D blocks**, each at 128 channels:
   1. depthwise `Conv1d k=9`;
   2. expand 1×1 to 256;
   3. HardSwish;
   4. **masked squeeze-excite** (mean over valid steps using `widths`, fc→ReLU→fc→sigmoid);
   5. project 1×1 back to 128;
   6. residual add;
   7. multiply by the padding mask.
3. **Head:** `Conv1d 128→|vocab|+1` (blank = 0). Its outputs (argmax ids, max-prob conf, log-softmax top-8 for the beam search, blank log-prob) are computed inside the graph.
4. **`rec.onnx` context branch** for the tall window (96 px = line ± one line height):
   1. `Conv2d(1→32, 4×4, s4)` + HardSwish;
   2. depthwise 3×3 + 1×1 to 64;
   3. 4 attention heads over rows with a learned `row_prior [1,4,24,1]`, softmax over the 24 rows;
   4. pool, project 64→128, add to the stem features.

   This gives the model page context (ascenders, neighbouring lines) at almost no cost.

Parameter budget (world): head 128×2,936 ≈ 0.38M; blocks ≈ 10 × ~75k ≈ 0.75M; the rest is the stem.

## Detector (`det.onnx`, "LineNet")
- **Backbone:** a tiny MobileNetV3-style network.
  - 3×3 stem, stride 2, 16 channels.
  - Inverted-bottleneck depthwise blocks (3×3, then 5×5) with HardSwish, at strides 4/8/16/32 and widths 24/32/48/64.
- **Neck:** FPN-like top-down path with nearest `Resize` and 1×1 lateral convs (24 channels).
- **Head:** depthwise 3×3, then 1×1 to 12 channels, then `DepthToSpace(2)`, giving 3 channels at stride 2. The distances go through `Softplus`.
- **Targets and decoding** (from `det_post.py`):
  - Core band: 40% of line height, ends trimmed by 0.15× height.
  - Top and bottom distances are regressed on core pixels.
  - Decoding: connected components on the core, then the mean top and bottom per component.
- **Images:** v1.1's `det.onnx` outputs 3 channels; the picture channel in the code is optional and absent in this build. Orus drops pictures entirely.

## Conversion to PyTorch (for warm start)
- `onnx2torch` converts `det.onnx` exactly (max abs diff 3.5e-5, 90,568 trainable params).
- `rec*.onnx` fail in `onnx2torch` (a `Clip` with dynamic min/max in the width-mask code). **Plan:** re-implement the recognizer in PyTorch (stem, 10 blocks, head; plus the context branch for the `rec.onnx` variant) and load the weights from the ONNX initializers by name (`rec.stem.*`, `rec.blocks.N.*`, `rec.head.*`). Depthwise weights are stored under anonymous names (`onnx::Conv_*`, because BN was folded), so they are mapped by node order. Check that the outputs match ORT.

## Implications for Orus
- Size target: det 0.09M + rec ~1.3–1.4M. Orus can afford a 3–8M recognizer (wider blocks, more blocks) and stay near Indra's CPU latency.
- Warm-start candidates:
  - `rec.onnx`: tall window, trained on Latin handwriting, vocab already includes Bengali, so it is the best start for handwriting;
  - `rec_world.onnx`: printed Bengali;
  - `det.onnx`: exact conversion.
- One recognizer for BN+EN removes the need for `script.onnx` routing, which breaks on mixed lines.
