"""Phase 2 / 8: line-level OCR benchmark for Orus-OCR-0.1-alpha and baselines.

  python ocr_bench.py make                  -> data/ocr/bench/<set>.pkl  (seeded sample of each TEST split)
      item = (gray png at source resolution incl. context, line box in it, text)
  python ocr_bench.py run <system> [arg]    -> runs/bench_<system>.json  CER / WER per set + ms per line
      systems: orus <ckpt.pt|.onnx>, indra, paddlevl, lighton
Sets: bnhtrd, banglawriting, iiit (BN handwriting), gnhk, iam, imgur5k (EN handwriting), prescription (doctor),
rupom (BN, other source), mixed (stitched BN+EN handwriting from test-split words), font (rendered, held-out fonts),
bnprinted (real printed BN lines), cord_en (printed EN receipt rows, CORD-v2 test).
"""
import json
import os
import pickle
import random
import sys
import time

import cv2
import numpy as np
from rapidfuzz.distance import Levenshtein

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # sub-project root (data/ links to the main project data)
B = f"{ROOT}/bench"                                   # frozen 0.1-alpha test sets (copied 2026-10-21)
SETS = ["bnhtrd", "banglawriting", "iiit", "gnhk", "iam", "imgur5k", "prescription", "rupom", "mixed", "font",
        "bnprinted", "cord_en"]
N = 300


def make():
    import ocr_data as D
    from ocr_model import canon, encode
    os.makedirs(B, exist_ok=True)
    for s in SETS[:8] + ["bnprinted"]:
        p = f"{ROOT}/data/ocr/pages/{s}.jsonl"
        if not os.path.exists(p):
            print("skip", s)
            continue
        pages = [pg for pg in map(json.loads, open(p)) if pg["split"] == "test"]
        cand = [(pg, j) for pg in pages for j in range(len(pg["lines"]))]
        random.Random(0).shuffle(cand)
        out = []
        for pg, j in cand:
            l = pg["lines"][j]
            t = canon(l["text"])
            if encode(t) is None:
                continue
            g = cv2.imread(f"{ROOT}/{pg['image']}", cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            if l.get("words"):
                g, box = D.rot_crop(g, *D.line_rect(l["words"]))
            elif l["box"] is None or s == "rupom":          # whole-image lines
                g, box = whole_line(g)
            elif len(l["box"]) == 5:
                g, box = D.rot_crop(g, *l["box"])
            else:
                x0, y0, x1, y1 = l["box"]
                h = y1 - y0
                X0, Y0 = int(max(0, x0 - h)), int(max(0, y0 - 2 * h))
                g = g[Y0:int(y1 + 2 * h), X0:int(x1 + h)]
                box = [x0 - X0, y0 - Y0, x1 - X0, y1 - Y0]
            out.append((cv2.imencode(".png", g)[1].tobytes(), [float(v) for v in box], t))
            if len(out) == N:
                break
        pickle.dump(out, open(f"{B}/{s}.pkl", "wb"))
        print(s, len(out), flush=True)
    out = cord_lines()
    pickle.dump(out, open(f"{B}/cord_en.pkl", "wb"))
    print("cord_en", len(out), flush=True)
    for s, f in (("mixed", "synth_stitch_test_0"), ("font", "synth_font_test_0")):
        p = f"{ROOT}/data/ocr/rec/{f}"
        if not os.path.exists(p + ".meta.npy"):
            print("skip", s)
            continue
        st = D.Store(p)
        items = [(st[i][0], st.text(i), 1) for i in range(len(st))]
        if s == "mixed":       # keep lines with both scripts
            import re
            items = [x for x in items if re.search(r"[ঀ-৿]", x[1]) and re.search(r"[A-Za-z]", x[1])]
        out = []
        for png, t, _ in items[:N]:
            ink = cv2.imdecode(np.frombuffer(png, np.uint8), 0)
            g = 255 - ink
            h = 32 / 1.4
            out.append((cv2.imencode(".png", g)[1].tobytes(), [0.3 * h, 32 + 0.2 * h, g.shape[1] - 0.3 * h, 64 - 0.2 * h], t))
        pickle.dump(out, open(f"{B}/{s}.pkl", "wb"))
        print(s, len(out), flush=True)


def cord_lines():
    """Printed English: CORD-v2 test receipts, words grouped into rows (row_id), row box = union of word quads."""
    import pyarrow.parquet as pq
    from ocr_model import canon, encode
    out = []
    for row in pq.read_table(f"{ROOT}/data/bench_raw/cord/data/test-00000-of-00001-9c204eb3f4e11791.parquet").to_pylist():
        g = cv2.imdecode(np.frombuffer(row["image"]["bytes"], np.uint8), cv2.IMREAD_GRAYSCALE)
        rows = {}
        for vl in json.loads(row["ground_truth"])["valid_line"]:
            for w in vl["words"]:
                q = w["quad"]
                rows.setdefault(w["row_id"], []).append((min(q["x1"], q["x4"]), min(q["y1"], q["y2"]),
                                                         max(q["x2"], q["x3"]), max(q["y3"], q["y4"]), w["text"]))
        for ws in rows.values():
            ws.sort()
            t = canon(" ".join(w[4] for w in ws))
            b = [min(w[0] for w in ws), min(w[1] for w in ws), max(w[2] for w in ws), max(w[3] for w in ws)]
            if len(t) >= 2 and encode(t) is not None and b[3] - b[1] >= 6:
                h = b[3] - b[1]
                X0, Y0 = int(max(0, b[0] - h)), int(max(0, b[1] - 2 * h))
                crop = g[Y0:int(b[3] + 2 * h), X0:int(b[2] + h)]
                out.append((cv2.imencode(".png", crop)[1].tobytes(), [b[0] - X0, b[1] - Y0, b[2] - X0, b[3] - Y0], t))
    random.Random(0).shuffle(out)
    return out[:N]


def whole_line(g):
    """A whole-image line (rupom: scans rotated with black fill) -> upright crop of the paper rectangle."""
    import ocr_data as D
    n, lab, st, _ = cv2.connectedComponentsWithStats((g > 40).astype(np.uint8))
    if n < 2:
        return g, [0, 0, g.shape[1], g.shape[0]]
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    (cx, cy), (w, h), ang = cv2.minAreaRect(cv2.findNonZero((lab == k).astype(np.uint8)))
    if w < h:
        w, h, ang = h, w, ang - 90
    ang = (ang + 45) % 90 - 45                        # rotations are small; never flip the line
    out, box = D.rot_crop(g, cx, cy, w - 6, h - 6, -ang)
    x0, y0, x1, y1 = map(int, box)
    out[:y0], out[y1:], out[:, :x0], out[:, x1:] = 255, 255, 255, 255   # blank everything outside the paper
    return out, box


def load_sets():
    return {s: pickle.load(open(f"{B}/{s}.pkl", "rb")) for s in SETS if os.path.exists(f"{B}/{s}.pkl")}


def dec(png):
    return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)


def line_only(g, box, pad=0.25):
    x0, y0, x1, y1 = box
    h = y1 - y0
    return g[int(max(0, y0 - pad * h)):int(y1 + pad * h), int(max(0, x0 - pad * h)):int(x1 + pad * h)]


# ---------------- systems: fn(list of (gray, box)) -> list of str ----------------
def sys_orus(path):
    import ocr
    m = ocr.Recognizer(path)
    return lambda items: m.read_boxes([(g, b) for g, b in items])


def sys_indra(_=None):
    from indra_ocr import OrtOCR, model_dir
    o = OrtOCR(model_dir(None), threads=4)
    return lambda items: [o.recognize(g, np.array([b], np.float32))[0][0] for g, b in items]


def _vlm(model_id, prompt):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from PIL import Image
    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()

    proc.tokenizer.padding_side = "left"                 # batched generation for a decoder-only model

    def run(items, bs=1):                                # bs>1 is slower: one looping line holds the batch
        out = []
        for k in range(0, len(items), bs):
            msgs = [[{"role": "user", "content": [{"type": "image", "image": Image.fromarray(line_only(g, b)).convert("RGB")}]
                      + ([{"type": "text", "text": prompt}] if prompt else [])}] for g, b in items[k:k + bs]]
            inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                           return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                o = model.generate(**inp, max_new_tokens=96, do_sample=False)
            out += [t.strip() for t in proc.batch_decode(o[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)]
        return out
    return run


def sys_paddlevl(_=None):
    return _vlm("PaddlePaddle/PaddleOCR-VL-1.6", "OCR:")


def sys_lighton(_=None):
    return _vlm("lightonai/LightOnOCR-2-1B", "")


def run(system, arg=None, limit=N):
    from ocr_model import canon
    fn = globals()[f"sys_{system}"](arg)
    res = {}
    for s, items in load_sets().items():
        items = items[:limit]
        inp = [(dec(p), b) for p, b, _ in items]
        t0 = time.perf_counter()
        pred = fn(inp)
        ms = 1000 * (time.perf_counter() - t0) / max(1, len(items))
        ed = sum(Levenshtein.distance(canon(p), t) for p, (_, _, t) in zip(pred, items))
        n = sum(len(t) for _, _, t in items)
        wed = sum(Levenshtein.distance(canon(p).split(), t.split()) for p, (_, _, t) in zip(pred, items))
        wn = sum(len(t.split()) for _, _, t in items)
        res[s] = dict(cer=round(ed / n, 4), wer=round(wed / wn, 4), ms_per_line=round(ms, 1), n=len(items),
                      sample=[(p, t) for p, (_, _, t) in list(zip(pred, items))[:3]])
        print(system, s, res[s]["cer"], res[s]["wer"], f"{ms:.1f} ms", flush=True)
    tag = system if not arg else f"{system}_{os.path.basename(str(arg)).split('.')[0]}"
    json.dump(res, open(f"{ROOT}/runs/bench_{tag}.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make()
    else:
        run(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None,
            int(os.environ.get("BENCH_N", N)))
