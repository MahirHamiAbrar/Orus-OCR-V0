"""Phase 7: distillation for Orus-OCR-0.1-alpha. A VLM teacher reads line crops of unlabeled real pages;
lines where the teacher and Orus agree closely are kept as new labelled training data.

  python ocr_distill.py lines [pages_per_src=120]   -> data/ocr/distill/lines.pkl  (Indra det boxes)
  python ocr_distill.py teach [paddlevl]            -> data/ocr/distill/teacher.jsonl (resumable; 1200 EN lines)
  python ocr_distill.py store <orus.pt> [max_cer=0.3] -> data/ocr/rec/distill_train_0 (+ _val_0)
Unlabeled pages: medical_hw (doctor), reddit_hw (EN notes), NoTeS-Bank (EN notes, Task_1 images),
CMATERdb BN+EN pages BE001-BE100 (BE101-BE150 stay for the page evaluation).
"""
import glob
import json
import os
import pickle
import random
import sys
import zlib

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # sub-project root
BENCH = f"{ROOT}/data/bench_raw"
OUT = f"{ROOT}/data/ocr/distill"


def pages(n):
    import pyarrow.parquet as pq
    r = random.Random(0)
    srcs = {
        "medical_hw": sorted(glob.glob(f"{BENCH}/medical_hw/*.jpg")),
        "notesbank": sorted(glob.glob(f"{BENCH}/notesbank/Task_1/Images/**/*.jp*g", recursive=True)),
        "cmater": sorted(glob.glob(f"{ROOT}/data/raw/cmaterdb_mixed/CMATERdb1.2.2/*.bmp"))[:100],
    }
    for s, ps in srcs.items():
        r.shuffle(ps)
        for p in ps[:n]:
            yield s, os.path.basename(p), cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    k = 0
    for f in sorted(glob.glob(f"{BENCH}/reddit_hw/data/*.parquet")):
        for bt in pq.ParquetFile(f).iter_batches(batch_size=4, columns=["image", "image_index"]):
            for row in bt.to_pylist():
                yield "reddit_hw", str(row["image_index"]), cv2.imdecode(np.frombuffer(row["image"]["bytes"], np.uint8), 0)
                k += 1
                if k == n:
                    return


def build_lines(n=120):
    """Indra det boxes on deskewed pages -> (id, ink crop png for Orus, plain line png for the teacher)."""
    from indra_ocr import model_dir
    from indra_ocr.pipeline import DESKEW_MIN, estimate_skew, preview, rotate_page
    from indra_ocr.rec_preproc import crop_line
    import ocr
    from ocr_bench import line_only
    from ocr_data import LINE_H
    det = ocr.Detector(os.path.join(str(model_dir(None)), "det.onnx"))
    os.makedirs(OUT, exist_ok=True)
    f = open(lines_path(), "wb")                         # one pickle record per line (streamed, RAM safe)
    k = 0
    for s, pid, g in pages(n):
        if g is None:
            continue
        if max(g.shape) > 3000:                           # big phone photos: detection runs at 1024 anyway
            z = 3000 / max(g.shape)
            g = cv2.resize(g, None, fx=z, fy=z, interpolation=cv2.INTER_AREA)
        a = estimate_skew(preview(g))
        if abs(a) >= DESKEW_MIN:
            g = rotate_page(g, a)
        for j, b in enumerate(det(g)):
            if b[3] - b[1] < 8 or b[2] - b[0] < 2 * (b[3] - b[1]):
                continue                                  # tiny or not line-shaped
            ink = crop_line(g, b[:4], height=LINE_H, window=3, max_width=2048)
            pickle.dump((f"{s}/{pid}/{j}", cv2.imencode(".png", (ink * 255).astype(np.uint8))[1].tobytes(),
                         cv2.imencode(".png", line_only(g, b[:4]))[1].tobytes()), f)
            k += 1
        print(s, pid, k, flush=True)
    f.close()
    print("lines", k)


def lines_path(h=None):
    from ocr_data import LINE_H
    h = h or LINE_H
    return f"{OUT}/lines.pkl" if h == 32 else f"{OUT}/lines{h}.pkl"


def read_lines(h=None):
    with open(lines_path(h), "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                return


def teach(system="paddlevl", n=1200):
    """Teacher on n English-page lines (PaddleOCR-VL makes up text on Bangla handwriting, see the report)."""
    import ocr_bench as B
    lines = [x for x in read_lines() if x[0].split("/")[0] in ("medical_hw", "notesbank", "reddit_hw")]
    lines = random.Random(0).sample(lines, min(n, len(lines)))
    p = f"{OUT}/teacher.jsonl"
    done = {json.loads(l)["id"] for l in open(p)} if os.path.exists(p) else set()
    fn = getattr(B, f"sys_{system}")()
    with open(p, "a") as f:
        for k in range(0, len(lines), 16):
            part = [x for x in lines[k:k + 16] if x[0] not in done]
            if not part:
                continue
            gs = [cv2.imdecode(np.frombuffer(x[2], np.uint8), 0) for x in part]
            pred = fn([(g, [0, 0, g.shape[1], g.shape[0]]) for g in gs])     # stored crop already is the line
            for x, t in zip(part, pred):
                f.write(json.dumps(dict(id=x[0], text=t), ensure_ascii=False) + "\n")
            f.flush()
            print("teacher", k + len(part), "/", len(lines), flush=True)


def store(ckpt, max_cer=0.3):
    """Keep lines where Orus (greedy) and the teacher agree within max_cer; label = teacher text.
    Agreement is judged by ckpt on crops of its own height; the stored crops are of ORUS_H (data/ocr/rec<h>)."""
    import ocr
    from rapidfuzz.distance import Levenshtein
    from ocr_data import LINE_H, REC, ctc_ok, save_store
    from ocr_model import canon, encode
    rec = ocr.Recognizer(ckpt, lm=None)
    lines = {x[0]: x[1] for x in read_lines(rec.height)}
    out_crops = lines if LINE_H == rec.height else {x[0]: x[1] for x in read_lines(LINE_H)}
    teacher = {d["id"]: canon(d["text"]) for d in map(json.loads, open(f"{OUT}/teacher.jsonl"))}
    ids = [i for i in teacher if i in lines and i in out_crops]
    inks = [cv2.imdecode(np.frombuffer(lines[i], np.uint8), 0).astype(np.float32) / 255 for i in ids]
    pred = rec.read_crops(inks)
    keep, stat = {"train": [], "val": []}, {}
    for i, (t, c), ink in zip(ids, pred, inks):
        tt = teacher[i]
        w_out = cv2.imdecode(np.frombuffer(out_crops[i], np.uint8), 0).shape[1]
        ok = (len(tt) >= 2 and encode(tt) is not None and ctc_ok(tt, min(ink.shape[1], w_out) // 4) and "\n" not in tt
              and Levenshtein.distance(canon(t), tt) / max(1, len(tt)) <= max_cer)
        src = i.split("/")[0]
        stat.setdefault(src, [0, 0])
        stat[src][0] += 1
        stat[src][1] += ok
        if ok:
            sp = "val" if zlib.crc32(i.split("/")[1].encode()) % 20 == 0 else "train"   # page-disjoint
            keep[sp].append((out_crops[i], tt, 1))
    for sp, items in keep.items():
        print(sp, save_store(items, f"{REC}/distill_{sp}_0"))
    print("kept per source [lines, kept]", stat)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "lines":
        build_lines(int(a[1]) if len(a) > 1 else 120)
    elif a[0] == "teach":
        teach(a[1] if len(a) > 1 else "paddlevl")
    else:
        store(a[1], float(a[2]) if len(a) > 2 else 0.3)
