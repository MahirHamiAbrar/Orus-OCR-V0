"""Turn every raw source into tile banks: data/tiles/<source>_<split>.npz
  x: (N,128,128) uint8 tiles   y: tile label (1 hand, 0 print, -1 unknown/mixed)
  p: page index into `pages`   pages: page names   py: page label (1 = page contains handwriting)
Usage: python build_data.py [source ...]"""
import glob
import hashlib
import io
import json
import os
import random
import sys
from multiprocessing import Pool

import cv2
import numpy as np
import pyarrow.parquet as pq

import render_printed
import tiles as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # sub-project root (data/ links to the main project data)
RAW = f"{ROOT}/data/raw"
OUT = f"{ROOT}/data/tiles"
MAX_TILES_TRAIN = 32         # per page cap, keeps big pages from dominating (and RAM bounded)
MAX_TILES_EVAL = 64
RVL_PER_LABEL = 700          # printed pages per RVL class; handwritten class kept whole
RVL_PRINTED = {1, 2, 8, 9, 10, 12, 13, 14, 15}   # budget,email,memo,news,presentation,resume,sci pub/report,spec
RVL_HAND = 5


def hsplit(name, val=10, test=15):
    h = int(hashlib.md5(name.encode()).hexdigest(), 16) % 100
    return "test" if h < test else "val" if h < test + val else "train"


def decode(b, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.frombuffer(b, np.uint8), flags)


def stack_lines(lines, seed):
    """Pseudo page from line images (IAM / printed BN lines): stacked with random gaps on white."""
    r = random.Random(seed)
    lines = [T.to_gray(l) for l in lines]
    W = max(l.shape[1] for l in lines) + 40
    rows = []
    for l in lines:
        row = np.full((l.shape[0] + r.randint(0, 30), W), int(np.percentile(l, 90)), np.uint8)
        x = r.randint(0, W - l.shape[1])
        row[: l.shape[0], x:x + l.shape[1]] = l
        rows.append(row)
    return np.vstack(rows)


# ---------- page generators: yield (name, image, page_label, split, labeler or None) ----------

def src_gnhk():
    for split_dir in ("train", "test"):
        for f in sorted(glob.glob(f"{RAW}/gnhk/{split_dir}/*.jpg")):
            name = os.path.basename(f)[:-4]
            split = "test" if split_dir == "test" else ("val" if hsplit(name, 12, 0) == "val" else "train")
            yield name, f, 1, split, f[:-4] + ".json"


def src_banglawriting():
    for f in sorted(glob.glob(f"{RAW}/banglawriting/converted/*.jpg")):
        name = os.path.basename(f)[:-4]
        yield name, f, 1, hsplit(name), None


def src_bnhtrd():
    for f in sorted(glob.glob(f"{RAW}/bnhtrd/pages/*")):
        name = os.path.basename(f)
        writer = name.split("_")[0]                                   # writer-level split, no leakage
        yield f"bnhtrd_{name}", f, 1, hsplit(f"bnhtrd_w{writer}"), None


def src_iam():
    for split in ("train", "validation", "test"):
        t = pq.read_table(f"{RAW}/iam_line/data/{split}.parquet").column("image").to_pylist()
        sp = "val" if split == "validation" else split
        for i in range(0, len(t), 8):    # Teklia splits are writer-disjoint; consecutive lines ~ same form
            yield f"iam_{split}_{i}", [d["bytes"] for d in t[i:i + 8]], 1, sp, "lines"


def src_rvl():
    seen = {}
    for f in sorted(glob.glob(f"{RAW}/rvlcdip/data/*.parquet")):
        t = pq.read_table(f)
        labels = t.column("label").to_pylist()
        imgs = t.column("image")
        for i, lab in enumerate(labels):
            if lab == RVL_HAND or (lab in RVL_PRINTED and seen.get(lab, 0) < RVL_PER_LABEL):
                seen[lab] = seen.get(lab, 0) + 1
                name = f"rvl_{os.path.basename(f)[5:10]}_{i}_{lab}"
                yield name, imgs[i].as_py()["bytes"], int(lab == RVL_HAND), hsplit(name), None


def src_bnprinted():
    t = pq.read_table(glob.glob(f"{RAW}/bn_printed_val/data/*.parquet")[0]).to_pylist()
    lines = [d["image"]["bytes"] for d in t if d["type"] == "line"]
    for d_i, d in enumerate(t):
        if d["type"] == "page":
            yield f"bnp_page_{d_i}", d["image"]["bytes"], 0, hsplit(f"bnp_page_{d_i}", 10, 40), None
    for i in range(0, len(lines), 10):
        yield f"bnp_lines_{i}", lines[i:i + 10], 0, hsplit(f"bnp_lines_{i}"), "lines"


def src_synth():
    # train/val use training fonts; test uses held-out fonts only
    for split, n, off in (("train", 3000, 0), ("val", 500, 10**6), ("test", 800, 2 * 10**6)):
        for i in range(n):
            yield f"synth_{split}_{i}", ("synth", off + i, "test" if split == "test" else "train"), 0, split, None


def src_photos():
    """Non-text negatives: collages of natural photos (Imagenette). Test uses the Imagenette val folder."""
    for split, n in (("train", 2500), ("val", 300), ("test", 500)):
        for i in range(n):
            yield f"photos_{split}_{i}", ("photos", i, split), 0, split, None


def photo_collage(i, split):
    r = random.Random(f"{split}{i}")
    fs = render_printed.photo_list(split)
    W, H = r.randint(500, 1600), r.randint(400, 1400)
    page = np.full((H, W), r.choice([255, 245, 30]), np.uint8)
    for _ in range(r.randint(1, 4)):
        w, h = r.randint(W // 3, W), r.randint(H // 3, H)
        y, x = r.randint(0, H - h), r.randint(0, W - w)
        page[y:y + h, x:x + w] = cv2.resize(cv2.imread(r.choice(fs), 0), (w, h))
    return page


SOURCES = dict(photos=src_photos, gnhk=src_gnhk, banglawriting=src_banglawriting, bnhtrd=src_bnhtrd, iam=src_iam,
               rvl=src_rvl, bnprinted=src_bnprinted, synth=src_synth)


def load(img, extra):
    if isinstance(img, tuple) and img[0] == "photos":
        return photo_collage(img[1], img[2])
    if isinstance(img, tuple):                                   # synthetic
        return render_printed.make_page(img[1], img[2])[0]
    if extra == "lines":
        return stack_lines([decode(b) for b in img], hash(str(len(img))) & 0xFFFF)
    if isinstance(img, bytes):
        return decode(img)
    return cv2.imread(img)


def gnhk_masks(json_path, shape):
    m = {"H": np.zeros(shape[:2], np.uint8), "P": np.zeros(shape[:2], np.uint8)}
    for w in json.load(open(json_path)):
        p = w["polygon"]
        pts = np.array([[p[f"x{k}"], p[f"y{k}"]] for k in range(4)], np.int32)
        if w["type"] in m:
            cv2.fillPoly(m[w["type"]], [pts], 1)
    return m


def process(job):
    name, img, page_label, split, extra = job
    try:
        im = load(img, extra)
        if im is None:
            return None
        r = random.Random(name)
        jitter = r.uniform(0.8, 1.25) if split == "train" else 1.0
        g, s = T.normalize(im, jitter)
        x, pos = T.tiles(g, stride=64 if split == "train" else T.TILE)
        y = np.full(len(x), page_label, np.int8)
        if extra and extra.endswith(".json"):                   # GNHK: per-tile labels from word polygons
            m = gnhk_masks(extra, im.shape)
            mh = cv2.resize(m["H"], g.shape[::-1], interpolation=cv2.INTER_NEAREST)
            mp = cv2.resize(m["P"], g.shape[::-1], interpolation=cv2.INTER_NEAREST)
            for i, (ty, tx) in enumerate(pos):
                h = mh[ty:ty + T.TILE, tx:tx + T.TILE].mean()
                p = mp[ty:ty + T.TILE, tx:tx + T.TILE].mean()
                y[i] = 1 if h > 0.05 and p < 0.005 else 0 if p > 0.05 and h < 0.005 else -1
        if split == "train":
            keep = y >= 0
            x, y = x[keep], y[keep]
        cap = MAX_TILES_TRAIN if split == "train" else MAX_TILES_EVAL
        if len(x) > cap:
            idx = sorted(r.sample(range(len(x)), cap))
            x, y = x[idx], y[idx]
        return name, page_label, split, x, y
    except Exception as e:
        print("ERR", name, e, flush=True)
        return None


def save(d, **arrays):
    """One .npy per array so training can memory-map tiles instead of loading them whole."""
    os.makedirs(d, exist_ok=True)
    for k, v in arrays.items():
        np.save(f"{d}/{k}.npy", v)


def build(source):
    os.makedirs(OUT, exist_ok=True)
    acc = {}
    with Pool(3) as pool:
        for k, res in enumerate(pool.imap_unordered(process, SOURCES[source](), chunksize=4)):
            if res is None:
                continue
            name, pl, split, x, y = res
            a = acc.setdefault(split, dict(x=[], y=[], p=[], pages=[], py=[]))
            a["x"].append(x); a["y"].append(y); a["p"].append(np.full(len(x), len(a["pages"]), np.int32))
            a["pages"].append(name); a["py"].append(pl)
            if k % 500 == 0:
                print(source, k, flush=True)
    for split, a in acc.items():
        x = np.concatenate(a["x"]) if a["x"] else np.zeros((0, T.TILE, T.TILE), np.uint8)
        save(f"{OUT}/{source}_{split}", x=x, y=np.concatenate(a["y"]), p=np.concatenate(a["p"]),
             pages=np.array(a["pages"]), py=np.array(a["py"], np.int8))
        y = np.concatenate(a["y"])
        print(f"{source}_{split}: pages={len(a['pages'])} tiles={len(x)} hand={int((y == 1).sum())} "
              f"print={int((y == 0).sum())} unk={int((y < 0).sum())}", flush=True)


if __name__ == "__main__":
    for s in sys.argv[1:] or SOURCES:
        build(s)
