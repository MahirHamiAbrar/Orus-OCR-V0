"""Phase 1: unify every OCR source into page manifests, then cut recognizer line crops.

  python ocr_data.py pages [src ...]   -> data/ocr/pages/<src>.jsonl
      {"image": path, "lines": [{"box": [x0,y0,x1,y1] | [cx,cy,w,h,deg], "text": str, "hand": 0|1}],
       "split": train|val|test, "src": str, "license": str}
      A crop-level source (word / line images) is one "page" per crop with box = whole image.
  python ocr_data.py crops [src ...]   -> data/ocr/rec/<src>_<split>.pkl  list of (png bytes, text, hand)
      Each line is cut with Indra's crop_line rule (32 px x tall window 3 = 96 px ink map).
Splits are writer/page disjoint (hsplit on a writer or page key); official test splits are kept.
"""
import csv
import glob
import io
import json
import os
import pickle
import re
import sys
from multiprocessing import Pool

import cv2
import numpy as np
import pyarrow.parquet as pq

from build_data import hsplit
from ocr_model import canon, encode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # sub-project root (data/ links to the main project data)
RAW, BENCH = f"{ROOT}/data/raw", f"{ROOT}/data/bench_raw"
OUT = f"{ROOT}/data/ocr"
LINE_H = int(os.environ.get("ORUS_H", "32"))        # recognizer line height in px (ink map = 3 * LINE_H tall)
REC = f"{OUT}/rec" if LINE_H == 32 else f"{OUT}/rec{LINE_H}"
SPECIAL = re.compile(r"%\w+%|\*")      # GNHK %math% %SC% %NA%, BanglaWriting struck-out '*'
MAX_W = 2048


def rel(p):
    return os.path.relpath(p, ROOT)


def page(img, lines, split, src, lic):
    return dict(image=rel(img), lines=lines, split=split, src=src, license=lic)


def dump_parquet_images(src, rows):
    """Parquet image bytes -> data/ocr/img/<src>/<i>.png (once)."""
    d = f"{OUT}/img/{src}"
    os.makedirs(d, exist_ok=True)
    paths = []
    for i, b in enumerate(rows):
        p = f"{d}/{i}.png"
        if not os.path.exists(p):
            im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(p, im)
        paths.append(p)
    return paths


def full(p):
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return [0, 0, im.shape[1], im.shape[0]] if im is not None else None


def group_lines(words):
    """Word boxes [(x0,y0,x1,y1,text)] -> lines, by vertical overlap with the running line."""
    words = sorted(words, key=lambda w: (w[1] + w[3]) / 2)
    lines = []
    for w in words:
        cy, h = (w[1] + w[3]) / 2, w[3] - w[1]
        for L in lines:
            ly0, ly1 = np.median([v[1] for v in L]), np.median([v[3] for v in L])
            if ly0 - 0.1 * h < cy < ly1 + 0.1 * h:
                L.append(w)
                break
        else:
            lines.append([w])
    out = []
    for L in lines:
        L.sort(key=lambda w: w[0])
        out.append(dict(box=[min(v[0] for v in L), min(v[1] for v in L), max(v[2] for v in L), max(v[3] for v in L)],
                        text=" ".join(v[4] for v in L), hand=1, words=[list(v[:4]) for v in L]))
    return sorted(out, key=lambda l: l["box"][1])


# ---------------- sources: yield page dicts ----------------
def src_gnhk():
    for sd in ("train", "test"):
        for f in sorted(glob.glob(f"{RAW}/gnhk/{sd}/*.jpg")):
            name = os.path.basename(f)[:-4]
            split = "test" if sd == "test" else hsplit(name, 10, 0)
            by = {}
            for w in json.load(open(f[:-4] + ".json")):
                by.setdefault(w["line_idx"], []).append(w)
            lines = []
            for ws in by.values():
                if any(SPECIAL.search(w["text"]) for w in ws):
                    continue
                pts = [(w["polygon"][f"x{i}"], w["polygon"][f"y{i}"]) for w in ws for i in range(4)]
                xs, ys = zip(*pts)
                ws.sort(key=lambda w: min(w["polygon"][f"x{i}"] for i in range(4)))
                wb = [[min(w["polygon"][f"x{i}"] for i in range(4)), min(w["polygon"][f"y{i}"] for i in range(4)),
                       max(w["polygon"][f"x{i}"] for i in range(4)), max(w["polygon"][f"y{i}"] for i in range(4))] for w in ws]
                lines.append(dict(box=[min(xs), min(ys), max(xs), max(ys)], text=" ".join(w["text"] for w in ws), hand=1,
                                  words=wb))
            yield page(f, lines, split, "gnhk", "CC BY 4.0")


def src_banglawriting():
    for f in sorted(glob.glob(f"{RAW}/banglawriting/converted/*.jpg")):
        name = os.path.basename(f)[:-4]
        ws = [(*s["points"][0], *s["points"][1], s["label"]) for s in json.load(open(f[:-4] + ".json"))["shapes"]]
        ws = [(min(a, c), min(b, d), max(a, c), max(b, d), t.strip()) for a, b, c, d, t in ws]
        lines = [l for l in group_lines(ws) if not SPECIAL.search(l["text"])]
        yield page(f, lines, hsplit(name), "banglawriting", "CC BY-NC 3.0")


def xlsx_rows(path):
    """Rows of every sheet as {col letter: value}; plain zip+XML (some BN-HTRd files break openpyxl styles)."""
    import xml.etree.ElementTree as ET
    import zipfile
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    z = zipfile.ZipFile(path)
    ss = []
    if "xl/sharedStrings.xml" in z.namelist():
        ss = ["".join(t.text or "" for t in si.iter(ns + "t")) for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(ns + "si")]
    for n in sorted(x for x in z.namelist() if x.startswith("xl/worksheets/sheet")):
        for row in ET.fromstring(z.read(n)).iter(ns + "row"):
            r = {}
            for c in row.iter(ns + "c"):
                v = c.find(ns + "v")
                t = c.find(f"{ns}is/{ns}t")
                val = ss[int(v.text)] if c.get("t") == "s" and v is not None else (t.text if t is not None else (v.text if v is not None else None))
                r[re.sub(r"\d", "", c.get("r"))] = val
            yield r


def src_bnhtrd():
    base = f"{RAW}/bnhtrd_full/BN-HTR_Dataset"
    for wdir in sorted(glob.glob(f"{base}/Recognition_Ground_Truth_Texts/*"), key=lambda p: int(os.path.basename(p))):
        w = os.path.basename(wdir)
        words = {}
        for r in (r for x in glob.glob(f"{wdir}/*.xlsx") for r in xlsx_rows(x)):
            m = re.fullmatch(r"\s*\d+_(\d+)_(\d+)_(\d+)\s*", str(r.get("A")))
            if m and r.get("B") is not None:                     # id = writer_page_line_word
                v = r["B"]
                v = v[:-2] if re.fullmatch(r"\d+\.0", v) else v
                words.setdefault((m[1], int(m[2])), []).append((int(m[3]), v.strip()))
        split = hsplit(f"bnhtrd_w{w}")
        for xml in sorted(glob.glob(f"{base}/Segmentation_Images/Lines/{w}/{w}_*.xml")):
            p = os.path.basename(xml)[:-4].split("_")[1]
            img = next((xml[:-4] + e for e in (".jpg", ".JPG", ".jpeg", ".png") if os.path.exists(xml[:-4] + e)), None)
            if img is None:
                continue
            bb = re.findall(r"<xmin>(\d+)</xmin>\s*<ymin>(\d+)</ymin>\s*<xmax>(\d+)</xmax>\s*<ymax>(\d+)</ymax>", open(xml).read())
            lines = []
            for i, b in enumerate(bb):
                t = " ".join(x for _, x in sorted(words.get((p, i + 1), [])) if x)
                b = [int(v) for v in b]
                yo = f"{base}/Segmentation_Images/Words/{w}/{w}_{p}/{w}_{p}_{i + 1}.txt"
                wb = []
                if os.path.exists(yo):                            # YOLO word boxes, relative to the line crop
                    bw, bh = b[2] - b[0], b[3] - b[1]
                    for r in open(yo):
                        v = r.split()
                        if len(v) == 5:
                            xc, yc, ww, hh = (float(z) for z in v[1:])
                            wb.append([b[0] + (xc - ww / 2) * bw, b[1] + (yc - hh / 2) * bh,
                                       b[0] + (xc + ww / 2) * bw, b[1] + (yc + hh / 2) * bh])
                lines.append(dict(box=b, text=t, hand=1, words=wb))
            yield page(img, lines, split, "bnhtrd", "CC BY 4.0")


def src_iam():
    for sp in ("train", "validation", "test"):
        t = pq.read_table(f"{RAW}/iam_line/data/{sp}.parquet").to_pylist()
        paths = dump_parquet_images(f"iam_{sp}", [d["image"]["bytes"] for d in t])
        for p, d in zip(paths, t):
            yield page(p, [dict(box=full(p), text=d["text"], hand=1)], "val" if sp == "validation" else sp, "iam",
                       "IAM non-commercial")


def src_prescription():
    base = f"{RAW}/prescription_bd/Dataset"
    for sd, sp in (("Training", "train"), ("Validation", "val"), ("Testing", "test")):
        for r in csv.DictReader(open(f"{base}/{sd}/{sd.lower()}_labels.csv")):
            p = f"{base}/{sd}/{sd.lower()}_words/{r['IMAGE']}"
            yield page(p, [dict(box=full(p), text=r["MEDICINE_NAME"], hand=1)], sp, "prescription", "CC BY 4.0")


def src_bnsynthwords():
    t = pq.read_table(glob.glob(f"{RAW}/bn_synth_words/data/*.parquet")[0]).to_pylist()
    for i, (p, d) in enumerate(zip(dump_parquet_images("bn_synth_words", [d["image"]["bytes"] for d in t]), t)):
        yield page(p, [dict(box=full(p), text=d["text"], hand=0)], hsplit(f"bsw{i}", 3, 3), "bnsynthwords", "CC BY 4.0")


def src_bnprinted():
    t = pq.read_table(glob.glob(f"{RAW}/bn_printed_val/data/*.parquet")[0]).to_pylist()
    for i, (p, d) in enumerate(zip(dump_parquet_images("bn_printed_val", [d["image"]["bytes"] for d in t]), t)):
        if d["type"] == "line":
            yield page(p, [dict(box=full(p), text=d["text"], hand=0)], hsplit(f"bnp{i}"), "bnprinted", "unstated")


def src_rupom():
    t = pq.read_table(f"{BENCH}/bn_hw_rupom/data/train-00000-of-00001.parquet").to_pylist()
    for p, d in zip(dump_parquet_images("bn_hw_rupom", [d["image"]["bytes"] for d in t]), t):
        yield page(p, [dict(box=full(p), text=d["text"], hand=1)], "test", "rupom", "unstated")


def src_imgur5k():
    base = f"{RAW}/imgur5k"
    for sp in ("train", "val", "test"):
        d = json.load(open(f"{base}/repo/dataset_info/imgur5k_annotations_{sp}.json"))
        for idx, anns in d["index_to_ann_map"].items():
            p = f"{base}/images/{idx}.jpg"
            if not os.path.exists(p):
                continue
            lines = []
            for a in anns:
                t = d["ann_id"][a]["word"]
                bb = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", d["ann_id"][a]["bounding_box"])]
                if t and t != "." and "#" not in t and len(bb) == 5:
                    lines.append(dict(box=bb, text=t, hand=1))
            yield page(p, lines, sp, "imgur5k", "CC BY-NC 4.0")


def src_iiit():
    """IIIT-INDIC-HW-WORDS bn: {split}.txt rows "<FileName>, <VocabId>", label = line VocabId (0-based) of vocab.txt."""
    base = glob.glob(f"{RAW}/iiit_bn/**/vocab.txt", recursive=True)
    d = os.path.dirname(base[0])
    vocab = open(base[0], encoding="utf-8").read().split("\n")
    for sp in ("train", "val", "test"):
        for l in open(f"{d}/{sp}.txt", encoding="utf-8"):
            f, _, v = l.strip().rpartition(",")
            if f and v.strip().isdigit():
                p = f"{d}/{f.strip()}"
                yield page(p, [dict(box=full(p), text=vocab[int(v)].strip(), hand=1)], sp, "iiit", "CC BY 4.0 (AIKosh)")


SOURCES = dict(gnhk=src_gnhk, banglawriting=src_banglawriting, bnhtrd=src_bnhtrd, iam=src_iam,
               prescription=src_prescription, bnsynthwords=src_bnsynthwords, bnprinted=src_bnprinted,
               rupom=src_rupom, imgur5k=src_imgur5k, iiit=src_iiit)


def build_pages(names):
    os.makedirs(f"{OUT}/pages", exist_ok=True)
    for n in names:
        cnt = {}
        with open(f"{OUT}/pages/{n}.jsonl", "w") as f:
            for pg in SOURCES[n]():
                f.write(json.dumps(pg, ensure_ascii=False) + "\n")
                cnt[pg["split"]] = cnt.get(pg["split"], 0) + len(pg["lines"])
        print(n, "lines per split", cnt, flush=True)


# ---------------- mmap store (RAM safe: shared page cache, no per-item Python objects) ----------------
META = np.dtype([("off", "i8"), ("n", "i4"), ("w", "i4"), ("loff", "i8"), ("ln", "i4"), ("hand", "i1")])


def save_store(items, base):
    """items: iterable of (png bytes, text, hand) -> base.blob.npy / base.meta.npy / base.lab.npy (int16 ids)."""
    blob, labs, meta = bytearray(), [], []
    lo = 0
    for png, t, hand in items:
        ids = encode(t)
        if ids is None:
            continue
        meta.append((len(blob), len(png), int.from_bytes(png[16:20], "big"), lo, len(ids), hand))
        blob += png
        labs += ids
        lo += len(ids)
    np.save(base + ".blob.npy", np.frombuffer(bytes(blob), np.uint8))
    np.save(base + ".meta.npy", np.array(meta, META))
    np.save(base + ".lab.npy", np.array(labs, np.int16))
    return len(meta)


class Store:
    """Read-only memory-mapped crop set: s[i] -> (png bytes, ids list); s.w widths; s.text(i)."""

    def __init__(self, base):
        self.blob = np.load(base + ".blob.npy", mmap_mode="r")
        self.meta = np.load(base + ".meta.npy")
        self.lab = np.load(base + ".lab.npy", mmap_mode="r")
        self.w = self.meta["w"]

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        m = self.meta[i]
        return self.blob[m["off"]:m["off"] + m["n"]].tobytes(), self.lab[m["loff"]:m["loff"] + m["ln"]].tolist()

    def text(self, i):
        from ocr_model import CHARS
        return "".join(CHARS[c - 1] for c in self[i][1])


def stores(split, names=None, rec=None):
    """{src: [Store, ...]} for <rec>/<src>_<split>[_<shard>].meta.npy (rec: data/ocr/rec, or rec48 for 48 px)"""
    out = {}
    for p in sorted(glob.glob(f"{rec or REC}/*_{split}*.meta.npy")):
        base = p[:-9]
        src = os.path.basename(base).split(f"_{split}")[0]
        if not names or src in names:
            out.setdefault(src, []).append(Store(base))
    return out


# ---------------- crops ----------------
def rot_crop(gray, cx, cy, w, h, deg):
    """Upright crop of a box rotated `deg` counter-clockwise, with page context (1 h left/right, 2 h above/below)."""
    mx, my = h, 2 * h
    M = cv2.getRotationMatrix2D((cx, cy), -deg, 1.0)
    M[:, 2] += (w / 2 + mx - cx, h / 2 + my - cy)
    out = cv2.warpAffine(gray, M, (int(w + 2 * mx), int(h + 2 * my)), flags=cv2.INTER_LINEAR, borderValue=255)
    return out, [mx, my, mx + w, my + h]


def line_rect(words):
    """Word boxes of one line -> rotated rect (cx, cy, length, text height, deg ccw) from a baseline fit."""
    w = np.array(words, np.float64)
    c = np.stack([(w[:, 0] + w[:, 2]) / 2, (w[:, 1] + w[:, 3]) / 2], 1)
    hs, ws = w[:, 3] - w[:, 1], w[:, 2] - w[:, 0]
    a = np.polyfit(c[:, 0], c[:, 1], 1)[0] if len(w) >= 2 and np.ptp(c[:, 0]) > 2 * np.median(hs) else 0.0
    a = float(np.clip(a, -0.5, 0.5))
    h = max(np.median(hs - abs(a) * ws), 0.6 * np.median(hs))     # axis box of a slanted word is too tall
    x0, x1 = w[:, 0].min(), w[:, 2].max()
    cx = (x0 + x1) / 2
    cy = np.mean(c[:, 1]) + a * (cx - np.mean(c[:, 0]))
    L = (x1 - x0) * np.sqrt(1 + a * a) + abs(a) * h
    return cx, cy, L, h * 1.1, -np.degrees(np.arctan(a))


def ctc_ok(t, steps):
    return len(t) + sum(a == b for a, b in zip(t, t[1:])) <= steps


def crop_page(pg):
    from indra_ocr.rec_preproc import crop_line
    gray = cv2.imread(f"{ROOT}/{pg['image']}", cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return []
    out = []
    for l in pg["lines"]:
        t = canon(l["text"])
        if encode(t) is None:
            continue
        g, box = gray, l["box"]
        if l.get("words"):
            g, box = rot_crop(gray, *line_rect(l["words"]))
        elif box is None:
            box = [0, 0, gray.shape[1], gray.shape[0]]
        elif len(box) == 5:
            if box[2] < 4 or box[3] < 4:
                continue
            g, box = rot_crop(gray, *box)
        if box[2] - box[0] < 2 or box[3] - box[1] < 4:
            continue
        ink = crop_line(g, box, height=LINE_H, window=3, max_width=MAX_W)
        ok, png = cv2.imencode(".png", (ink * 255).astype(np.uint8))
        if ok and ctc_ok(t, ink.shape[1] // 4):           # CTC needs a step per char + one per repeat
            out.append((png.tobytes(), t, l.get("hand", 1)))
    return out


def build_crops(names):
    os.makedirs(REC, exist_ok=True)
    for n in names:
        pages = [json.loads(l) for l in open(f"{OUT}/pages/{n}.jsonl")]
        by = {}
        with Pool(int(os.environ.get("ORUS_PROCS", 6))) as pool:
            for pg, cr in zip(pages, pool.imap(crop_page, pages, chunksize=16)):
                by.setdefault(pg["split"], []).extend(cr)
        for sp, items in by.items():
            save_store(items, f"{REC}/{n}_{sp}")
        print(n, {k: len(v) for k, v in by.items()}, flush=True)


if __name__ == "__main__":
    cmd, names = sys.argv[1], sys.argv[2:] or list(SOURCES)
    {"pages": build_pages, "crops": build_crops}[cmd](names)
