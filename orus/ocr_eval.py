"""Phase 8: page-level evaluation of Orus-OCR-0.1-alpha against Indra OCR 1.1 on CPU.

  python ocr_eval.py <orus|indra> [rec.onnx] [n=30]   -> runs/eval_pages_<tag>.json

Per page set: page CER/WER (whitespace collapsed), structure (line-count error, CER with newlines kept),
language-ID accuracy, handwritten Yes/No accuracy (Orus only: Indra has no such output), latency p50/p95,
and, on synthetic pages with pasted pictures, the share of output lines that sit on a picture.
Pages are read one at a time (RAM safe).
"""
import json
import os
import random
import re
import sys
import time

import cv2
import numpy as np
from rapidfuzz.distance import Levenshtein

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # sub-project root
RAW = f"{ROOT}/data/raw"
BENCH = f"{ROOT}/data/bench_raw"


def pages_jsonl(src, n, text=True):
    pg = [p for p in map(json.loads, open(f"{ROOT}/data/ocr/pages/{src}.jsonl")) if p["split"] == "test"]
    random.Random(0).shuffle(pg)
    for p in pg[:n]:
        yield cv2.imread(f"{ROOT}/{p['image']}", cv2.IMREAD_GRAYSCALE), "\n".join(l["text"] for l in p["lines"]) if text else None, None


def parquet_rows(path, n, text_key):
    import pyarrow.parquet as pq
    f = pq.ParquetFile(path)
    k = 0
    for i in range(f.num_row_groups):
        for row in f.read_row_group(i, columns=["image", text_key] if text_key else ["image"]).to_pylist():
            g = cv2.imdecode(np.frombuffer(row["image"]["bytes"], np.uint8), cv2.IMREAD_GRAYSCALE)
            yield g, row[text_key] if text_key else None, None
            k += 1
            if k == n:
                return


def cmater(n):
    import glob
    for p in sorted(glob.glob(f"{RAW}/cmaterdb_mixed/CMATERdb1.2.2/*.bmp"))[100:][:n]:   # BE001-100: distillation
        yield cv2.imread(p, cv2.IMREAD_GRAYSCALE), None, None


def synth_pics(n):
    from ocr_det import synth_page
    k = 0
    for i in range(10 * n):
        pics = []
        g, _ = synth_page(random.Random(10_000 + i), "test", pics)
        if pics:
            yield g, None, np.array(pics, np.float32)
            k += 1
            if k == n:
                return


# name -> (pages, expected language, expected handwritten)
SETS = {
    "bnhtrd": (lambda n: pages_jsonl("bnhtrd", n), "bn", True),
    # page text GT is incomplete here (GNHK: lines with %..% / * tokens skipped; BanglaWriting: lines grouped
    # from word boxes), so these two sets give language / handwritten / latency only; line CER is in ocr_bench
    "banglawriting": (lambda n: pages_jsonl("banglawriting", n, False), "bn", True),
    "gnhk": (lambda n: pages_jsonl("gnhk", n, False), "en", True),
    "cmater_mixed": (cmater, "bn+en", True),
    "swapnil_bn_print": (lambda n: parquet_rows(f"{BENCH}/bn_ocr_swapnil/data/train-00000-of-00001.parquet", n, "text"),
                         "bn", False),
    "cord_en_print": (lambda n: parquet_rows(f"{BENCH}/cord/data/test-00000-of-00001-9c204eb3f4e11791.parquet", n, None),
                      "en", False),
    "synth_pictures": (synth_pics, None, None),
}


def flat(t):
    return " ".join(t.split())


def lines_of(t):
    return [" ".join(l.split()) for l in t.split("\n") if l.strip()]


def on_picture(boxes, pics):
    if not len(boxes) or pics is None:
        return 0
    b = np.array(boxes, np.float32).reshape(-1, 4)
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    return int(((cx[:, None] > pics[None, :, 0]) & (cx[:, None] < pics[None, :, 2]) &
                (cy[:, None] > pics[None, :, 1]) & (cy[:, None] < pics[None, :, 3])).any(1).sum())


STAGE_MS = []                                                   # Orus per-stage timings of the current set


def system(name, rec):
    from ocr import language
    if name == "orus":
        import ocr
        o = ocr.OCR(rec=rec, threads=4)

        def run(g):
            r = o(g)
            STAGE_MS.append(r["ms"])
            return r["text"], [l["bbox"] for l in r["lines"]], r["language"], r["handwritten"] == "Yes"
        return run
    from indra_ocr import OrtOCR, model_dir
    o = OrtOCR(model_dir(None), threads=4)

    def run(g):
        p = o(g)
        return p.text, [l.box[:4] for l in p.lines], language(p.text), None
    return run


def main(name, rec=None, n=30):
    from ocr_model import canon
    fn = system(name, rec)
    fn(np.full((200, 400), 255, np.uint8))                       # warm-up
    STAGE_MS.clear()
    res = {}
    for s, (gen, lang, hand) in SETS.items():
        ms, ed, nc, wed, nw, led, lc, lerr, lang_ok, hand_ok, k, pic_lines, all_lines = ([] if i == 0 else 0 for i in range(13))
        for g, gt, pics in gen(n):
            if g is None:
                continue
            t0 = time.perf_counter()
            text, boxes, plang, phand = fn(g)
            ms.append(1000 * (time.perf_counter() - t0))
            k += 1
            if gt is not None:
                gt, text_c = ("\n".join(canon(l) for l in t.split("\n")) for t in (gt, text))
                ed += Levenshtein.distance(flat(text_c), flat(gt)); nc += len(flat(gt))
                wed += Levenshtein.distance(flat(text_c).split(), flat(gt).split()); nw += len(flat(gt).split())
                a, b = "\n".join(lines_of(text_c)), "\n".join(lines_of(gt))
                led += Levenshtein.distance(a, b); lc += len(b)
                lerr += abs(len(lines_of(text_c)) - len(lines_of(gt))) / max(1, len(lines_of(gt)))
            lang_ok += lang is not None and plang == lang
            hand_ok += hand is not None and phand is not None and phand == hand
            pic_lines += on_picture(boxes, pics)
            all_lines += len(boxes)
        r = dict(pages=k, ms_p50=round(float(np.median(ms)), 1), ms_p95=round(float(np.percentile(ms, 95)), 1))
        if STAGE_MS:
            r["stage_ms_p50"] = {k2: float(np.median([m[k2] for m in STAGE_MS])) for k2 in STAGE_MS[0]}
            STAGE_MS.clear()
        if nc:
            r.update(cer=round(ed / nc, 4), wer=round(wed / nw, 4), cer_with_newlines=round(led / lc, 4),
                     line_count_err=round(lerr / k, 4))
        if lang:
            r["lang_acc"] = round(lang_ok / k, 4)
        if hand is not None and name == "orus":
            r["hand_acc"] = round(hand_ok / k, 4)
        if s == "synth_pictures":
            r.update(lines_on_pictures=pic_lines, lines=all_lines)
        res[s] = r
        print(name, s, r, flush=True)
    tag = name if not rec else f"{name}_{os.path.basename(rec).split('.')[0]}{'_int8' if 'int8' in rec else ''}"
    json.dump(res, open(f"{ROOT}/runs/eval_pages_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1] if len(a) > 1 and a[1] != "-" else None, int(a[2]) if len(a) > 2 else 30)
