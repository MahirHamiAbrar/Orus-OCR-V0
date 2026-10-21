"""Picture-region net for Orus-OCR-0.1-alpha: lines that sit on a photo / chart are dropped (images are ignored).

Indra det.onnx v1.1 has no picture channel, so this is a separate ~30k-param net on a 384 px thumbnail,
output at 1/8 (48 x 48 cells), sigmoid = picture probability.
Training data, made on the fly: real train pages (BN-HTRd, GNHK, BanglaWriting) and rendered synthetic
pages, with 0-3 photos (Imagenette train) or printed charts pasted on; target = pasted rectangles.

  python ocr_pic.py data [n=4000]        -> data/ocr/pic_{x,y}.npy (uint8 thumbnails + 1/8 masks)
  python ocr_pic.py train [steps=4000]   -> runs/orus_pic.onnx (models/ holds the frozen release net)
  python ocr_pic.py eval                 -> IoU / false-picture rate on held-out pages (Imagenette val photos)
"""
import json
import os
import random
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # sub-project root (data/ links to the main project data)
S = 384
STRIDE = 8
PAGES = ["bnhtrd", "gnhk", "banglawriting"]


def thumb(gray):
    """Gray page -> [S, S] float ink map (aspect kept, zero padded) and the scale used."""
    s = S / max(gray.shape)
    g = cv2.resize(gray, (max(1, round(gray.shape[1] * s)), max(1, round(gray.shape[0] * s))), interpolation=cv2.INTER_AREA)
    x = np.zeros((S, S), np.float32)
    x[:g.shape[0], :g.shape[1]] = 1.0 - g.astype(np.float32) / 255.0
    return x, s


def make(r, pages, split):
    import render_printed as RP
    from ocr_det import photometric, synth_page
    rects = []
    if r.random() < 0.25:
        g, _ = synth_page(r, split, rects)              # rendered page; its own pictures are targets too
        g = g.copy()
    else:
        g = cv2.imread(f"{ROOT}/{r.choice(pages)['image']}", cv2.IMREAD_GRAYSCALE)
    H, W = g.shape
    for _ in range(r.choices([0, 1, 2, 3], [0.3, 0.4, 0.2, 0.1])[0]):
        w, h = r.randint(W // 6, int(W * 0.6)), r.randint(H // 10, H // 2)
        px, py = r.randint(0, W - w), r.randint(0, H - h)
        if r.random() < 0.75:
            blk = cv2.resize(cv2.imread(r.choice(RP.photo_list(split)), 0), (w, h), interpolation=cv2.INTER_AREA)
            if r.random() < 0.3:                         # faded / photocopied photo
                blk = (blk * r.uniform(0.4, 0.9) + 255 * r.uniform(0.1, 0.5)).clip(0, 255).astype(np.uint8)
        else:
            blk = RP.chart(r, w, h)
        g[py:py + h, px:px + w] = blk
        rects.append([px, py, px + w, py + h])
    if r.random() < 0.5:
        g = photometric(r, g)
    x, s = thumb(g)
    y = np.zeros((S // STRIDE, S // STRIDE), np.float32)
    for x0, y0, x1, y1 in rects:
        y[int(round(y0 * s / STRIDE)):int(round(y1 * s / STRIDE)), int(round(x0 * s / STRIDE)):int(round(x1 * s / STRIDE))] = 1
    return x, y


def net():
    import torch.nn as nn

    def cb(i, o, s=1, d=1):
        return nn.Sequential(nn.Conv2d(i, o, 3, s, d, dilation=d, bias=False), nn.BatchNorm2d(o), nn.Hardswish())
    return nn.Sequential(cb(1, 16, 2), cb(16, 24, 2), cb(24, 32, 2), cb(32, 32, 1, 2), cb(32, 32, 1, 4),
                         cb(32, 32, 1, 8), nn.Conv2d(32, 1, 1))


def _gen(args):
    seed, n, split = args
    pages = [p for s in PAGES for p in map(json.loads, open(f"{ROOT}/data/ocr/pages/{s}.jsonl")) if p["split"] == split]
    r = random.Random(seed)
    xs, ys = [], []
    while len(xs) < n:
        try:
            x, y = make(r, pages, split)
        except Exception:
            continue
        xs.append((x * 255).astype(np.uint8))
        ys.append(y.astype(np.uint8))
    return np.stack(xs), np.stack(ys)


def build(n=4000, procs=3):
    """Pre-render n training thumbnails (making one costs ~0.5 s, too slow to do per step)."""
    from multiprocessing import Pool
    with Pool(procs) as p:
        parts = p.map(_gen, [(i, n // (procs * 4), "train") for i in range(procs * 4)])
    np.save(f"{ROOT}/data/ocr/pic_x.npy", np.concatenate([a for a, _ in parts]))
    np.save(f"{ROOT}/data/ocr/pic_y.npy", np.concatenate([b for _, b in parts]))
    print("pic data", sum(len(a) for a, _ in parts), flush=True)


def train(steps=4000):
    import torch
    import torch.nn.functional as F
    X = np.load(f"{ROOT}/data/ocr/pic_x.npy", mmap_mode="r")
    Y = np.load(f"{ROOT}/data/ocr/pic_y.npy", mmap_mode="r")
    dev = torch.device("cuda")
    m = net().to(dev)
    print("pic params", sum(p.numel() for p in m.parameters()), flush=True)
    opt = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps, pct_start=0.05)
    t0, run = time.time(), 0.0
    for step in range(1, steps + 1):
        i = np.sort(np.random.choice(len(X), 32, replace=False))
        x = torch.from_numpy(X[i]).to(dev).float() / 255
        y = torch.from_numpy(Y[i]).to(dev).float()
        if np.random.rand() < 0.5:                       # mirror: pictures and text blocks stay what they are
            x, y = x.flip(-1), y.flip(-1)
        x = x * (0.7 + 0.6 * torch.rand(len(x), 1, 1, device=dev))
        loss = F.binary_cross_entropy_with_logits(m(x[:, None])[:, 0], y, pos_weight=torch.tensor(2.0, device=dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sch.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 200 == 0:
            print(f"pic step {step} loss {run:.4f} {time.time() - t0:.0f}s", flush=True)
    m = m.cpu().eval()
    out = f"{ROOT}/runs/orus_pic.onnx"
    torch.onnx.export(m, torch.zeros(1, 1, S, S), out, input_names=["x"], output_names=["logit"], opset_version=17,
                      dynamo=False)
    print("saved", out, flush=True)
    evaluate(path=out)


class PictureNet:
    """Picture rectangles [P, 4] in page pixels."""

    def __init__(self, path=f"{ROOT}/models/orus_pic.onnx", thr=0.5, min_area=0.01):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        self.sess, self.thr, self.min_area = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"]), thr, min_area

    def __call__(self, gray):
        x, s = thumb(gray)
        p = self.sess.run(None, {"x": x[None, None]})[0][0, 0]
        mask = (p > np.log(self.thr / (1 - self.thr))).astype(np.uint8)
        n, _, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        f = STRIDE / s
        out = [[x0 * f, y0 * f, min((x0 + w) * f, gray.shape[1]), min((y0 + h) * f, gray.shape[0])]
               for x0, y0, w, h, a in st[1:n] if a * STRIDE * STRIDE / (s * s) >= self.min_area * gray.size]
        return np.array(out, np.float32).reshape(-1, 4)


def evaluate(n=200, path=f"{ROOT}/models/orus_pic.onnx"):
    """Held-out: test pages + Imagenette val photos. Cell-level IoU of the picture mask, and the share of
    pages with no pasted picture where any picture is reported (false alarms drop real text)."""
    pages = [p for s in PAGES for p in map(json.loads, open(f"{ROOT}/data/ocr/pages/{s}.jsonl")) if p["split"] == "test"]
    pn = PictureNet(path)
    r = random.Random(123)
    inter = union = fa = neg = 0
    for _ in range(n):
        x, y = make(r, pages, "test")
        p = pn.sess.run(None, {"x": x[None, None]})[0][0, 0] > 0
        inter += (p & (y > 0)).sum()
        union += (p | (y > 0)).sum()
        if y.sum() == 0:
            neg += 1
            fa += p.sum() >= 4
    res = dict(iou=round(inter / max(union, 1), 4), false_alarm_pages=round(fa / max(neg, 1), 4), neg_pages=neg)
    print("pic eval", res, flush=True)
    return res


if __name__ == "__main__":
    if sys.argv[1] == "data":
        build(int(sys.argv[2]) if len(sys.argv) > 2 else 4000)
    elif sys.argv[1] == "train":
        train(int(sys.argv[2]) if len(sys.argv) > 2 else 4000)
    else:
        evaluate(path=sys.argv[2] if len(sys.argv) > 2 else f"{ROOT}/models/orus_pic.onnx")
