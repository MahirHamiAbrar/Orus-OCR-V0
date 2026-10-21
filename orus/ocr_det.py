"""Phase 5: LineNet line detector for Orus-OCR-0.1-alpha (Indra det.onnx architecture, warm start).

  python ocr_det.py eval [det.onnx]          line recall / precision (IoU >= 0.5) on test pages
  python ocr_train.py det --name orus_det [--init indra|scratch] [--steps N]   (calls train() below)
Training pages: BN-HTRd, GNHK, BanglaWriting (real line boxes) + synthetic pages with pasted pictures that
carry NO line targets (pictures are out of scope), at Indra's encode_targets geometry.
"""
import json
import math
import os
import random
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # sub-project root
PAGES = ["bnhtrd", "gnhk", "banglawriting"]


def page_boxes(pg):
    out = []
    for l in pg["lines"]:
        b = l["box"]
        if l.get("words"):
            w = np.array(l["words"])
            b = [w[:, 0].min(), w[:, 1].min(), w[:, 2].max(), w[:, 3].max()]
        if b is not None and len(b) == 4:
            out.append(b)
    return np.array(out, np.float32).reshape(-1, 4)


def iou(a, b):
    x0, y0 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x1, y1 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (ar(a)[:, None] + ar(b)[None] - inter + 1e-6)


def match(pred, gt, thr=0.5):
    """Greedy one-to-one IoU matching -> (tp, n_pred, n_gt)."""
    if not len(pred) or not len(gt):
        return 0, len(pred), len(gt)
    m = iou(pred, gt)
    tp = 0
    while m.size and m.max() >= thr:
        i, j = np.unravel_index(m.argmax(), m.shape)
        tp += 1
        m[i, :] = 0
        m[:, j] = 0
    return tp, len(pred), len(gt)


def evaluate(det_path=None, n=60, split="test"):
    from indra_ocr import model_dir
    from ocr import Detector
    det = Detector(det_path or os.path.join(str(model_dir(None)), "det.onnx"))
    res = {}
    for src in PAGES:
        pages = [pg for pg in map(json.loads, open(f"{ROOT}/data/ocr/pages/{src}.jsonl")) if pg["split"] == split]
        random.Random(0).shuffle(pages)
        T = P = G = 0
        ms = 0.0
        for pg in pages[:n]:
            g = cv2.imread(f"{ROOT}/{pg['image']}", 0)
            t0 = time.perf_counter()
            b = det(g)
            ms += time.perf_counter() - t0
            t, p, gg = match(b[:, :4], page_boxes(pg))
            T, P, G = T + t, P + p, G + gg
        res[src] = dict(recall=round(T / max(G, 1), 3), precision=round(T / max(P, 1), 3), ms=round(1000 * ms / n))
        print(os.path.basename(det_path or "indra det.onnx"), src, res[src], flush=True)
    return res


# ---------------- training ----------------
def synth_page(r, split="train", pics=None):
    """Rendered page (BN / EN / mixed lines, print + hand fonts) with 0-3 pictures pasted on top.
    Returns gray page and line boxes; picture areas carry no lines (any text under them is removed)."""
    import render_printed as RP
    import ocr_synth as S
    from PIL import Image, ImageDraw
    W = r.randint(700, 1400)
    H = int(W * r.uniform(0.6, 1.5))
    fbn, fen = S.pick_fonts(r, split)
    size = r.randint(16, 44)
    pitch = int(size * r.uniform(1.4, 2.4))
    bg = RP.paper(r, W, H, pitch)
    im = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(im)
    boxes = []
    y = r.randint(10, 80)
    while y < H - pitch:
        if r.random() < 0.12:
            y += pitch
            continue
        x = r.randint(10, 120) + (r.randint(20, 120) if r.random() < 0.2 else 0)
        t = S.gen_text(r)
        if not S.text_ok(t, fbn, fen):
            y += pitch
            continue
        layer = Image.new("L", (W, H), 0)
        S.draw_text(ImageDraw.Draw(layer), x, y, t, fbn, fen, size, size * 0.35)
        a = np.asarray(layer)
        bb = S.tight(a.astype(np.float32) / 255)
        if bb and bb[2] < W - 2:
            im.paste(255, mask=layer)
            boxes.append(bb)
        y += int(pitch * r.uniform(0.9, 1.4))
    alpha = np.asarray(im, np.float32) / 255
    page = bg * (1 - alpha) + r.randint(0, 70) * alpha
    boxes = np.array(boxes, np.float32).reshape(-1, 4)
    for _ in range(r.choices([0, 1, 2, 3], [0.3, 0.4, 0.2, 0.1])[0]):          # pictures: no targets
        w, h = r.randint(W // 5, int(W * 0.6)), r.randint(H // 8, H // 2)
        px, py = r.randint(0, W - w), r.randint(0, H - h)
        blk = cv2.resize(cv2.imread(r.choice(RP.photo_list(split)), 0), (w, h)) if r.random() < 0.7 else RP.chart(r, w, h)
        page[py:py + h, px:px + w] = blk
        if pics is not None:
            pics.append([px, py, px + w, py + h])
        if len(boxes):
            cover = iou(boxes, np.array([[px, py, px + w, py + h]], np.float32))[:, 0]
            inside = (boxes[:, 0] < px + w) & (boxes[:, 2] > px) & (boxes[:, 1] < py + h) & (boxes[:, 3] > py)
            boxes = boxes[~inside]                                              # partly hidden lines: drop
    return photometric(r, page), boxes


def photometric(r, img):
    """Light, blur, noise, JPEG, sometimes binarized: no geometry, so line boxes stay exact."""
    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = img * (1 - r.uniform(0, 0.4) * ((xx / W) * r.uniform(-1, 1) + (yy / H) * r.uniform(-1, 1)) ** 2)
    img = cv2.GaussianBlur(img, (0, 0), r.uniform(0.3, 1.5))
    img = img + np.random.default_rng(r.randint(0, 1 << 30)).normal(0, r.uniform(0, 8), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if r.random() < 0.15:
        img = np.where(img < r.randint(110, 170), 0, 255).astype(np.uint8)
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, r.randint(35, 95)])[1], 0)


class DetData:
    """Random page -> (ink map input, core, dist, weight) at a random long side, cropped to 640x640."""

    def __init__(self, split="train", synth_frac=0.35, size=640):
        self.pages = []
        for src in PAGES:
            p = f"{ROOT}/data/ocr/pages/{src}.jsonl"
            self.pages += [pg for pg in map(json.loads, open(p)) if pg["split"] == split]
        self.split, self.synth_frac, self.size = split, synth_frac, size

    def sample(self, r):
        from indra_ocr.det_post import encode_targets, page_to_input
        if r.random() < self.synth_frac:
            g, b = synth_page(r, "test" if self.split == "test" else "train")
        else:
            pg = r.choice(self.pages)
            g = cv2.imread(f"{ROOT}/{pg['image']}", 0)
            b = page_boxes(pg)
        ls = r.choice([768, 896, 1024, 1024, 1152, 1280])
        x, s = page_to_input(g, ls)
        b = b * s
        S = self.size
        H, W = x.shape
        y0 = r.randint(0, max(0, H - S)) // 2 * 2
        x0 = r.randint(0, max(0, W - S)) // 2 * 2
        crop = np.zeros((S, S), np.float32)
        c = x[y0:y0 + S, x0:x0 + S]
        crop[:c.shape[0], :c.shape[1]] = c
        bb = b - [x0, y0, x0, y0]
        # lines cut by the crop edge: no loss there (ignore), fully inside: positives
        inside = (bb[:, 0] >= 0) & (bb[:, 1] >= 0) & (bb[:, 2] <= c.shape[1]) & (bb[:, 3] <= c.shape[0])
        part = ~inside & (bb[:, 2] > 0) & (bb[:, 3] > 0) & (bb[:, 0] < c.shape[1]) & (bb[:, 1] < c.shape[0])
        core, dist, wt, _ = encode_targets(bb[inside], (S, S), c.shape, ignore=bb[part])
        return crop, core, dist, wt

    def __iter__(self):
        import torch
        wi = torch.utils.data.get_worker_info()
        r = random.Random((wi.id if wi else 0) + time.time())
        while True:
            try:
                yield tuple(torch.from_numpy(np.ascontiguousarray(v)) for v in self.sample(r))
            except Exception:
                continue


def train(a):
    import onnx
    import torch
    import torch.nn.functional as F
    from indra_ocr import model_dir
    from onnx2torch import convert

    class It(torch.utils.data.IterableDataset):
        def __init__(self, d):
            self.d = d

        def __iter__(self):
            return iter(self.d)

    dev = torch.device("cuda")
    net = convert(onnx.load(os.path.join(str(model_dir(None)), "det.onnx")))
    if a.init == "scratch":
        for m in net.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
    net = net.to(dev).train()
    dl = torch.utils.data.DataLoader(It(DetData("train")), batch_size=12, num_workers=a.workers,
                                     prefetch_factor=4, persistent_workers=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    it, t0, run = iter(dl), time.time(), 0.0
    os.makedirs(f"{ROOT}/runs", exist_ok=True)
    best = -1
    for step in range(1, a.steps + 1):
        x, core, dist, wt = (v.to(dev, non_blocking=True) for v in next(it))
        x = x[:, None]
        if torch.rand(1) < 0.5:                                    # photometric jitter on the ink map
            x = (x * (0.6 + 0.6 * torch.rand(x.shape[0], 1, 1, 1, device=dev)) +
                 0.1 * torch.rand(x.shape[0], 1, 1, 1, device=dev) * torch.rand_like(x)).clamp(0, 1)
        out = net(x)
        logit, d = out[:, 0], out[:, 1:3]
        bce = (F.binary_cross_entropy_with_logits(logit, core, reduction="none") * wt).sum() / wt.sum().clamp(min=1)
        m = (core > 0.5).unsqueeze(1).float()
        l1 = (F.smooth_l1_loss(d, dist, reduction="none", beta=2.0) * m).sum() / (m.sum() * 2).clamp(min=1)
        hn = core.sum((1, 2)).clamp(min=1)
        loss = bce + 0.05 * l1
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        opt.step()
        sched.step()
        run = loss.item() if step == 1 else 0.98 * run + 0.02 * loss.item()
        if step % 100 == 0:
            print(f"det step {step} loss {run:.4f} bce {bce.item():.4f} l1 {l1.item():.3f} {time.time() - t0:.0f}s", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            path = f"{ROOT}/runs/{a.name}_s{step}.onnx"
            export(net, path)
            res = evaluate(path, n=40, split="val")
            f1 = float(np.mean([2 * v["recall"] * v["precision"] / max(1e-6, v["recall"] + v["precision"]) for v in res.values()]))
            print("det val", res, "mean F1", round(f1, 4), flush=True)
            if f1 > best:
                best = f1
                os.replace(path, f"{ROOT}/runs/{a.name}.onnx")
                print("  saved best", round(best, 4), flush=True)
            else:
                os.remove(path)
            net.train()


def export(net, path):
    """Write trained weights back into Indra's det.onnx graph (a traced export would bake in Resize sizes)."""
    import onnx
    from indra_ocr import model_dir
    from onnx import numpy_helper
    m = onnx.load(os.path.join(str(model_dir(None)), "det.onnx"))
    mods = dict(net.named_modules())
    init = {i.name: i for i in m.graph.initializer}
    for n in m.graph.node:
        if n.op_type == "Conv":
            mod = mods[n.name.lstrip("/").replace(".", "/")]
            for name, p in zip(n.input[1:], (mod.weight, mod.bias)):
                init[name].CopyFrom(numpy_helper.from_array(p.detach().float().cpu().numpy(), name))
    m.metadata_props.add(key="name", value="Orus-OCR-0.1-alpha LineNet")
    onnx.save(m, path)


if __name__ == "__main__":
    if sys.argv[1] == "eval":
        evaluate(sys.argv[2] if len(sys.argv) > 2 else None)
    elif sys.argv[1] == "show":
        r = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
        g, b = synth_page(r)
        c = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        for x0, y0, x1, y1 in b.astype(int):
            cv2.rectangle(c, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.imwrite(sys.argv[3] if len(sys.argv) > 3 else "/tmp/synth_page.jpg", c)
        print(len(b), "lines")
