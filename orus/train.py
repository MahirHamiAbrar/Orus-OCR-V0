"""Train a tile classifier (1 = handwritten, 0 = printed) and pick the page-level threshold.
  python train.py --model tinycnn --name tiny
  python train.py --model test_efficientnet --name teff
  python train.py --model tinycnn --teacher runs/mnv4/best.pt --name tiny_kd
  python train.py --init models/v2_teff.pt --pool data/tiles/pool --name teff_distill   (SigLIP2 page distillation)"""
import argparse
import glob
import json
import math
import os
import time

import sys

import numpy as np
import timm
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # sub-project root (data/ links to the main project data)
TILES = f"{ROOT}/data/tiles"
MEAN, STD = 0.5, 0.25
GROUP_CAP = 40000   # tiles per (source,label) group in RAM; 15 GB box


# ---------------- models ----------------
def dw(cin, cout, s):
    return nn.Sequential(nn.Conv2d(cin, cin, 3, s, 1, groups=cin, bias=False), nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
                         nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class TinyCNN(nn.Module):
    """Depthwise-separable CNN for 1x128x128 tiles. width=1.0 -> ~45k params."""
    def __init__(self, width=1.0):
        super().__init__()
        c = [max(8, int(v * width)) for v in (16, 32, 48, 64, 96, 128)]
        self.f = nn.Sequential(
            nn.Conv2d(1, c[0], 3, 2, 1, bias=False), nn.BatchNorm2d(c[0]), nn.ReLU(inplace=True),   # 64
            dw(c[0], c[1], 2), dw(c[1], c[1], 1),                                                   # 32
            dw(c[1], c[2], 2), dw(c[2], c[2], 1),                                                   # 16
            dw(c[2], c[3], 2), dw(c[3], c[3], 1),                                                   # 8
            dw(c[3], c[4], 2), dw(c[4], c[5], 1))                                                   # 4
        self.head = nn.Linear(c[5], 1)

    def forward(self, x):
        return self.head(self.f(x).mean((2, 3)))


def build_model(name, pretrained=True):
    if name.startswith("tinycnn"):
        w = float(name.split("_")[1]) if "_" in name else 1.0
        return TinyCNN(w)
    return timm.create_model(name, pretrained=pretrained, num_classes=1, in_chans=1)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ---------------- data ----------------
def load_split(split):
    """{source: {x, y, p, py, pages}}; x stays memory-mapped on disk (RAM only for what is read)."""
    out = {}
    for d in sorted(glob.glob(f"{TILES}/*_{split}/")):
        out[os.path.basename(d.rstrip("/"))[: -len(split) - 1]] = {
            k: np.load(f"{d}{k}.npy", mmap_mode="r" if k == "x" else None) for k in ("x", "y", "p", "py", "pages")}
    return out


def drop_cache(arr):
    """Tell the kernel we are done with a memory-mapped file: frees page cache, keeps memory pressure low."""
    if isinstance(arr, np.memmap):
        fd = os.open(arr.filename, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)


def train_arrays(data, dev):
    """Train tiles straight onto the GPU (chunked, so CPU RAM stays small) + sampling weights:
    each (source,label) group, capped at GROUP_CAP, gets share ~ sqrt(size); classes 50/50."""
    groups = []
    for src, d in data.items():
        for lab in (0, 1):
            idx = np.flatnonzero(d["y"] == lab)
            if len(idx) > GROUP_CAP:
                idx = np.sort(np.random.default_rng(0).choice(idx, GROUP_CAP, replace=False))
            if len(idx):
                groups.append((src, lab, idx))
    n = sum(len(i) for _, _, i in groups)
    X = torch.empty((n, *next(iter(data.values()))["x"].shape[1:]), dtype=torch.uint8, device=dev)
    Y, W = np.empty(n, np.float32), np.empty(n, np.float64)
    tot = {lab: sum(math.sqrt(len(i)) for _, l, i in groups if l == lab) for lab in (0, 1)}
    o = 0
    for src, lab, idx in groups:
        for c in range(0, len(idx), 8192):              # reads only the chosen tiles, 128 MB at a time
            part = idx[c:c + 8192]
            X[o + c:o + c + len(part)] = torch.from_numpy(data[src]["x"][part]).to(dev)
        Y[o:o + len(idx)] = lab
        W[o:o + len(idx)] = 0.5 * math.sqrt(len(idx)) / tot[lab] / len(idx)
        o += len(idx)
    for d in data.values():
        drop_cache(d["x"])
    print("train groups:", [(s, l, len(i)) for s, l, i in groups])
    return X, torch.from_numpy(Y).to(dev), torch.from_numpy(W)


def load_pool(path, dev):
    """Distillation pool (distill_pool.py): tiles (memmap) + per-page tile index (pad -1), SigLIP2 targets, holdout mask."""
    meta = json.load(open(f"{path}/meta.json"))
    x = np.load(f"{path}/x.npy", mmap_mode="r")
    n = meta["n_tiles"]
    X = x[:n]                                                # stays a memmap: GPU is shared, ~256 pool tiles read per step
    p = np.load(f"{path}/p.npy")
    cnt = np.bincount(p, minlength=meta["n_pages"])
    start = np.r_[0, np.cumsum(cnt)[:-1]]
    ar = np.arange(cnt.max())
    idx = np.where(ar < cnt[:, None], start[:, None] + ar, -1)
    src, hold, t = np.load(f"{path}/src.npy"), np.load(f"{path}/hold.npy"), np.load(f"{path}/t.npy")
    # A page capped to a random tile subset keeps a valid label only if negative (MIL): a "hand" page may have lost
    # all its handwritten tiles (e.g. a printed form with one signature), which would teach printed tiles = hand.
    ok = ~hold & ~((cnt >= cnt.max()) & (t > 0.5))
    W = np.zeros(len(src))                                   # train pages: each source share ~ sqrt(size)
    for s in np.unique(src):
        m = (src == s) & ok
        if not m.any():
            continue
        W[m] = math.sqrt(m.sum()) / m.sum()
    print("pool", meta, {s: (int((src == s).sum()), int(((src == s) & ok).sum())) for s in np.unique(src)})
    return dict(X=X, idx=torch.from_numpy(idx).to(dev), t=torch.from_numpy(np.load(f"{path}/t.npy")).to(dev),
                W=torch.from_numpy(W / W.sum()), hold=np.flatnonzero(hold), idx_np=idx, t_np=np.load(f"{path}/t.npy"))


def soft_page_score(prob, idx, frac=0.2):
    """Differentiable page_score: mean of the top ceil(frac*n) tile probabilities per page. prob: (n_valid,), idx: (P,K)."""
    valid = idx >= 0
    full = torch.full(idx.shape, -1.0, device=prob.device, dtype=prob.dtype).masked_scatter(valid, prob)
    srt = full.sort(1, descending=True).values
    k = torch.ceil(frac * valid.sum(1).float()).clamp(min=1)
    top = torch.arange(idx.shape[1], device=idx.device)[None] < k[:, None]
    return (srt * top).sum(1) / k


@torch.no_grad()
def pool_agreement(model, pool, dev):
    """Holdout pool pages: AUC of our page score against SigLIP2's call (target > 0.5), and mean |score - target|."""
    from sklearn.metrics import roc_auc_score
    h = pool["hold"]
    rows = pool["idx_np"][h]
    pr = predict_tiles(model, np.asarray(pool["X"][rows[rows >= 0]]), dev)
    sc = soft_page_score(torch.from_numpy(pr), torch.from_numpy(rows)).numpy()
    t = pool["t_np"][h]
    return float(roc_auc_score(t > 0.5, sc)), float(np.abs(sc - t).mean())


def gpu_aug(x):
    """x: (B,1,H,W) float in [0,1] on GPU. Geometry, photometric, blur, noise, resolution loss."""
    B, dev = x.shape[0], x.device
    ang = (torch.rand(B, device=dev) - 0.5) * math.radians(8)
    sc = torch.exp((torch.rand(B, device=dev) - 0.5) * 0.45)
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    tx, ty = (torch.rand(2, B, device=dev) - 0.5) * 0.3
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), padding_mode="border", align_corners=False)
    if torch.rand(1) < 0.5:                                    # resolution loss
        f = float(torch.empty(1).uniform_(0.45, 0.85))
        x = F.interpolate(F.interpolate(x, scale_factor=f, mode="area"), size=x.shape[-2:], mode="bilinear")
    if torch.rand(1) < 0.4:                                    # blur
        s = float(torch.empty(1).uniform_(0.4, 1.4))
        k = torch.exp(-torch.arange(-3, 4, device=dev, dtype=x.dtype) ** 2 / (2 * s * s)); k /= k.sum()
        x = F.conv2d(F.pad(x, (3, 3, 0, 0), mode="replicate"), k.view(1, 1, 1, 7))
        x = F.conv2d(F.pad(x, (0, 0, 3, 3), mode="replicate"), k.view(1, 1, 7, 1))
    g = torch.exp((torch.rand(B, 1, 1, 1, device=dev) - 0.5) * 0.8)
    c = 0.6 + 0.7 * torch.rand(B, 1, 1, 1, device=dev)
    b = (torch.rand(B, 1, 1, 1, device=dev) - 0.5) * 0.25
    m = x.mean((2, 3), keepdim=True)
    x = ((x.clamp(1e-4, 1) ** g - m) * c + m + b)
    x = x + torch.randn_like(x) * 0.05 * torch.rand(B, 1, 1, 1, device=dev)
    return x.clamp(0, 1)


def prep(xb, dev):
    return xb.to(dev, non_blocking=True).float().unsqueeze(1) / 255.0


# ---------------- eval ----------------
def page_score(p, frac=0.2):
    """Mean of the top 20% tile probabilities: mixed pages (printed questions + handwritten answers)
    still score high, a few false-positive tiles on a printed page get diluted."""
    if len(p) == 0:
        return 0.0
    k = max(1, math.ceil(frac * len(p)))
    return float(np.sort(p)[-k:].mean())


@torch.no_grad()
def predict_tiles(model, x, dev, bs=1024):
    model.eval()
    out = []
    for i in range(0, len(x), bs):
        xb = (prep(torch.from_numpy(np.array(x[i:i + bs])), dev) - MEAN) / STD
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.sigmoid(model(xb).float()).squeeze(1).cpu())
    return torch.cat(out).numpy() if out else np.zeros(0)


def evaluate(model, data, dev, thr=None):
    """Tile + page metrics per source. If thr is None, choose page threshold maximising balanced acc."""
    rows, pages = {}, []
    for src, d in data.items():
        p = predict_tiles(model, d["x"], dev)
        lab = d["y"] >= 0
        tile_acc = float(((p[lab] > 0.5) == (d["y"][lab] == 1)).mean()) if lab.any() else float("nan")
        sc = np.array([page_score(p[d["p"] == i]) for i in range(len(d["pages"]))])
        pages.append((src, sc, d["py"]))
        rows[src] = dict(tiles=int(lab.sum()), tile_acc=tile_acc, pages=len(sc))
    allsc = np.concatenate([s for _, s, _ in pages]); ally = np.concatenate([y for _, _, y in pages])
    def bal(t):
        pr = allsc > t
        return 0.5 * ((pr[ally == 1]).mean() + (~pr[ally == 0]).mean())
    if thr is None:
        thr = max(np.concatenate([np.linspace(0.05, 0.95, 91), np.linspace(0.955, 0.995, 9)]), key=bal)
    for src, sc, y in pages:
        rows[src]["page_acc"] = float(((sc > thr) == (y == 1)).mean())
    pr = allsc > thr
    summary = dict(thr=float(thr), page_bal_acc=float(bal(thr)),
                   hand_recall=float(pr[ally == 1].mean()), print_recall=float((~pr[ally == 0]).mean()),
                   n_hand_pages=int((ally == 1).sum()), n_print_pages=int((ally == 0).sum()))
    return summary, rows


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tinycnn")
    ap.add_argument("--name", default=None)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--steps", type=int, default=600, help="steps per epoch")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--kd_alpha", type=float, default=0.7)
    ap.add_argument("--kd_T", type=float, default=2.0)
    ap.add_argument("--init", default=None, help="start from this checkpoint (fine-tune)")
    ap.add_argument("--pool", default=None, help="distillation pool dir (distill_pool.py)")
    ap.add_argument("--pool_w", type=float, default=1.0, help="weight of the page distillation loss")
    ap.add_argument("--pool_pages", type=int, default=16, help="pool pages per step")
    ap.add_argument("--eval", default=None, help="re-tune threshold on val + re-test an existing best.pt")
    a = ap.parse_args()
    dev = "cuda"
    if a.eval:
        ck = torch.load(a.eval, map_location=dev)
        model = build_model(ck["model"], pretrained=False).to(dev)
        model.load_state_dict(ck["state"])
        ck["thr"] = evaluate(model, load_split("val"), dev)[0]["thr"]
        torch.save(ck, a.eval)
        summ, rows = evaluate(model, load_split("test"), dev, thr=ck["thr"])
        res_path = os.path.join(os.path.dirname(a.eval), "result.json")
        res = json.load(open(res_path)) if os.path.exists(res_path) else {}
        res.update(test=summ, test_sources=rows)
        json.dump(res, open(res_path, "w"), indent=1)
        print("TEST", json.dumps(summ))
        return
    torch.manual_seed(0)
    out = f"{ROOT}/runs/{a.name}"
    os.makedirs(out, exist_ok=True)
    log_f = open(f"{ROOT}/runs/{a.name}.log", "a")
    stdout = sys.stdout

    class Tee:                                          # stdout -> terminal + runs/<name>.log; tqdm bar stays on stderr
        def write(self, t): stdout.write(t); log_f.write(t); log_f.flush()
        def flush(self): stdout.flush()
    sys.stdout = Tee()

    X, Y, W = train_arrays(load_split("train"), dev)   # whole train set on GPU (~4 GB): no CPU RAM / swap pressure
    val = load_split("val")
    print(f"train tiles {len(X)}  hand={int(Y.sum())}  print={int((1 - Y).sum())}")

    pool = load_pool(a.pool, dev) if a.pool else None
    if a.init:
        ck = torch.load(a.init, map_location=dev)
        a.model = ck["model"]
        model = build_model(a.model, pretrained=False)
        model.load_state_dict(ck["state"])
        model = model.to(dev).to(memory_format=torch.channels_last)
    else:
        model = build_model(a.model).to(dev).to(memory_format=torch.channels_last)
    print(a.model, "params", n_params(model))
    teacher = None
    if a.teacher:
        ck = torch.load(a.teacher, map_location=dev)
        teacher = build_model(ck["model"], pretrained=False).to(dev).eval()
        teacher.load_state_dict(ck["state"])
    lr = a.lr or (3e-3 if a.model.startswith("tinycnn") else 1e-3)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    total = a.epochs * a.steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=total, pct_start=0.08)

    best, log = -1, []
    for ep in range(a.epochs):
        model.train()
        t0, tl = time.time(), 0.0
        idx = torch.multinomial(W, a.steps * a.bs, replacement=True).view(a.steps, a.bs).to(dev)
        if pool is not None:
            pi = torch.multinomial(pool["W"], a.steps * a.pool_pages, replacement=True).view(a.steps, a.pool_pages).to(dev)
        bar = tqdm(range(a.steps), desc=f"{a.name} ep {ep + 1}/{a.epochs}", leave=False, file=sys.stderr,
                   disable=not sys.stderr.isatty(), dynamic_ncols=True)
        for step in bar:
            bi = idx[step]
            xb = gpu_aug(prep(X[bi], dev))
            if pool is not None:                         # pool tiles share the forward pass (and BN batch stats)
                pidx = pool["idx"][pi[step]]
                xb = torch.cat([xb, gpu_aug(prep(torch.from_numpy(pool["X"][pidx[pidx >= 0].cpu().numpy()]).to(dev), dev))])
            xb = (xb - MEAN) / STD
            yb = Y[bi].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(xb.contiguous(memory_format=torch.channels_last)).float().squeeze(1)
                if pool is not None:
                    page = soft_page_score(torch.sigmoid(logit[len(bi):]), pidx).clamp(1e-6, 1 - 1e-6)
                    tgt = pool["t"][pi[step]]
                    dl = -(tgt * page.log() + (1 - tgt) * (1 - page).log()).mean()
                    logit = logit[:len(bi)]
                loss = F.binary_cross_entropy_with_logits(logit, yb, )
                if pool is not None:
                    loss = loss + a.pool_w * dl
                    xb = xb[:len(bi)]
                if teacher is not None:
                    with torch.no_grad():
                        tl_ = teacher(xb).float().squeeze(1)
                    soft = torch.sigmoid(tl_ / a.kd_T)
                    kd = F.binary_cross_entropy_with_logits(logit / a.kd_T, soft) * a.kd_T ** 2
                    loss = (1 - a.kd_alpha) * loss + a.kd_alpha * kd
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            tl += loss.item()
            if step % 20 == 0:
                bar.set_postfix(loss=f"{tl / (step + 1):.4f}", lr=f"{sched.get_last_lr()[0]:.1e}")
        summ, rows = evaluate(model, val, dev)
        tile_mean = float(np.nanmean([r["tile_acc"] for r in rows.values()]))
        score = summ["page_bal_acc"] + 0.1 * tile_mean
        if pool is not None:
            p_auc, p_mae = pool_agreement(model, pool, dev)
            score += p_auc
            summ.update(pool_auc=p_auc, pool_mae=p_mae)
            print(f"   pool holdout vs SigLIP2: AUC {p_auc:.4f}  mean|score-target| {p_mae:.4f}")
        log.append(dict(epoch=ep, loss=tl / a.steps, **summ, tile_acc_mean=tile_mean,
                        per_source={k: (round(v["tile_acc"], 4), round(v["page_acc"], 4)) for k, v in rows.items()}))
        print(f"ep {ep} loss {tl / a.steps:.4f} val page_bal {summ['page_bal_acc']:.4f} thr {summ['thr']:.2f} "
              f"tile_mean {tile_mean:.4f} {time.time() - t0:.0f}s | " +
              " ".join(f"{k}:{v['tile_acc']:.3f}/{v['page_acc']:.3f}" for k, v in rows.items()), flush=True)
        if score > best:
            best = score
            torch.save(dict(model=a.model, state=model.state_dict(), thr=summ["thr"], params=n_params(model)),
                       f"{out}/best.pt")

    ck = torch.load(f"{out}/best.pt", map_location=dev)
    model.load_state_dict(ck["state"])
    del X, Y, W, val, pool
    summ, rows = evaluate(model, load_split("test"), dev, thr=ck["thr"])
    res = dict(model=a.model, params=n_params(model), teacher=a.teacher, init=a.init, pool=a.pool, pool_w=a.pool_w, test=summ, test_sources=rows, log=log)
    json.dump(res, open(f"{out}/result.json", "w"), indent=1)
    print("TEST", json.dumps(summ), "\n", json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
