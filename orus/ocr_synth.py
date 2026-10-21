"""Phase 3: synthetic recognizer lines (and detector pages) for Orus-OCR-0.1-alpha.

  python ocr_synth.py words                 -> data/ocr/words.pkl   real handwritten word crops (bank for stitching)
  python ocr_synth.py lines <kind> <n> <split> [shard]
        kind = font   : text rendered with print + handwriting-style fonts (BN, EN, code-mixed, lists, Rx)
               stitch : real handwritten word crops (BN-HTRd, BanglaWriting, IIIT, GNHK, IMGUR5K, Rx) stitched
                        into BN / EN / mixed lines
        -> data/ocr/rec/synth_<kind>_<split>_<shard>.pkl   list of (png ink map 96 x W, text, hand)
  python ocr_synth.py demo                  -> self-check + sample sheet
Each sample is 3 stacked lines (context above / target / below) cut with Indra's crop_line(window=3),
the same rule real crops use. Splits: train/val use train fonts + train-split words; test uses held-out ones.
"""
import json
import os
import pickle
import random
import re
import sys
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import render_printed as RP
from ocr_model import canon, encode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # sub-project root (data/ links to the main project data)
OUT = f"{ROOT}/data/ocr"
BN_DIG = "০১২৩৪৫৬৭৮৯"
IS_BN = re.compile(r"[ঀ-৿]")

# ---------------- text ----------------
_WORDS = {}


def corpus(lang):
    return RP.text(lang)


_PUNCT = re.compile(r" ([,.;:!?%)\]'’]|n't|'s)")


def words_of(lang, r):
    t = r.choice(corpus(lang))
    if lang == "en":                                   # wikitext tokenization: "word , word" -> "word, word"
        t = _PUNCT.sub(r"\1", t).replace("( ", "(").replace(" @-@ ", "-")
    return t.split()


def bn_digits(s):
    return s.translate(str.maketrans("0123456789", BN_DIG))


MEDS = ["Napa", "Ace", "Seclo", "Losectil", "Fexo", "Monas", "Azithrocin", "Ciprocin", "Maxpro", "Pantonix",
        "Amodis", "Flagyl", "Histacin", "Tusca", "Ceevit", "Calbo-D", "Neuro-B", "Omeprazole", "Paracetamol",
        "Metformin", "Amlodipine", "Montelukast", "Esomeprazole", "Cefixime", "Domperidone", "Atorvastatin"]


def special_text(r):
    """Numbers, dates, prices, phone numbers, prescriptions: the strings the corpora rarely contain."""
    d = lambda a, b: str(r.randint(a, b))
    k = r.randint(0, 6)
    if k == 0:
        t = f"{r.choice(['Tab.', 'Cap.', 'Syp.', 'Inj.', 'Tab', 'T.'])} {r.choice(MEDS)} {r.choice(['5', '10', '20', '40', '250', '500'])}{r.choice(['mg', ' mg', 'mg', 'ml'])} " \
            f"{r.choice(['1+0+1', '1+1+1', '0+0+1', '1-0-1', '1+0+0', '½+0+½'])}"
        if r.random() < 0.5:
            t += " " + r.choice([f"({bn_digits(d(3, 30))} দিন)", f"x {d(3, 30)} days", "খাবার পরে", "খাবার আগে", "after meal", "--- চলবে"])
    elif k == 1:
        t = f"{d(1, 28)}/{d(1, 12)}/{d(1990, 2026)}"
    elif k == 2:
        t = r.choice(["৳ ", "Tk ", "Tk. ", "৳", "$"]) + d(5, 99999) + r.choice(["", "/-", ".00", " টাকা"])
    elif k == 3:
        t = f"01{d(3, 9)}{d(10, 99)}-{d(100000, 999999)}"
    elif k == 4:
        t = f"{r.choice(['BP', 'Pulse', 'Wt', 'Temp', 'Age'])}: {d(60, 180)}{r.choice(['/' + d(50, 110), ' kg', '°F', ' yrs', ''])}"
    elif k == 5:
        t = " ".join(r.choice(["Rx", "Dx:", "C/C:", "O/E:", "Adv:", "Investigation:", "CBC", "RBS", "X-ray", "ECG"]) for _ in range(r.randint(1, 4)))
    else:
        t = f"{d(1, 999)} {r.choice(['x', '+', '-', '=', '÷', '×'])} {d(1, 99)} = {d(1, 9999)}"
    return bn_digits(t) if r.random() < 0.3 else t


def mixed(r, base):
    """Code-mixed line: a BN fragment with EN words swapped in, or EN with BN words."""
    other = "en" if base == "bn" else "bn"
    w = words_of(base, r)
    ow = words_of(other, r)
    p = r.uniform(0.15, 0.45)
    out = [r.choice(ow) if r.random() < p else x for x in w]
    if not any(bool(IS_BN.search(x)) != (base == "bn") for x in out):
        out[r.randrange(len(out))] = r.choice(ow)
    return out


LIST = ["{}.", "{})", "({})", "{}।", "{}:"]


def list_prefix(r):
    i = r.randint(1, 20)
    k = r.random()
    if k < 0.35:
        n = str(i)
    elif k < 0.6:
        n = bn_digits(str(i))
    elif k < 0.75:
        n = "কখগঘঙচছজঝঞটঠ"[i % 12]
    elif k < 0.88:
        n = "abcdefghij"[i % 10]
    else:
        n = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii"][i % 8]
    return r.choice(["•", "-", "*", "➤"]) if r.random() < 0.15 else r.choice(LIST).format(n)


def gen_text(r):
    """One target line of 1..12 tokens; language mix ~ bn 35%, en 30%, mixed 35%."""
    k = r.random()
    if k < 0.1:
        toks = special_text(r).split()
    else:
        lang = r.choices(["bn", "en", "mixbn", "mixen"], [0.35, 0.3, 0.22, 0.13])[0]
        toks = mixed(r, lang[3:]) if lang.startswith("mix") else words_of(lang, r)
        n = r.choice([1, 1, 2, 3]) if r.random() < 0.2 else r.randint(3, 12)
        s = r.randint(0, max(0, len(toks) - n))
        toks = toks[s:s + n]
    if r.random() < 0.15:
        toks = [list_prefix(r)] + toks
    t = canon(" ".join(toks))
    return t if t and encode(t) is not None and len(t) <= 120 else gen_text(r)


# ---------------- fonts ----------------
_FONTS = None


def font_pool():
    global _FONTS
    if _FONTS is None:
        import glob, hashlib
        fl = RP.fonts()
        seen = {f["path"] for f in fl}
        from fontTools.ttLib import TTFont
        for p in sorted(glob.glob(f"{ROOT}/data/raw/bn_fonts_extra/**/*.[tTo][tTt][fF]", recursive=True)):
            if p in seen:
                continue
            try:
                cmap = TTFont(p, fontNumber=0, lazy=True).getBestCmap()
            except Exception:
                continue
            if sum(c in cmap for c in RP.BN_RANGE) <= 60:
                continue                                   # legacy (Bijoy) fonts map Bangla to ASCII codes
            has_en = all(ord(c) in cmap for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            split = "test" if int(hashlib.md5(os.path.basename(p).encode()).hexdigest(), 16) % 100 < 15 else "train"
            hand = bool(re.search(r"hand|likhan|script|cursive|brush|pen", p, re.I))
            fl.append(dict(path=p, kind="hand" if hand else "bn", bn=True, en=has_en, split=split, cmap=set(cmap)))
        for f in fl:
            if "cmap" not in f:
                try:
                    f["cmap"] = set(TTFont(f["path"], fontNumber=0, lazy=True).getBestCmap())
                except Exception:
                    f["cmap"] = set()
        _FONTS = fl
    return _FONTS


def pick_fonts(r, split):
    fl = [f for f in font_pool() if f["split"] == split]
    bn = r.choice([f for f in fl if f["bn"]])
    en_pool = [f for f in fl if f["en"]]
    if r.random() < 0.5:
        en_pool = [f for f in en_pool if f["kind"] == "hand"] or en_pool
    en = bn if bn["en"] and r.random() < 0.5 else r.choice(en_pool)
    return bn, en


def runs(t):
    """Split text into script runs: [(is_bn, str)]; spaces/punct stick to the run before them."""
    out = []
    for ch in t:
        b = bool(IS_BN.search(ch)) if ch.isalpha() or IS_BN.search(ch) else (out[-1][0] if out else False)
        if out and out[-1][0] == b:
            out[-1][1] += ch
        else:
            out.append([b, ch])
    return out


_TT = {}


def tt(path, size, bn):
    k = (path, size, bn)
    if k not in _TT:
        if len(_TT) > 300:
            _TT.clear()
        _TT[k] = ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.RAQM if bn else ImageFont.Layout.BASIC)
    return _TT[k]


def draw_text(dr, x, y, t, fbn, fen, size, space):
    """Draw mixed-script text word by word, run by run (explicit space width, so a label space is always
    visible). Falls back to the BN font when the EN font lacks a glyph. Returns (end x, fonts used)."""
    used = set()
    for wd in t.split(" "):
        for b, s in runs(wd):
            f = fbn if b else fen
            if not b and not all(ord(c) in f["cmap"] for c in s):
                f = fbn
            font = tt(f["path"], size, True)
            dr.text((x, y), s, font=font, fill=255)
            x += font.getlength(s)
            used.add(f["kind"])
        x += space
    return x, used


def text_ok(t, fbn, fen):
    for b, s in runs(t):
        f = fbn if b else fen
        if not all(ord(c) in f["cmap"] or ord(c) in fbn["cmap"] or c == " " for c in s):
            return False
    return True


# ---------------- composition ----------------
def warp(r, a, strength):
    """Elastic + shear distortion of an alpha layer (handwriting-like wobble for font text)."""
    H, W = a.shape
    sh = r.uniform(-0.3, 0.3) * strength
    M = np.float32([[1, sh, -sh * H / 2], [0, 1, 0]])
    a = cv2.warpAffine(a, M, (W, H), borderValue=0)
    if strength > 0.3:
        n = np.random.default_rng(r.randint(0, 1 << 30))
        s = max(4, int(H / 8))
        dx = cv2.resize(n.normal(0, 1, (H // s + 2, W // s + 2)).astype(np.float32), (W, H)) * strength * 2.0
        dy = cv2.resize(n.normal(0, 1, (H // s + 2, W // s + 2)).astype(np.float32), (W, H)) * strength * 2.0
        xx, yy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        a = cv2.remap(a, xx + dx, yy + dy, cv2.INTER_LINEAR, borderValue=0)
    return a


def compose(r, alpha, box):
    """Alpha layer (0..1 ink) -> degraded gray strip; returns ink map crop via Indra's rule."""
    from indra_ocr.rec_preproc import crop_line
    H, W = alpha.shape
    bg = RP.paper(r, W, H, max(8, int((box[3] - box[1]) * r.uniform(1.4, 2.2)))) if r.random() < 0.6 else \
        np.full((H, W), r.randint(215, 255), np.float32)
    ink = r.randint(0, 60) if r.random() < 0.8 else r.randint(60, 140)
    if r.random() < 0.3:                                   # uneven light
        g = np.linspace(r.uniform(0.75, 1.0), r.uniform(0.75, 1.0), W, dtype=np.float32)[None]
        bg = bg * g
    img = bg * (1 - alpha) + ink * alpha
    if r.random() < 0.5:
        img = cv2.GaussianBlur(img, (0, 0), r.uniform(0.3, 1.2))
    img = img + np.random.default_rng(r.randint(0, 1 << 30)).normal(0, r.uniform(0, 8), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if r.random() < 0.5:
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, r.randint(30, 95)])[1], 0)
    if r.random() < 0.1:
        img = np.where(img < r.randint(100, 170), 0, 255).astype(np.uint8)
    from ocr_data import LINE_H
    out = crop_line(img, box, height=LINE_H, window=3, max_width=2048)
    return out


def tight(a, thr=0.25):
    ys, xs = np.where(a > thr)
    return None if not len(xs) else [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def font_line(r, split):
    fbn, fen = pick_fonts(r, split)
    t = gen_text(r)
    if not text_ok(t, fbn, fen):
        fen = fbn
        if not text_ok(t, fbn, fen):
            return None
    size = r.randint(18, 48)
    pitch = int(size * r.uniform(1.3, 2.2))
    ctx = [gen_text(r) if r.random() < 0.75 else "" for _ in range(2)]
    W = int(max(tt(fbn["path"], size, True).getlength(s) for s in [t] + ctx) * 1.25 + size * 4)
    H = pitch * 3 + size * 2
    layers = []
    space = size * r.uniform(0.28, 0.5)
    W = int(W + space * 14)
    hand = False
    for i, s in enumerate([ctx[0], t, ctx[1]]):
        im = Image.new("L", (W, H), 0)
        if s and (i == 1 or text_ok(s, fbn, fen)):
            _, used = draw_text(ImageDraw.Draw(im), size * 2 + r.randint(-size, size) * (i != 1), size // 2 + i * pitch,
                                s, fbn, fen, size, space)
            hand = hand or (i == 1 and "hand" in used)
        layers.append(np.asarray(im, np.float32) / 255)
    strength = r.uniform(0.2, 1.0) if hand or r.random() < 0.3 else 0.0
    if r.random() < 0.5:
        ang = r.uniform(-2, 2)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1)
        layers = [cv2.warpAffine(a, M, (W, H)) for a in layers]
    layers = [warp(random.Random(r.random()), a, strength) for a in layers]   # same distortion seed per layer
    box = tight(layers[1])
    if box is None:
        return None
    alpha = np.clip(sum(layers), 0, 1)
    return compose(r, alpha, box), t, int(hand)


# ---------------- word bank + stitching ----------------
def build_words(cap=60000):
    """Real handwritten word crops from page manifests whose line text splits 1:1 onto word boxes."""
    import ocr_data as D
    bank = {"bn": {"train": [], "test": []}, "en": {"train": [], "test": []}}
    srcs = ["bnhtrd", "banglawriting", "gnhk", "imgur5k", "prescription", "iiit"]
    for src in srcs:
        p = f"{OUT}/pages/{src}.jsonl"
        if not os.path.exists(p):
            continue
        n = 0
        for l in open(p):
            pg = json.loads(l)
            sp = "test" if pg["split"] == "test" else "train" if pg["split"] == "train" else None
            if sp is None:
                continue
            gray = None
            for ln in pg["lines"]:
                ws = canon(ln["text"]).split()
                boxes = ln.get("words") or ([ln["box"]] if len(ws) == 1 else [])
                if not ws or len(ws) != len(boxes):
                    continue
                if gray is None:
                    gray = cv2.imread(f"{ROOT}/{pg['image']}", cv2.IMREAD_GRAYSCALE)
                    if gray is None:
                        break
                if boxes[0] is None:
                    boxes = [[0, 0, gray.shape[1], gray.shape[0]]]
                order = sorted(range(len(boxes)), key=lambda i: boxes[i][0]) if len(boxes[0]) == 4 else [0]
                for w, i in zip(ws, order):
                    b = boxes[i]
                    if encode(w) is None:
                        continue
                    if len(b) == 5:
                        g, bb = D.rot_crop(gray, *b)
                        g = g[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2])]
                    else:
                        x0, y0, x1, y1 = (int(round(v)) for v in b)
                        g = gray[max(0, y0):y1, max(0, x0):x1]
                    if g.size == 0 or g.shape[0] < 8 or g.shape[1] < 4:
                        continue
                    s = 48 / g.shape[0]
                    g = cv2.resize(g, (max(4, int(g.shape[1] * s)), 48), interpolation=cv2.INTER_AREA)
                    lang = "bn" if IS_BN.search(w) else "en"
                    bank[lang][sp].append((cv2.imencode(".png", g)[1].tobytes(), w))
                    n += 1
            if n >= cap:
                break
        print(src, n, flush=True)
    from ocr_data import save_store
    os.makedirs(f"{OUT}/words", exist_ok=True)
    for lang, d in bank.items():
        for sp, items in d.items():
            print(lang, sp, save_store(((p, w, 1) for p, w in items), f"{OUT}/words/{lang}_{sp}"))


_BANK = None


def bank():
    """{lang: {split: Store}} of word crops, memory-mapped (shared across Pool workers)."""
    global _BANK
    if _BANK is None:
        from ocr_data import Store
        _BANK = {l: {s: Store(f"{OUT}/words/{l}_{s}") for s in ("train", "test")} for l in ("bn", "en")}
    return _BANK


def word_alpha(png):
    g = cv2.imdecode(np.frombuffer(png, np.uint8), 0).astype(np.float32)
    bg, fg = np.percentile(g, 92), np.percentile(g, 3)
    a = np.clip((bg - g) / max(20.0, bg - fg), 0, 1)
    a[a < 0.15] = 0
    return a


def stitch_row(r, split, h, mode, n):
    """Stitch n words into one alpha row of height ~3h. Returns (alpha, text)."""
    B = bank()
    toks, imgs = [], []
    lead = "bn" if mode in ("bn", "mixbn") else "en"
    for i in range(n):
        lang = lead
        if mode.startswith("mix") and r.random() < 0.35:
            lang = "en" if lead == "bn" else "bn"
        pool = B[lang][split]
        k = r.randrange(len(pool))
        png, w = pool[k][0], pool.text(k)
        a = word_alpha(png)
        s = h * r.uniform(0.85, 1.15) / a.shape[0]
        imgs.append(cv2.resize(a, (max(2, int(a.shape[1] * s)), max(2, int(a.shape[0] * s)))))
        toks.append(w)
    if mode.startswith("mix") and all(bool(IS_BN.search(t)) == (lead == "bn") for t in toks) and n > 1:
        return stitch_row(r, split, h, mode, n)
    gap = [int(h * r.uniform(0.25, 0.7)) for _ in imgs]
    W = sum(i.shape[1] for i in imgs) + sum(gap) + 4
    H = int(h * 2)
    row = np.zeros((H, W), np.float32)
    x = 2
    for a, g in zip(imgs, gap):
        y = int((H - a.shape[0]) / 2 + r.uniform(-0.08, 0.08) * h)
        y = min(max(0, y), H - a.shape[0])
        row[y:y + a.shape[0], x:x + a.shape[1]] = np.maximum(row[y:y + a.shape[0], x:x + a.shape[1]], a)
        x += a.shape[1] + g
    return row, " ".join(toks)


def stitch_line(r, split):
    h = r.randint(28, 56)
    mode = r.choices(["bn", "en", "mixbn", "mixen"], [0.3, 0.25, 0.3, 0.15])[0]
    n = r.choice([1, 2]) if r.random() < 0.15 else r.randint(3, 10)
    rows = []
    for i in range(3):
        if i != 1 and r.random() < 0.25:
            rows.append((np.zeros((int(h * 2), 10), np.float32), ""))
        else:
            rows.append(stitch_row(r, split, h, mode, n if i == 1 else r.randint(2, 10)))
    pitch = int(h * r.uniform(1.0, 1.6))
    W = max(a.shape[1] for a, _ in rows) + 2 * h
    H = pitch * 2 + int(h * 2) + 2 * h
    alpha = np.zeros((H, W), np.float32)
    tgt = None
    for i, (a, t) in enumerate(rows):
        x, y = h + (r.randint(-h // 2, h // 2) if i != 1 else 0), h + i * pitch
        x = max(0, min(x, W - a.shape[1]))
        alpha[y:y + a.shape[0], x:x + a.shape[1]] = np.maximum(alpha[y:y + a.shape[0], x:x + a.shape[1]], a)
        if i == 1:
            layer = np.zeros_like(alpha)
            layer[y:y + a.shape[0], x:x + a.shape[1]] = a
            tgt = tight(layer)
    t = canon(rows[1][1])
    if tgt is None or encode(t) is None:
        return None
    return compose(r, alpha, tgt), t, 1


# ---------------- shards ----------------
def _one(args):
    kind, seed, split = args
    r = random.Random(f"{kind}{split}{seed}")
    try:
        out = (font_line if kind == "font" else stitch_line)(r, "test" if split == "test" else "train")
    except Exception:
        return None
    if out is None:
        return None
    ink, t, hand = out
    from ocr_data import ctc_ok
    if not ctc_ok(t, ink.shape[1] // 4):
        return None
    return cv2.imencode(".png", (ink * 255).astype(np.uint8))[1].tobytes(), t, hand


def build_lines(kind, n, split, shard=0):
    from ocr_data import REC
    os.makedirs(REC, exist_ok=True)
    off = shard * 10 ** 7
    with Pool(int(os.environ.get("ORUS_PROCS", 6))) as pool:
        items = [x for x in pool.imap(_one, ((kind, off + i, split) for i in range(n)), chunksize=64) if x]
    from ocr_data import save_store
    print(kind, split, shard, save_store(items, f"{REC}/synth_{kind}_{split}_{shard}"), "of", n, flush=True)


def demo(out=None):
    out = out or f"{OUT}/synth_demo.png"
    rows = []
    for kind in ("font", "stitch"):
        if kind == "stitch" and not os.path.exists(f"{OUT}/words/bn_train.meta.npy"):
            continue
        k = 0
        for i in range(40):
            x = _one((kind, i, "train"))
            if x is None:
                continue
            png, t, hand = x
            im = 255 - cv2.imdecode(np.frombuffer(png, np.uint8), 0)
            assert im.shape[0] % 3 == 0 and encode(t) is not None
            im = im[:, :1200]
            rows.append(cv2.copyMakeBorder(im, 0, 3, 0, 1200 - im.shape[1], cv2.BORDER_CONSTANT, value=128))
            print(kind, hand, t)
            k += 1
            if k == 6:
                break
        assert k >= 4, f"{kind}: too many failed samples"
    cv2.imwrite(out, np.vstack(rows))
    print("wrote", out)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "words":
        build_words()
    elif cmd == "lines":
        build_lines(sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 0)
    else:
        demo(*sys.argv[2:])
