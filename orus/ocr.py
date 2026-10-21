"""Orus-OCR-0.1: Bangla + English (+ mixed) OCR, handwriting first, CPU fast.

  python ocr.py IMAGE [IMAGE ...] [--json] [--rec models/orus_rec.onnx] [--det models/det.onnx]
  python ocr.py demo                                                # self-check
Pipeline: deskew -> LineNet line detector -> one BN+EN CTC recognizer (tall window) -> layout assembler.
Output: {"model", "language": bn|en|bn+en, "handwritten": "Yes"|"No", "text", "lines": [...]}
Pictures inside the page are not text: the detector is trained to give them no lines, so they never
reach the output.
"""
import json
import math
import os
import re
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # sub-project root
NAME = "Orus-OCR-0.1"
BN = re.compile(r"[ঀ-৿]")
EN = re.compile(r"[A-Za-z]")
LIST = re.compile(r"^\s*(\(?([0-9০-৯]{1,3}|[a-zA-Zক-হ]|[ivxIVX]{1,4})[.)।:]|[•\-*➤])\s+")


def _ort(path, threads):
    import onnxruntime as ort
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    o.inter_op_num_threads = 1
    return ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])


# ---------------- recognizer ----------------
class Recognizer:
    """Line recognizer from a .pt checkpoint (PyTorch, CPU/GPU) or an exported .onnx (ONNX Runtime)."""

    def __init__(self, path, threads=4, device=None, lm=f"{ROOT}/models/orus_lm.npz", alpha=0.3, beta=1.0, gate=0.9):
        from ocr_model import CHARS
        self.chars = CHARS
        self.lm = None
        if lm and os.path.exists(lm):
            from indra_ocr.lm import NgramLM
            self.lm, self.alpha, self.beta, self.gate = NgramLM(lm), alpha, beta, gate
        self.onnx = path.endswith(".onnx")
        if self.onnx:
            self.sess = _ort(path, threads)
            meta = self.sess.get_modelmeta().custom_metadata_map
            self.height = json.loads(meta["cfg"]).get("height", 32) if "cfg" in meta else 32
            self.name = meta.get("name", NAME)
        else:
            import torch
            from ocr_model import RecNet
            self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
            ck = torch.load(path, map_location=self.dev)
            self.height = ck["cfg"].get("height", 32)
            self.name = ck.get("name", NAME)
            self.net = RecNet(**ck["cfg"]).eval().to(self.dev)
            self.net.load_state_dict(ck["model"])
            torch.set_num_threads(threads)

    def logits(self, x, w):
        if self.onnx:
            return self.sess.run(["logits", "lengths"], {"x": x, "widths": w})
        import torch
        with torch.no_grad():
            lg, ln = self.net(torch.from_numpy(x).to(self.dev), torch.from_numpy(w).to(self.dev))
        return lg.float().cpu().numpy(), ln.cpu().numpy()

    def read_crops(self, crops, bs=64, area=64 * 1024):
        """Ink maps [3*height, W] -> (text, confidence) per crop. Greedy CTC, width-bucketed batches."""
        from indra_ocr.rec_preproc import pad_batch, width_batches
        from ocr_model import decode
        out = [("", 0.0)] * len(crops)
        for part in width_batches([c.shape[1] for c in crops], bs, max_area=area):
            x, w = pad_batch([crops[i] for i in part])
            lg, ln = self.logits(x, w)
            mx = lg.max(-1, keepdims=True)
            lp = lg - mx - np.log(np.exp(lg - mx).sum(-1, keepdims=True))
            pr = np.exp(lp)
            for k, i in enumerate(part):
                n = int(ln[k])
                ids = pr[k, :n].argmax(-1)
                keep = (ids != 0) & np.concatenate([[True], ids[1:] != ids[:-1]])
                conf = float(pr[k, :n].max(-1)[keep].mean()) if keep.any() else 0.0
                txt = decode(ids.tolist())
                if self.lm is not None and 0.3 <= conf < self.gate:     # gated LM rescoring of unsure lines
                    from indra_ocr.lm import beam_search
                    top = np.argsort(-lp[k, :n], -1)[:, :8]
                    tok, _ = beam_search(lp[k, :n, 0], np.take_along_axis(lp[k, :n], top, -1), top, self.lm,
                                         self.alpha, self.beta, 8)
                    txt = "".join(self.chars[c - 1] for c in tok)
                out[i] = (txt, conf)
        return out

    def read_boxes(self, items):
        """[(gray, box)] -> [text]."""
        from indra_ocr.rec_preproc import crop_line
        return [t for t, _ in self.read_crops([crop_line(g, b, height=self.height, window=3) for g, b in items])]


# ---------------- detector ----------------
class Detector:
    """LineNet ONNX (ours, or Indra's det.onnx) + Indra's decode. Returns boxes [N,5] in page pixels."""

    def __init__(self, path, threads=4, long_side=1024, pic=f"{ROOT}/models/orus_pic.onnx"):
        self.sess, self.long_side = _ort(path, threads), long_side
        self.pic = None
        if pic and os.path.exists(pic):
            from ocr_pic import PictureNet
            self.pic = PictureNet(pic)

    def __call__(self, gray):
        """Line boxes; lines centred inside a picture region (det.onnx picture channel, or models/orus_pic.onnx) are dropped,
        so text inside photos / charts / logos is ignored. Picture boxes are kept in self.pictures."""
        from indra_ocr.det_post import decode, page_to_input
        from indra_ocr.pipeline import picture_boxes
        x, s = page_to_input(gray, self.long_side)
        out = self.sess.run(None, {"x": x[None, None]})[0][0]
        b = decode(out[:3])
        if len(b):
            med = float(np.median(b[:, 3] - b[:, 1]))
            if med < 12 and len(b) >= 8:                  # tiny text: run again at a larger scale (as Indra)
                ls = int(min(2048, self.long_side * 16 / max(med, 1))) // 32 * 32
                if ls > self.long_side:
                    x, s = page_to_input(gray, ls)
                    out = self.sess.run(None, {"x": x[None, None]})[0][0]
                    b = decode(out[:3])
            b[:, :4] /= s
            b[:, [0, 2]] = b[:, [0, 2]].clip(0, gray.shape[1])
            b[:, [1, 3]] = b[:, [1, 3]].clip(0, gray.shape[0])
        if out.shape[0] > 3:
            self.pictures = picture_boxes(out[3], s, gray.shape)
        else:                                             # Indra det.onnx v1.1 has no picture channel
            self.pictures = self.pic(gray) if self.pic else np.zeros((0, 4), np.float32)
        if len(b) and len(self.pictures):
            cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
            p = self.pictures
            inside = ((cx[:, None] > p[None, :, 0]) & (cx[:, None] < p[None, :, 2]) &
                      (cy[:, None] > p[None, :, 1]) & (cy[:, None] < p[None, :, 3])).any(1)
            b = b[~inside]
        return b


# ---------------- handwritten yes/no ----------------
class HandFlag:
    """Existing 99k-param tile classifier (models/v2_teff.onnx) on the page."""

    def __init__(self, path=f"{ROOT}/models/v2_teff.onnx", threads=2):
        self.sess = _ort(path, threads)
        j = path[:-5] + ".json"                            # release folders carry the threshold as json (no torch)
        self.thr = (json.load(open(j))["thr"] if os.path.exists(j)
                    else float(__import__("torch").load(path[:-5] + ".pt", map_location="cpu")["thr"]))

    def __call__(self, img):
        import tiles as T
        from train import MEAN, STD, page_score
        g, _ = T.normalize(img)
        x, _ = T.tiles(g)
        if len(x) == 0:
            return False, 0.0
        x = ((x.astype(np.float32) / 255.0 - MEAN) / STD)[:, None]
        p = 1 / (1 + np.exp(-self.sess.run(None, {"tiles": x})[0].ravel()))
        s = float(page_score(p))
        return s > self.thr, s


# ---------------- layout ----------------
def language(text, min_chars=3, min_share=0.05):
    b, e = len(BN.findall(text)), len(EN.findall(text))
    if b + e == 0:
        return "unknown"
    has_b = b >= min_chars and b / (b + e) >= min_share
    has_e = e >= min_chars and e / (b + e) >= min_share
    return "bn+en" if has_b and has_e else "bn" if b >= e else "en"


def assemble(lines):
    """lines: [{bbox, text}] -> structured text. Reading order = rows top to bottom (boxes overlapping
    vertically share a row, left to right); indentation from the left margin; a blank line for a big gap.
    ponytail: single-column order; multi-column pages need XY-cut (Indra order.onnx) when they show up."""
    L = [l for l in lines if l["text"].strip()]
    if not L:
        return ""
    h = float(np.median([l["bbox"][3] - l["bbox"][1] for l in L]))
    L.sort(key=lambda l: (l["bbox"][1] + l["bbox"][3]) / 2)
    rows = []
    for l in L:
        cy = (l["bbox"][1] + l["bbox"][3]) / 2
        if rows and abs(cy - rows[-1]["cy"]) < 0.5 * h:
            rows[-1]["items"].append(l)
        else:
            rows.append(dict(cy=cy, items=[l]))
    left = min(l["bbox"][0] for l in L)
    cw = 0.45 * h                                          # rough glyph width for indentation / gaps
    out, prev = [], None
    for r in rows:
        it = sorted(r["items"], key=lambda l: l["bbox"][0])
        top = min(l["bbox"][1] for l in it)
        if prev is not None and top - prev > 1.0 * h:
            out.append("")
        s = " " * int(round(max(0.0, it[0]["bbox"][0] - left) / cw / 2))
        for i, l in enumerate(it):
            if i:
                gap = l["bbox"][0] - it[i - 1]["bbox"][2]
                s += " " * max(1, min(8, int(round(gap / cw / 2))))
            s += l["text"].strip()
        out.append(s.rstrip())
        prev = max(l["bbox"][3] for l in it)
    return "\n".join(out)


class OCR:
    def __init__(self, rec=None, det=None, threads=4, hand=True, release=None):
        """release: a frozen release folder (e.g. release/orus-ocr-0.1-alpha) holding det.onnx, orus_rec.onnx,
        orus_lm.npz, orus_pic.onnx, v2_teff.onnx + v2_teff.json; it overrides the models/ defaults."""
        from indra_ocr import model_dir
        if release:
            self.rec, self.det = (Recognizer(f"{release}/orus_rec.onnx", threads, lm=f"{release}/orus_lm.npz"),
                                  Detector(f"{release}/det.onnx", threads, pic=f"{release}/orus_pic.onnx"))
            self.hand = HandFlag(f"{release}/v2_teff.onnx") if hand else None
            return
        # fp32 first: dynamic INT8 (ConvInteger) is slower on CPU and costs 7-10 CER points on handwriting
        rec = rec or next((p for p in (f"{ROOT}/models/orus_rec.onnx", f"{ROOT}/models/orus_rec.int8.onnx",
                                       f"{ROOT}/runs/orus_rec.pt") if os.path.exists(p)))
        det = det or next((p for p in (f"{ROOT}/models/det.onnx", f"{ROOT}/models/orus_det.onnx") if os.path.exists(p)),
                          os.path.join(str(model_dir(None)), "det.onnx"))
        self.rec, self.det = Recognizer(rec, threads), Detector(det, threads)
        self.hand = HandFlag() if hand else None

    def __call__(self, img):
        from indra_ocr.pipeline import DESKEW_MIN, estimate_skew, preview, rotate_page
        from indra_ocr.rec_preproc import crop_line
        if isinstance(img, str):
            img = cv2.imread(img)
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        t0 = time.perf_counter()
        a = estimate_skew(preview(gray))
        if abs(a) >= DESKEW_MIN:
            gray = rotate_page(gray, a)
        boxes = self.det(gray)
        t1 = time.perf_counter()
        res = self.rec.read_crops([crop_line(gray, b[:4], height=self.rec.height, window=3) for b in boxes])
        t2 = time.perf_counter()
        lines = [dict(bbox=[round(float(v), 1) for v in b[:4]], text=t, conf=round(c, 3), lang=language(t))
                 for b, (t, c) in zip(boxes, res) if c >= 0.35 and t.strip()]
        text = assemble(lines)
        hw, hs = self.hand(img) if self.hand is not None else (None, None)
        t3 = time.perf_counter()
        return dict(model=self.rec.name, language=language(text), handwritten=None if hw is None else ("Yes" if hw else "No"),
                    handwritten_score=None if hs is None else round(hs, 3), text=text, lines=lines,
                    ms=dict(detect=round(1000 * (t1 - t0)), recognize=round(1000 * (t2 - t1)),
                            handwritten=round(1000 * (t3 - t2)), total=round(1000 * (t3 - t0))))


# ---------------- export ----------------
def export(ckpt, out=None):
    import onnx
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from ocr_model import RecNet
    out = out or f"{ROOT}/runs/orus_rec.onnx"          # models/ holds the frozen release
    ck = torch.load(ckpt, map_location="cpu")
    net = RecNet(**ck["cfg"]).eval()
    net.load_state_dict(ck["model"])
    x, w = torch.zeros(2, 1, 3 * net.height, 256), torch.tensor([256, 200])
    torch.onnx.export(net, (x, w), out, input_names=["x", "widths"], output_names=["logits", "lengths"],
                      dynamic_axes={"x": {0: "B", 3: "W"}, "widths": {0: "B"}, "logits": {0: "B", 1: "T"},
                                    "lengths": {0: "B"}}, opset_version=17, dynamo=False)
    m = onnx.load(out)
    for k, v in dict(name=NAME, vocab=json.dumps(ck["chars"], ensure_ascii=False), cfg=json.dumps(ck["cfg"]),
                     val=json.dumps(ck.get("val", {}))).items():
        m.metadata_props.add(key=k, value=v)
    onnx.save(m, out)
    q = out[:-5] + ".int8.onnx"
    quantize_dynamic(out, q, weight_type=QuantType.QUInt8, op_types_to_quantize=["Conv", "MatMul"])  # ORT CPU ConvInteger: uint8 only
    # check: ONNX fp32 == PyTorch, INT8 close
    import onnxruntime as ort
    xr = (torch.rand(3, 1, 3 * net.height, 320) > 0.85).float()
    wr = torch.tensor([320, 250, 100])
    with torch.no_grad():
        ref = torch.log_softmax(net(xr, wr)[0], -1).numpy()
    for p in (out, q):
        lg = ort.InferenceSession(p, providers=["CPUExecutionProvider"]).run(["logits"], {"x": xr.numpy(), "widths": wr.numpy()})[0]
        lp = lg - np.log(np.exp(lg - lg.max(-1, keepdims=True)).sum(-1, keepdims=True)) - lg.max(-1, keepdims=True)
        agree = float((lp.argmax(-1) == ref.argmax(-1)).mean())
        print(os.path.basename(p), f"{os.path.getsize(p) / 1e6:.2f} MB", "argmax agreement vs torch", round(agree, 4))
        assert agree > (0.999 if p == out else 0.97)


def demo(release=None):
    """Self-check: language rules, assembler structure, and (if a model exists) a rendered mixed page."""
    assert language("আমি rice খাই and tea") == "bn+en" and language("hello world") == "en" and language("আমি ভাত") == "bn"
    L = [dict(bbox=[10, 10, 300, 30], text="Shopping list"), dict(bbox=[30, 40, 200, 60], text="1. চাল"),
         dict(bbox=[30, 70, 200, 90], text="2. eggs"), dict(bbox=[10, 140, 300, 160], text="শেষ")]
    t = assemble(L)
    assert t.split("\n")[0] == "Shopping list" and t.split("\n")[1].startswith(" ") and "" in t.split("\n"), t
    print(t)
    try:
        o = OCR(release=release) if release else OCR()
    except StopIteration:
        print("no trained recognizer yet: model check skipped")
        return
    from PIL import Image, ImageDraw, ImageFont
    font = next((p for p in (f"{ROOT}/fonts/HindSiliguri-Regular.ttf",
                              f"{ROOT}/data/fonts/bn/HindSiliguri-Regular.ttf") if os.path.exists(p)), None)
    if font is None:
        print("no Bangla font for the rendered page: model check skipped")
        return
    im = Image.new("L", (900, 420), 255)
    dr = ImageDraw.Draw(im)
    f = ImageFont.truetype(font, 34, layout_engine=ImageFont.Layout.RAQM)
    gt = ["আজকের বাজার list", "১. চাল ৫ kg", "২. ডিম 12 টি"]
    for i, s in enumerate(gt):
        dr.text((40 + 40 * (i > 0), 40 + 70 * i), s, font=f, fill=0)
    page = np.array(im)
    page[300:400, 500:850] = cv2.resize(cv2.imread(f"{ROOT}/data/raw/imagenette/imagenette2-320/val/n01440764/ILSVRC2012_val_00009111.JPEG", 0)
                                        if os.path.exists(f"{ROOT}/data/raw/imagenette/imagenette2-320/val/n01440764/ILSVRC2012_val_00009111.JPEG")
                                        else np.full((100, 350), 90, np.uint8), (350, 100))
    r = o(cv2.cvtColor(page, cv2.COLOR_GRAY2BGR))
    print(json.dumps({k: v for k, v in r.items() if k != "lines"}, ensure_ascii=False, indent=1))
    from rapidfuzz.distance import Levenshtein
    cer = Levenshtein.distance(" ".join(r["text"].split()), " ".join(" ".join(gt).split())) / len(" ".join(gt))
    print("demo CER", round(cer, 3))
    assert r["language"] == "bn+en" and r["handwritten"] == "No" and cer < 0.15, (r["language"], r["handwritten"], cer)


def main():
    a = sys.argv[1:]
    if a and a[0] == "export":
        return export(*a[1:])
    if not a or a[0] == "demo":
        return demo()
    kw = {}
    for k in ("--rec", "--det", "--release"):
        if k in a:
            i = a.index(k)
            kw[k[2:]] = a[i + 1]
            del a[i:i + 2]
    as_json = "--json" in a
    a = [x for x in a if x != "--json"]
    o = OCR(**kw)
    for f in a:
        r = o(f)
        if as_json:
            print(json.dumps(r, ensure_ascii=False, indent=1))
        else:
            print(f"== {f}\nLanguage: {r['language']}\nHandwritten: {r['handwritten']}\n{r['text']}\n({r['ms']['total']} ms)")


if __name__ == "__main__":
    main()
