"""Error analysis for Orus-OCR-0.1-alpha on the line benchmark sets (data/ocr/bench/*.pkl).

  python ocr_errors.py collect <rec.pt>      -> runs/preds_<tag>.json  (gt, greedy, gated-LM, confidence per line)
  python ocr_errors.py analyze <preds.json>  -> runs/errors_<tag>.json + printed summary

What it breaks errors down into (per set):
  - CER / WER, greedy vs gated LM, and normalised WER (case + punctuation + space-before-punct ignored)
  - edit types (substitution / insertion / deletion) and which character classes they hit:
    space (word segmentation), punctuation, digit, Latin upper/lower, Bangla consonant, vowel sign (kar),
    hasanta (conjuncts), other Bangla marks
  - top character confusions (gt -> pred)
  - CER by line length and by model confidence (is confidence a usable reject signal?)
  - share of all errors that the worst 10% of lines cause (outliers / label noise / crop failures)
"""
import json
import os
import pickle
import re
import sys
from collections import Counter

import cv2
import numpy as np
from rapidfuzz.distance import Levenshtein

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # sub-project root


def collect(ckpt):
    import ocr
    import ocr_bench as B
    from indra_ocr.rec_preproc import crop_line
    from ocr_model import canon
    lm = ocr.Recognizer(ckpt)
    out = []
    for s, items in B.load_sets().items():
        crops = [crop_line(B.dec(p), b, height=lm.height, window=3) for p, b, _ in items]
        g = [lm.read_crops(crops[i:i + 64]) for i in range(0, len(crops), 64)]
        g = [x for part in g for x in part]
        lm_lm, lm.lm = lm.lm, None                         # same model, greedy only
        gr = [x for i in range(0, len(crops), 64) for x in lm.read_crops(crops[i:i + 64])]
        lm.lm = lm_lm
        for (p, b, t), (tl, c), (tg, _) in zip(items, g, gr):
            out.append(dict(set=s, gt=t, lm=canon(tl), greedy=canon(tg), conf=round(c, 4)))
        print(s, len(items), flush=True)
    tag = os.path.basename(ckpt).split(".")[0]
    json.dump(out, open(f"{ROOT}/runs/preds_{tag}.json", "w"), ensure_ascii=False)


def cls(ch):
    if ch == " ":
        return "space"
    o = ord(ch)
    if 0x0980 <= o <= 0x09FF:
        if o == 0x09CD:
            return "bn_hasanta"
        if 0x09BE <= o <= 0x09CC or o == 0x09D7:
            return "bn_vowel_sign"
        if 0x09E6 <= o <= 0x09EF:
            return "digit"
        if 0x0995 <= o <= 0x09B9 or 0x09DC <= o <= 0x09DF:
            return "bn_consonant"
        if 0x0985 <= o <= 0x0994:
            return "bn_vowel"
        return "bn_other"                                    # chandrabindu, anusvara, visarga, nukta, khanda ta, ।
    if ch.isdigit():
        return "digit"
    if ch.isupper():
        return "latin_upper"
    if ch.islower():
        return "latin_lower"
    return "punct"


PUNCT = re.compile(r"[^\w\sঀ-৿]")


def norm(t):
    """Lenient form for 'normalised WER': lower case, punctuation dropped, spaces collapsed."""
    return " ".join(PUNCT.sub(" ", t.lower()).replace("।", " ").split())


def analyze(path):
    P = json.load(open(path))
    res = {}
    for s in sorted({p["set"] for p in P}, key=[p["set"] for p in P].index):
        L = [p for p in P if p["set"] == s]
        n = sum(len(p["gt"]) for p in L)
        r = {}
        for k in ("greedy", "lm"):
            r[f"cer_{k}"] = sum(Levenshtein.distance(p[k], p["gt"]) for p in L) / n
            r[f"wer_{k}"] = sum(Levenshtein.distance(p[k].split(), p["gt"].split()) for p in L) / sum(len(p["gt"].split()) for p in L)
        nn = sum(len(norm(p["gt"]).split()) for p in L)
        r["wer_normalised"] = sum(Levenshtein.distance(norm(p["lm"]).split(), norm(p["gt"]).split()) for p in L) / max(nn, 1)
        r["cer_normalised"] = sum(Levenshtein.distance(norm(p["lm"]), norm(p["gt"])) for p in L) / max(1, sum(len(norm(p["gt"])) for p in L))
        ops, conf_pairs, by_cls, gt_cls = Counter(), Counter(), Counter(), Counter()
        for p in L:
            gt, pr = p["gt"], p["lm"]
            gt_cls.update(cls(c) for c in gt)
            for op, i, j in Levenshtein.editops(gt, pr):
                ops[op] += 1
                if op == "replace":
                    conf_pairs[(gt[i], pr[j])] += 1
                    by_cls[cls(gt[i])] += 1
                elif op == "delete":
                    by_cls[cls(gt[i])] += 1
                else:
                    by_cls["ins:" + cls(pr[j])] += 1
        tot = sum(ops.values())
        r["edit_types"] = {k: round(v / max(tot, 1), 3) for k, v in ops.items()}
        r["errors_by_class"] = {k: round(v / max(tot, 1), 3) for k, v in by_cls.most_common()}
        r["error_rate_within_class"] = {k: round(by_cls[k] / gt_cls[k], 3) for k in gt_cls if gt_cls[k] >= 30}
        r["top_confusions"] = [(a, b, c) for (a, b), c in conf_pairs.most_common(12)]
        e = np.array([Levenshtein.distance(p["lm"], p["gt"]) for p in L], float)
        r["worst10pct_share_of_errors"] = round(float(np.sort(e)[::-1][:max(1, len(e) // 10)].sum() / max(e.sum(), 1)), 3)
        lens = np.array([len(p["gt"]) for p in L])
        conf = np.array([p["conf"] for p in L])
        lcer = e / np.maximum(lens, 1)
        r["cer_by_length"] = {f"{a}-{b}": round(float(lcer[(lens >= a) & (lens < b)].mean()), 3)
                              for a, b in ((1, 10), (10, 30), (30, 60), (60, 1000)) if ((lens >= a) & (lens < b)).sum() >= 10}
        r["cer_by_conf"] = {f"{a}-{b}": [round(float(lcer[(conf >= a) & (conf < b)].mean()), 3), int(((conf >= a) & (conf < b)).sum())]
                            for a, b in ((0, 0.6), (0.6, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.01)) if ((conf >= a) & (conf < b)).sum() >= 5}
        r["worst_examples"] = [(p["gt"], p["lm"], p["conf"]) for p in sorted(L, key=lambda p: -Levenshtein.distance(p["lm"], p["gt"]) / max(1, len(p["gt"])))[:5]]
        res[s] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
        print(f"\n== {s} ({len(L)} lines)")
        for k, v in res[s].items():
            if k != "worst_examples":
                print(f"  {k}: {v}")
    tag = os.path.basename(path)[6:-5]
    json.dump(res, open(f"{ROOT}/runs/errors_{tag}.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    {"collect": collect, "analyze": analyze}[sys.argv[1]](sys.argv[2])
