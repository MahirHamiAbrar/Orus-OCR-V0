"""Markdown tables for the Orus reports from runs/bench_*.json and runs/eval_pages_*.json.

  python scripts/report_tables.py line  name=runs/bench_x.json ...   (CER / WER / ms per set)
  python scripts/report_tables.py page  name=runs/eval_pages_x.json ...
"""
import json
import sys

SETS = ["bnhtrd", "banglawriting", "iiit", "rupom", "gnhk", "iam", "imgur5k", "prescription", "mixed", "font", "bnprinted", "cord_en"]


def pct(v):
    return "–" if v is None else f"{100 * v:.1f}"


def line(systems):
    R = {n: json.load(open(p)) for n, p in systems}
    for key, title in (("cer", "CER %"), ("wer", "WER %"), ("ms_per_line", "ms / line")):
        print(f"\n**{title}** (lines per set: " + ", ".join(f"{n} {next(iter(R[n].values()))['n']}" for n, _ in systems) + ")\n")
        print("| set | " + " | ".join(n for n, _ in systems) + " |")
        print("|---|" + "---|" * len(systems))
        for s in SETS:
            row = []
            for n, _ in systems:
                v = R[n].get(s, {}).get(key)
                row.append("–" if v is None else (f"{v:.1f}" if key == "ms_per_line" else pct(v)))
            print(f"| {s} | " + " | ".join(row) + " |")
        if key != "ms_per_line":
            def mean(n, group):
                v = [R[n][s][key] for s in group if s in R[n]]
                return pct(sum(v) / len(v)) if v else "–"
            for g, group in (("mean BN hand", SETS[:4]), ("mean EN hand", SETS[4:8])):
                print(f"| **{g}** | " + " | ".join(mean(n, group) for n, _ in systems) + " |")


def page(systems):
    R = {n: json.load(open(p)) for n, p in systems}
    sets = list(next(iter(R.values())))
    cols = [("cer", "page CER %", pct), ("cer_with_newlines", "CER w/ newlines %", pct),
            ("line_count_err", "line-count err %", pct), ("lang_acc", "lang acc %", pct),
            ("hand_acc", "handwritten acc %", pct), ("ms_p50", "ms p50", str), ("ms_p95", "ms p95", str),
            ("lines_on_pictures", "lines on pictures", str)]
    for s in sets:
        used = [c for c in cols if any(c[0] in R[n].get(s, {}) for n, _ in systems)]
        print(f"\n**{s}** ({next(iter(R.values()))[s]['pages']} pages)\n")
        print("| system | " + " | ".join(t for _, t, _ in used) + " |")
        print("|---|" + "---|" * len(used))
        for n, _ in systems:
            r = R[n].get(s, {})
            print(f"| {n} | " + " | ".join(f(r[k]) if k in r else "–" for k, _, f in used)
                  + (f" (of {r['lines']} lines)" if "lines" in r else "") + " |")


if __name__ == "__main__":
    systems = [a.split("=", 1) for a in sys.argv[2:]]
    {"line": line, "page": page}[sys.argv[1]](systems)
