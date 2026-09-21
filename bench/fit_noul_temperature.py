"""Fit one temperature for the yes/no logit on an out-of-domain calibration eval, then report what it does elsewhere.

Inputs are eval_schema.py / baseline_frozen_vl.py JSONs (items with p = P(yes) and y). The temperature T minimises
NLL of sigmoid(logit / T) on the calibration items; it is then applied to every other eval given and ECE / AUROC / acc
are reported before and after. Writing --into <checkpoint> records "noul_temp" in that checkpoint's s1_config.json so
serve.py applies it.

python bench/fit_noul_temperature.py --calib runs/x/final/eval_gqa_val.json --apply runs/x/final/eval_general.json results/jevbench/x_hard_items.json --into runs/x/final
"""

import argparse
import json
import math
from pathlib import Path


def items_of(path):
    r = json.load(open(path))
    out = []
    for it in r.get("items", []):
        if "y" in it and "p" in it:
            p = min(max(float(it["p"]), 1e-6), 1 - 1e-6)
            out.append((math.log(p / (1 - p)), int(it["y"]), it.get("qid") or it.get("source") or ""))
    return out


def sig(z):
    z = max(-30.0, min(30.0, z))
    return min(max(1 / (1 + math.exp(-z)), 1e-9), 1 - 1e-9)


def metrics(items, T):
    ps = [sig(z / T) for z, _, _ in items]; ys = [y for _, y, _ in items]
    conf = [max(p, 1 - p) for p in ps]; hit = [int((p >= 0.5) == bool(y)) for p, y in zip(ps, ys)]
    bins = [[] for _ in range(10)]
    for c, h in zip(conf, hit):
        bins[min(9, int(c * 10))].append((c, h))
    ece = sum(len(b) / len(conf) * abs(sum(h for _, h in b) / len(b) - sum(c for c, _ in b) / len(b)) for b in bins if b)
    nll = -sum(math.log(p if y else 1 - p) for p, y in zip(ps, ys)) / len(ps)
    pos = [p for p, y in zip(ps, ys) if y]; neg = [p for p, y in zip(ps, ys) if not y]
    auroc = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg)) if pos and neg else float("nan")
    hi_err = sum(1 for c, h in zip(conf, hit) if not h and c >= 0.99)
    return {"n": len(items), "acc": round(sum(hit) / len(hit), 3), "ece": round(ece, 3), "nll": round(nll, 3), "auroc": round(auroc, 3), "err_at_99": hi_err}


def fit(items):
    best = (None, float("inf"))
    for i in range(1, 400):
        T = i / 20  # 0.05 .. 19.95
        nll = -sum(math.log(sig(z / T) if y else 1 - sig(z / T)) for z, y, _ in items) / len(items)
        if nll < best[1]:
            best = (T, nll)
    return best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True); ap.add_argument("--apply", nargs="*", default=[]); ap.add_argument("--into", default=None)
    a = ap.parse_args()
    cal = items_of(a.calib); T = fit(cal)
    print(f"calibration set n={len(cal)}  T={T}  before {metrics(cal, 1.0)}  after {metrics(cal, T)}")
    for path in a.apply:
        its = items_of(path)
        by = {}
        for z, y, q in its:
            by.setdefault(q, []).append((z, y, q))
        print(f"== {path}")
        for q, sub in sorted(by.items()):
            print(f"  {q:10} before {metrics(sub, 1.0)}\n  {' ' * 10} after  {metrics(sub, T)}")
    if a.into:
        cfg_path = Path(a.into) / "s1_config.json"; cfg = json.loads(cfg_path.read_text()); cfg["noul_temp"] = T
        cfg_path.write_text(json.dumps(cfg)); print("wrote noul_temp", T, "->", cfg_path)


if __name__ == "__main__":
    main()
