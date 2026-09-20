"""Score a submission against the step-verifier benchmark items.

Submission: jsonl with one object per item: {"id": ..., "p": float}  (probability of "yes" for noul questions;
for ground items give {"id": ..., "choice": "<criteria key>"} and optionally "p" = probability of that choice).
Metrics per question: n, acc, acc_pos, acc_neg, ECE (10 bins), AUROC, selective accuracy at 80% and 90% coverage
(keep the most confident fraction), plus mean ms if the submission reports "ms" per item or --ms is given.

python bench/eval_submission.py --items bench/v0_items.jsonl --pred bench/baselines/v5b.jsonl --name V5b
"""

import argparse
import json
from collections import defaultdict


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def ece(rows, bins=10):
    n = len(rows); b = [[] for _ in range(bins)]
    for conf, hit in rows:
        b[min(bins - 1, int(conf * bins))].append((conf, hit))
    return sum(len(x) / n * abs(sum(h for _, h in x) / len(x) - sum(c for c, _ in x) / len(x)) for x in b if x)


def selective(rows, coverage):
    rows = sorted(rows, key=lambda r: -r[0]); k = max(1, int(round(len(rows) * coverage)))
    return sum(h for _, h in rows[:k]) / k


def score(items, preds, name="submission", ms=None):
    by_q = defaultdict(list); missing = 0; ms_seen = []
    for it in items:
        p = preds.get(it["id"])
        if p is None:
            missing += 1; continue
        if it["type"] == "noul":
            prob = float(p["p"]); y = int(it["label"]); pred = prob > 0.5
            conf = prob if pred else 1 - prob; hit = pred == bool(y)
            by_q[it["question"]].append({"p": prob, "y": y, "conf": conf, "hit": hit})
        else:
            choice = str(p.get("choice")); y = str(it["label"]); hit = choice == y
            conf = float(p.get("p", 1.0))
            by_q[it["question"]].append({"p": conf, "y": 1 if hit else 0, "conf": conf, "hit": hit, "choice": True})
        if "ms" in p:
            ms_seen.append(float(p["ms"]))
    report = {"_name": name, "_missing": missing, "_ms": ms if ms is not None else (sum(ms_seen) / len(ms_seen) if ms_seen else None)}
    for q, rows in by_q.items():
        n = len(rows); acc = sum(r["hit"] for r in rows) / n
        ch = [(r["conf"], r["hit"]) for r in rows]
        rep = {"n": n, "acc": round(acc, 3), "ece": round(ece(ch), 3),
               "sel80": round(selective(ch, 0.8), 3), "sel90": round(selective(ch, 0.9), 3)}
        if not rows[0].get("choice"):
            pos = [r["p"] for r in rows if r["y"]]; neg = [r["p"] for r in rows if not r["y"]]
            rep.update(auroc=round(auroc(pos, neg), 3),
                       acc_pos=round(sum(r["hit"] for r in rows if r["y"]) / max(1, len(pos)), 3),
                       acc_neg=round(sum(r["hit"] for r in rows if not r["y"]) / max(1, len(neg)), 3), n_pos=len(pos))
        report[q] = rep
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="bench/v0_items.jsonl"); ap.add_argument("--pred", required=True)
    ap.add_argument("--name", default=None); ap.add_argument("--ms", type=float, default=None)
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.items)]
    preds = {}
    for l in open(a.pred):
        d = json.loads(l); preds[d["id"]] = d
    print(json.dumps(score(items, preds, a.name or a.pred, a.ms), indent=1))


if __name__ == "__main__":
    main()
