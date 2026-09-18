"""Entropy-weighted selection of Jev-labelled rows.

Keeps every row where Jev's top probability is below --max-top (the model was
uncertain) and a small random share of the confident rest, so the training
distribution contains "what uncertainty looks like" instead of near-one-hot
labels (run 2b lesson).

python select_hard.py --in ../model/data/jsonl_jev --out ../model/data/jsonl_jev_hard \
    --max-top 0.8 --confident-share 0.1
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def top_p(row: dict) -> float:
    tops = []
    for q in row["questions"]:
        t = row["targets"][q["qid"]]
        tops.append(max(t, 1 - t) if q["qtype"] == "noul" else max(t.values()))
    return min(tops)  # a multi-question row counts as uncertain if any question is


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="../model/data/jsonl_jev")
    ap.add_argument("--out", default="../model/data/jsonl_jev_hard")
    ap.add_argument("--max-top", type=float, default=0.8)
    ap.add_argument("--confident-share", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kept, total = Counter(), Counter()
    for path in sorted(Path(a.inp).glob("*.train.jsonl")):
        rows = [json.loads(l) for l in open(path)]
        sel = []
        for r in rows:
            total[path.name] += 1
            if top_p(r) < a.max_top or rng.random() < a.confident_share:
                sel.append(r)
        with (out / path.name).open("w") as f:
            for r in sel:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        kept[path.name] = len(sel)
        print(f"{path.name:32s} kept {len(sel):5d} / {len(rows)}")
    print(f"total kept {sum(kept.values())} / {sum(total.values())}")


if __name__ == "__main__":
    main()
