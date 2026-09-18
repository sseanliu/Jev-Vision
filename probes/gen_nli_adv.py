"""Same-format adversarial NLI rows for the calibration test (run 3d).

The student's NLI confidence is locked at ~0.75 whatever the instance (runs 3a-3c).
Hypothesis: it never saw NLI-format rows whose correct target is *uncertain*. Build
NLI rows in exactly the MNLI training format from sources the student has not seen:
  - HANS (McCoy et al.): heuristic-exploiting premise/hypothesis pairs, all three heuristics
  - MNLI train rows beyond the first 20k used by convert.py
Rows carry no 3-way gold (HANS is 2-way); Jev supplies the targets via distill_jev.py.
ANLI itself is deliberately excluded (it is the held-out set).

python gen_nli_adv.py --out ../model/data/jsonl_nli_adv --hans 4000 --mnli 6000
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset

LABELS = {"entailment": "The hypothesis follows from the premise",
          "neutral": "The hypothesis may or may not be true given the premise",
          "contradiction": "The hypothesis contradicts the premise"}
INSTR = "What is the relationship between the hypothesis and the premise?"


def row(premise, hypothesis, meta):
    return {"state": f"Premise: {premise}\nHypothesis: {hypothesis}",
            "questions": [{"qid": "q", "qtype": "choice", "instructions": INSTR, "criteria": LABELS}],
            "targets": None, **meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../model/data/jsonl_nli_adv")
    ap.add_argument("--hans", type=int, default=4000)
    ap.add_argument("--mnli", type=int, default=6000)
    ap.add_argument("--mnli-offset", type=int, default=20000, help="skip the rows convert.py used")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    hans = load_dataset("jhu-cogsci/hans", split="train", revision="refs/convert/parquet")
    by_sub = {}
    for ex in hans:
        by_sub.setdefault(ex["subcase"], []).append(ex)
    per = max(1, a.hans // len(by_sub))
    rows = []
    for sub, exs in sorted(by_sub.items()):
        for ex in rng.sample(exs, min(per, len(exs))):
            rows.append(row(ex["premise"], ex["hypothesis"],
                            {"source": "hans", "hans_label": ["entailment", "non_entailment"][ex["label"]],
                             "heuristic": ex["heuristic"], "subcase": sub}))
    rng.shuffle(rows)
    with (out / "hans.train.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"hans: {len(rows)} rows, {len(by_sub)} subcases, {Counter(r['heuristic'] for r in rows)}")

    mnli = load_dataset("nyu-mll/glue", "mnli", split="train")
    pool = [i for i in range(a.mnli_offset, len(mnli))]
    idx = rng.sample(pool, min(a.mnli * 2, len(pool)))
    rows = []
    for i in idx:
        ex = mnli[i]
        if ex["label"] < 0:
            continue
        rows.append(row(ex["premise"], ex["hypothesis"],
                        {"source": "mnli_extra", "mnli_gold": list(LABELS)[ex["label"]]}))
        if len(rows) >= a.mnli:
            break
    with (out / "mnli_extra.train.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"mnli_extra: {len(rows)} rows")


if __name__ == "__main__":
    main()
