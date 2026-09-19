"""State-question rows from Mind2Web items (one annotation = ordered steps with screenshots).

Produces rows in the V5 format: {"images": [paths], "state": text, "questions": [...], "targets": {...}}.
  skip   (noul, subject = one candidate): at step t+1, the field typed at step t reappears as a candidate with
         the value already entered -> 1 (already satisfied); the gold element of the step -> 0; random other
         candidates -> 0 (they are not satisfied, they are simply not the target).
  effect (noul, two images): step t screenshot + action -> step t+1 screenshot: 1 (Mind2Web actions always
         changed the page). Negatives come from the recorder, not from here.
  done   (noul): every Mind2Web step is before the last action -> 0; positives come from the recorder.

python build_state_rows_m2w.py --items ../../model/data/vision/m2w_train_j --out ../../model/data/vision/state_rows/m2w.train.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

SKIP_INSTR = ["Is this element already in the state the task needs, so it should be left alone?",
              "Has this element already been handled for the task (no action needed on it)?"]
EFFECT_INSTR = ["Did the last action change the page as intended?", "Compare the two screenshots: did the action take effect?"]
DONE_INSTR = ["Is the task already complete on this screen?", "Has the goal been fully achieved, with nothing left to do?"]


def attrs(desc: str) -> str:
    return desc.split(":", 1)[1].strip() if ":" in desc else desc


def state_text(task, history):
    hist = "\n".join(f"  - {h}" for h in history) or "  (none)"
    return f"Task: {task}\nActions already taken:\n{hist}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="../../model/data/vision/m2w_train_j")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--neg-per-pos", type=int, default=2, help="skip=0 rows per skip=1 row")
    ap.add_argument("--done-frac", type=float, default=0.15, help="fraction of steps emitting a done=0 row")
    ap.add_argument("--effect-frac", type=float, default=0.3, help="fraction of consecutive steps emitting an effect=1 row")
    a = ap.parse_args()
    rng = random.Random(a.seed); root = Path(a.items).resolve()
    items = [json.loads(l) for l in open(root / "items.jsonl")]
    by = collections.defaultdict(dict)
    for it in items:
        by[it["id"].split("_")[0]][it["step_index"]] = it
    rows = []; n = collections.Counter()
    for ann, steps in by.items():
        for t, it in sorted(steps.items()):
            img = str(root / it["image"]); task = it["task"]; hist = it["history"]
            cands = {k: v for k, v in it["criteria"].items() if k != "none"}
            # done = 0 for every recorded step
            if rng.random() < a.done_frac:
                rows.append({"images": [img], "state": state_text(task, hist), "questions": [{"qid": "done", "qtype": "noul", "instructions": rng.choice(DONE_INSTR)}], "targets": {"done": 0}}); n["done0"] += 1
            # effect = 1: previous step's screenshot + action -> this screenshot
            prev = steps.get(t - 1)
            if prev and rng.random() < a.effect_frac and prev["gold"] != "none":
                act = f"{prev['op'].lower()} on {attrs(prev['criteria'][prev['gold']])[:60]}" + (f" typed '{prev['value']}'" if prev["op"] == "TYPE" and prev["value"] else "")
                rows.append({"images": [str(root / prev["image"]), img], "state": state_text(task, prev["history"]) + f"Last action: {act}\n",
                             "questions": [{"qid": "effect", "qtype": "noul", "instructions": rng.choice(EFFECT_INSTR)}], "targets": {"effect": 1}}); n["effect1"] += 1
            # skip: typed field at t-1 reappearing at t as a non-gold candidate
            if prev and prev["op"] == "TYPE" and prev["gold"] != "none":
                tdesc = attrs(prev["criteria"][prev["gold"]])
                hit = next((k for k, d in cands.items() if attrs(d) == tdesc and k != it["gold"]), None)
                if hit:
                    subj = f"Candidate {hit}: {cands[hit]}"  # no value hint: the filled field is visible in the screenshot only
                    rows.append({"images": [img], "state": state_text(task, hist) + subj + "\n", "questions": [{"qid": "skip", "qtype": "noul", "instructions": rng.choice(SKIP_INSTR)}], "targets": {"skip": 1}}); n["skip1"] += 1
                    negs = [k for k in cands if k not in (hit,)]
                    for k in rng.sample(negs, min(a.neg_per_pos, len(negs))):
                        subj = f"Candidate {k}: {cands[k]}"
                        rows.append({"images": [img], "state": state_text(task, hist) + subj + "\n", "questions": [{"qid": "skip", "qtype": "noul", "instructions": rng.choice(SKIP_INSTR)}], "targets": {"skip": 0}}); n["skip0"] += 1
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", len(rows), dict(n), "->", out)


if __name__ == "__main__":
    main()
