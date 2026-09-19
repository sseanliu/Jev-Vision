"""Recorder steps (harness/record_triplets.py) -> V5 state rows.

From each recorded step:
  effect (two images + last action): labels.effect, skipping noisy steps
  done   (after image): labels.done_after; also done_before on the before image
  skip   (before image, one candidate as subject): labels.skip[k] for a sample of candidates, balanced
  ground (before image, choice): the goal's target element when known (bonus grounding on new sites)
Site-level split: --val-sites N puts the last N sites of the list into validation.

python build_state_rows_record.py --steps ../../model/data/vision/triplets/run1/steps.jsonl --out-dir ../../model/data/vision/state_rows --tag rec1 --val-sites 30
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "harness"))
from record_triplets import search_done  # strict URL rule

SKIP_INSTR = ["Is this element already in the state the task needs, so it should be left alone?",
              "Has this element already been handled for the task (no action needed on it)?"]
EFFECT_INSTR = ["Did the last action change the page as intended?", "Compare the two screenshots: did the action take effect?"]
DONE_INSTR = ["Is the task already complete on this screen?", "Has the goal been fully achieved, with nothing left to do?"]
GROUND_INSTR = "Which numbered element should be acted on next to make progress on the task?"


def _field_value(cand: str) -> str:
    import re
    m = re.search(r" value='([^']*)'$", cand); return m.group(1) if m else ""


def relabel(recs):
    """Recompute done_before / done_after (search goals: strict URL rule; sticky within a task) and the skip labels
    that depend on done_before. Follow / click goals keep their recorded labels."""
    n_flip = 0
    by_task = {}
    for r in recs:
        by_task.setdefault((r["site"], r["id"].rsplit("_", 1)[0]), []).append(r)
    for task, steps in by_task.items():
        steps.sort(key=lambda r: r["step"]); reached = False
        for r in steps:
            if r["goal_kind"] != "search":
                continue
            L = r["labels"]; old = (L["done_before"], L["done_after"])
            done_before = int(reached or search_done(r["before_url"], r["goal_value"]))
            reached = bool(done_before) or search_done(r["after_url"], r["goal_value"])
            L["done_before"] = done_before; L["done_after"] = int(reached)
            for k, cand in r["candidates"].items():
                val_ok = (r["target_idx"] == k) and _field_value(cand).strip().lower() == r["goal_value"].lower()
                L["skip"][k] = int(val_ok or done_before)
            n_flip += (old != (L["done_before"], L["done_after"]))
    print("done labels flipped on", n_flip, "search steps")
    return recs


def state_text(goal, history):
    hist = "\n".join(f"  - {h}" for h in history) or "  (none)"
    return f"Task: {goal}\nActions already taken:\n{hist}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", required=True); ap.add_argument("--out-dir", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--val-sites", type=int, default=30); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-per-step", type=int, default=2)
    a = ap.parse_args(); rng = random.Random(a.seed)
    steps_path = Path(a.steps).resolve(); root = steps_path.parent
    recs = [json.loads(l) for l in open(steps_path)]; recs = [r for r in recs if "error" not in r]
    recs = relabel(recs); print("relabelled done/skip with the strict search rule")
    sites = []
    for r in recs:
        if r["site"] not in sites:
            sites.append(r["site"])
    val_sites = set(sites[-a.val_sites:]) if a.val_sites else set()
    out = {"train": [], "validation": []}; n = collections.Counter()
    for r in recs:
        split = "validation" if r["site"] in val_sites else "train"
        before = str(root / r["before_img"]); after = str(root / r["after_img"]); st = state_text(r["goal"], r["history"])
        L = r["labels"]
        tb = "\n".join(r["candidates"].values()); ta = "\n".join(r.get("candidates_after", {}).values())
        meta = {"site": r["site"], "step_id": r["id"], "text_before": tb, "text_after": ta, "url_before": r["before_url"], "url_after": r["after_url"], "goal_kind": r["goal_kind"]}
        # effect
        if not L["noisy"]:
            act = f"{r['action']} on {r['candidates'].get(r['chosen'], '?')}" + (f" typed '{r['typed']}'" if r["typed"] else "")
            out[split].append({"images": [before, after], "state": st + f"Last action: {act}\n", "questions": [{"qid": "effect", "qtype": "noul", "instructions": rng.choice(EFFECT_INSTR)}], "targets": {"effect": L["effect"]}, "meta": meta}); n[f"{split}/effect{L['effect']}"] += 1
        # done: before image (done_before) and after image (done_after)
        out[split].append({"images": [before], "state": st, "questions": [{"qid": "done", "qtype": "noul", "instructions": rng.choice(DONE_INSTR)}], "targets": {"done": L["done_before"]}, "meta": {**meta, "which": "before"}}); n[f"{split}/done{L['done_before']}"] += 1
        hist_after = r["history"] + [f"{r['action']} on {r['candidates'].get(r['chosen'], '?')[:40]}" + (f" typed '{r['typed']}'" if r["typed"] else "")]
        out[split].append({"images": [after], "state": state_text(r["goal"], hist_after), "questions": [{"qid": "done", "qtype": "noul", "instructions": rng.choice(DONE_INSTR)}], "targets": {"done": L["done_after"]}, "meta": {**meta, "which": "after"}}); n[f"{split}/done{L['done_after']}"] += 1
        # skip: balanced sample of candidates
        pos = [k for k, v in L["skip"].items() if v]; neg = [k for k, v in L["skip"].items() if not v]
        picks = rng.sample(pos, min(a.skip_per_step, len(pos))) + rng.sample(neg, min(a.skip_per_step, len(neg)))
        for k in picks:
            out[split].append({"images": [before], "state": st + f"Candidate {k}: {r['candidates'][k]}\n", "questions": [{"qid": "skip", "qtype": "noul", "instructions": rng.choice(SKIP_INSTR)}], "targets": {"skip": L["skip"][k]}, "meta": {**meta, "candidate": k}}); n[f"{split}/skip{L['skip'][k]}"] += 1
        # ground: goal target known and task not yet done
        if r["target_idx"] and not L["done_before"] and rng.random() < 0.5:
            crit = dict(r["candidates"]); crit["none"] = "none of the marked elements is the right target"
            out[split].append({"images": [before], "state": st, "questions": [{"qid": "ground", "qtype": "choice", "instructions": GROUND_INSTR, "criteria": crit}], "targets": {"ground": r["target_idx"]}, "meta": meta}); n[f"{split}/ground"] += 1
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    for split, rows in out.items():
        rng.shuffle(rows)
        with open(od / f"{a.tag}.{split}.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("sites", len(sites), "val sites", len(val_sites)); print(json.dumps(dict(sorted(n.items())), indent=1))


if __name__ == "__main__":
    main()
