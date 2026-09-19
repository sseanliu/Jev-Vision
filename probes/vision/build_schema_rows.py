"""Schema-mix training rows: teach the student to fill in *any* caller-supplied schema, not
three fixed questions. Each screenshot gets 1-3 questions sampled from a template pool with
randomised phrasing, option subsets and option order, all labelled from Mind2Web fields:

  ground   choice   which marked element to act on (+ none)            gold element / none
  act      choice   what to do next, options a random subset of the     dataset op (click/type/select)
                    action vocabulary that always contains the answer
  final    noul     is this the last action the task needs              step_index == n_steps-1
  needs_text noul   does this step require typing text                  op == TYPE
  progress score    how far along the task is                           step_index / n_steps bucket
  tag      choice   what kind of element is the target                  target_tag (link/button/...)
  history_len score how many actions were taken before this screen      len(history)

Rows keep the ground question most of the time so the hard grounding signal is not diluted.
python build_schema_rows.py --items /workspace/m2w_train_j --out /workspace/m2w_train_j/schema.train.jsonl --per-item 2
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

GROUND_INSTR = [
    "Which numbered element should be acted on next to make progress on the task?",
    "Pick the marked element the next action should target.",
    "Which of the boxed elements is the right target for the next step?",
    "Select the element to interact with next.",
]
ACT_INSTR = [
    "What kind of action should be taken next?",
    "Which action moves the task forward from this screen?",
    "What should the agent do next?",
]
ACT_VOCAB = {
    "click": "click on an element", "type": "type text into a field", "select": "choose an option from a dropdown",
    "scroll": "scroll the page", "go_back": "go back to the previous page", "done": "stop, the task is complete",
    "ask_user": "ask the user for clarification", "none": "none of these",
}
OP_TO_ACT = {"CLICK": "click", "TYPE": "type", "SELECT": "select"}
FINAL_INSTR = ["After this action, will the task be complete?", "Is this the last step needed to finish the task?",
               "Does the task end with this action?"]
NEEDS_TEXT_INSTR = ["Does this step require typing text?", "Will the agent need to enter text here?"]
PROGRESS_INSTR = ["How far along is the task?", "Estimate the task's progress."]
PROGRESS_LEVELS = ["just started, nothing done yet", "early, a few steps done", "about halfway", "almost finished", "final step"]
TAG_INSTR = ["What kind of element is the target of the next action?", "Which element type is being acted on?"]
TAG_VOCAB = {"link": "a link", "button": "a button", "combobox": "a dropdown or combobox", "textbox": "a text input",
             "label": "a label or option", "checkbox": "a checkbox", "radio": "a radio button", "tab": "a tab",
             "input": "an input field", "select": "a select menu", "menuitem": "a menu item", "other": "something else"}
HIST_INSTR = ["How many actions have been taken before this screen?"]
HIST_LEVELS = ["none", "one", "two", "three or four", "five or more"]


def q(qid, qtype, instr, crit=None):
    d = {"qid": qid, "qtype": qtype, "instructions": instr}
    if crit is not None:
        d["criteria"] = crit
    return d


def make_questions(it: dict, rng: random.Random):
    qs, targets = [], {}
    # ground is always present (keys stay numeric because they are drawn on the image)
    qs.append(q("ground", "choice", rng.choice(GROUND_INSTR), it["criteria"])); targets["ground"] = it["gold"]
    pool = []
    act = OP_TO_ACT.get(it.get("op"))
    if act:
        def make_act():
            extras = rng.sample([k for k in ACT_VOCAB if k != act], rng.randint(1, 5))
            keys = extras + [act]; rng.shuffle(keys)
            return q("act", "choice", rng.choice(ACT_INSTR), {k: ACT_VOCAB[k] for k in keys}), act
        pool.append(make_act)
    if it.get("step_index") is not None and it.get("n_steps"):
        last = it["step_index"] == it["n_steps"] - 1
        pool.append(lambda: (q("final", "noul", rng.choice(FINAL_INSTR)), last))
        frac = (it["step_index"] + 1) / it["n_steps"]
        lvl = 4 if last else min(3, int(frac * 4))
        pool.append(lambda: (q("progress", "score", rng.choice(PROGRESS_INSTR), PROGRESS_LEVELS), lvl))
    pool.append(lambda: (q("needs_text", "noul", rng.choice(NEEDS_TEXT_INSTR)), it.get("op") == "TYPE"))
    m = re.match(r"\[(\w+)\]", it.get("target_repr") or "")
    tag = (it.get("target_tag") or (m.group(1) if m else "")).lower()
    if tag:
        def make_tag():
            gold = tag if tag in TAG_VOCAB else "other"
            keys = rng.sample([k for k in TAG_VOCAB if k != gold], rng.randint(2, 5)) + [gold]; rng.shuffle(keys)
            return q("tag", "choice", rng.choice(TAG_INSTR), {k: TAG_VOCAB[k] for k in keys}), gold
        pool.append(make_tag)
    h = len(it.get("history", []))
    pool.append(lambda: (q("history_len", "score", rng.choice(HIST_INSTR), HIST_LEVELS), 0 if h == 0 else 1 if h == 1 else 2 if h == 2 else 3 if h <= 4 else 4))
    for maker in rng.sample(pool, min(len(pool), rng.randint(1, 2))):
        qq, t = maker()
        qs.append(qq); targets[qq["qid"]] = t
    rng.shuffle(qs)
    return qs, targets


def state_text(task: str, history: list[str]) -> str:
    hist = "\n".join(f"  - {h}" for h in history) or "  (none)"
    return f"Task: {task}\nActions already taken:\n{hist}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True, help="dir with items.jsonl (+ img/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-item", type=int, default=2, help="rows per screenshot (different question mixes)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    root = Path(a.items).resolve()
    items = [json.loads(l) for l in open(root / "items.jsonl")]
    n = 0
    with open(a.out, "w") as f:
        for it in items:
            for k in range(a.per_item):
                qs, targets = make_questions(it, rng)
                f.write(json.dumps({"id": f"{it['id']}#{k}", "image": str(root / it["image"]), "state": state_text(it["task"], it["history"]),
                                    "questions": qs, "targets": targets}, ensure_ascii=False) + "\n"); n += 1
    print(f"wrote {n} schema rows from {len(items)} items to {a.out}")


if __name__ == "__main__":
    main()
