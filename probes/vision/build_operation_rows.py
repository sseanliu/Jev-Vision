"""Recorder steps -> training rows for jev-ultrafast's `operation` and `*_target` questions, in the exact form the
harness sends them at run time (so the rules text, the option keys and the DONE description are seen in training).

Question format is copied from jev_ultrafast/model.py `choose()`:
  operation:   choice over {CLICK, TYPE_TEXT, DONE, BLOCKED}; instructions = {"goal", "rules": NEXT_ACTION} as JSON
  click_target / type_text_target: choice over element indices; instructions = {"goal", "operation", "rules": [NEXT_ACTION, TARGET]}
State is the harness's JSON state (page url, elements, recent_actions) rendered like serve.py does. Images are the raw
screenshots (no marks), which is what S1_SCREENSHOT=1 sends.

Gold comes from the TASK, not from the recorded policy: 57% of recorded steps are deliberately wrong actions (random /
unrelated / repeat policies, there to label effect and skip negatives). DONE on the before image when
labels.done_before and on the after image when labels.done_after; otherwise the operation the oracle would take at
that state (TYPE_TEXT into the goal field while it lacks the goal value, else CLICK). Target rows use the goal's
target element (target_idx) and are skipped when it is unknown.

python build_operation_rows.py --steps ../../model/data/vision/triplets/run2_train/steps.jsonl --out ../../model/data/vision/state_rows/rec2train_ops.train.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

import importlib.util
_spec = importlib.util.spec_from_file_location("jev_questions", Path.home() / "Github/jev-ultrafast/jev_ultrafast/questions.py")
_q = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_q)  # load the file alone (the package imports the browser)
NEXT_ACTION, TARGET = _q.NEXT_ACTION, _q.TARGET  # verbatim rules text

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "harness"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_state_rows_record import relabel  # same strict done/skip relabel as the state rows

OP_LABELS = {
    "CLICK": "Click an element, button, menu option, autocomplete suggestion, or calendar day.",
    "TYPE_TEXT": "Enter or replace text in an editable field. A small LLM will supply the value from the goal.",
}
TERMINAL = {"DONE": "Every requirement is visibly satisfied.", "BLOCKED": "No supported operation can progress."}
CAND_RE = re.compile(r"^element (\d+): (\w+)(?: (\w+))? '(.*?)'(?: value='(.*?)')?$")
TYPEABLE = {"input", "textarea", "searchbox", "combobox"}


def parse_candidate(s: str):
    m = CAND_RE.match(s.strip())
    if not m:
        return {"label": s.strip(), "typeable": False}
    idx, tag, typ, text, value = m.groups()
    typeable = tag in TYPEABLE and (typ or "") not in {"submit", "button", "checkbox", "radio"}
    e = {"label": text, "typeable": typeable}
    if value is not None:
        e["value"] = value
    return e


def as_text(v):
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def make_state(url, candidates, history):
    elements = []
    for k, s in candidates.items():
        e = parse_candidate(s)
        ops = ["CLICK"] + (["TYPE_TEXT"] if e["typeable"] else [])
        el = {"index": str(k), "label": e["label"], "operations": ops}
        if "value" in e:
            el["value"] = e["value"]
        elements.append(el)
    recent = []
    for h in history[-10:]:
        m = re.match(r"^(\w+) on (.*?)(?: typed '(.*)')?$", h)
        recent.append({"action": m.group(2) if m else h, "kind": m.group(1) if m else None, "text": m.group(3) if m else None, "page_changed": None})
    st = {"page": {"url": url, "title": "", "text": ""}, "elements": elements, "recent_actions": recent}
    return json.dumps(st, indent=1, ensure_ascii=False), any(e["typeable"] for e in map(parse_candidate, candidates.values()))


def op_question(goal, has_typeable):
    crit = {"CLICK": OP_LABELS["CLICK"]}
    if has_typeable:
        crit["TYPE_TEXT"] = OP_LABELS["TYPE_TEXT"]
    crit.update(TERMINAL)
    return {"qid": "operation", "qtype": "choice", "instructions": as_text({"goal": goal, "rules": NEXT_ACTION}), "criteria": crit}


def target_question(goal, operation, candidates, only_typeable):
    crit = {}
    for k, s in candidates.items():
        e = parse_candidate(s)
        if only_typeable and not e["typeable"]:
            continue
        crit[str(k)] = as_text({"element": f"[{k}] {e['label']}", "current_value": e.get("value", "")})
    return {"qid": operation.lower() + "_target", "qtype": "choice",
            "instructions": as_text({"goal": goal, "operation": operation, "rules": [NEXT_ACTION, TARGET]}), "criteria": crit}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--target-frac", type=float, default=0.5)
    ap.add_argument("--after-done-frac", type=float, default=0.6, help="keep this fraction of after-image DONE rows")
    a = ap.parse_args(); rng = random.Random(a.seed)
    steps_path = Path(a.steps).resolve(); root = steps_path.parent
    recs = [json.loads(l) for l in open(steps_path)]; recs = [r for r in recs if "error" not in r and r.get("raw_before_img")]
    recs = relabel(recs)
    rows = []; n = collections.Counter()
    for r in recs:
        L = r["labels"]; goal = r["goal"]
        before = str(root / r["raw_before_img"]); after = str(root / r["raw_after_img"])
        meta = {"site": r["site"], "step_id": r["id"], "url_before": r["before_url"], "url_after": r["after_url"], "goal_kind": r["goal_kind"]}
        st, typeable = make_state(r["before_url"], r["candidates"], r["history"])
        tidx = r.get("target_idx")
        tcand = r["candidates"].get(str(tidx), "") if tidx else ""
        if L["done_before"]:
            gold = "DONE"
        elif r["goal_kind"] == "search" and tidx:
            e = parse_candidate(tcand); gv = (r.get("goal_value") or "").strip().lower()
            gold = "CLICK" if (e.get("value", "").strip().lower() == gv and gv) else "TYPE_TEXT"
        elif r["policy"] == "oracle":
            gold = "TYPE_TEXT" if r["action"] == "type" else "CLICK"
        else:
            gold = "CLICK"
        q = op_question(goal, typeable or gold == "TYPE_TEXT")
        if gold == "TYPE_TEXT" and "TYPE_TEXT" not in q["criteria"]:
            q["criteria"] = {"CLICK": OP_LABELS["CLICK"], "TYPE_TEXT": OP_LABELS["TYPE_TEXT"], **TERMINAL}
        rows.append({"images": [before], "state": st, "questions": [q], "targets": {"operation": gold}, "meta": {**meta, "which": "before", "policy": r["policy"]}}); n[f"op/{gold}"] += 1
        # target question on the same before state: the goal's target element, whatever the recorded policy did
        if gold != "DONE" and tidx and str(tidx) in r["candidates"] and rng.random() < a.target_frac:
            tq = target_question(goal, gold, r["candidates"], only_typeable=(gold == "TYPE_TEXT"))
            if str(tidx) in tq["criteria"] and len(tq["criteria"]) >= 2:
                rows.append({"images": [before], "state": st, "questions": [tq], "targets": {tq["qid"]: str(tidx)}, "meta": {**meta, "which": "before", "policy": r["policy"]}}); n[f"target/{gold}"] += 1
        # after image: terminal state -> DONE
        if L["done_after"] and r.get("candidates_after") and rng.random() < a.after_done_frac:
            hist_after = r["history"] + [f"{r['action']} on {r['candidates'].get(r['chosen'], '?')[:40]}" + (f" typed '{r['typed']}'" if r["typed"] else "")]
            st2, typeable2 = make_state(r["after_url"], r["candidates_after"], hist_after)
            rows.append({"images": [after], "state": st2, "questions": [op_question(goal, typeable2)], "targets": {"operation": "DONE"}, "meta": {**meta, "which": "after"}}); n["op/DONE_after"] += 1
    rng.shuffle(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(len(rows), "rows ->", a.out); print(dict(n))


if __name__ == "__main__":
    main()
