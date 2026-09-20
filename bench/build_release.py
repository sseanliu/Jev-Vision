"""Assemble the step-verifier benchmark release from the held-out state rows.

Input : model/data/vision/state_rows/<split>.jsonl (rows with images, state, questions, targets, meta)
Output: bench/<name>_items.jsonl        (tracked; one item per question, image paths relative to the release dir)
        release/<name>/images/*.png     (gitignored; copied screenshots, deduplicated)
        release/<name>/items.jsonl      (same as the tracked file, for the HF upload)
        bench/baselines/<judge>.jsonl   (predictions converted from results/, keyed by item id)

python bench/build_release.py --split rec1b.validation --name v0
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def item_id(meta, qid):
    return f"{meta['step_id']}:{qid}:{meta.get('candidate', '')}:{meta.get('which', '')}"


def build_items(split):
    rows = [json.loads(l) for l in open(ROOT / "model/data/vision/state_rows" / f"{split}.jsonl")]
    items, seen = [], Counter()
    for r in rows:
        q = r["questions"][0]; m = r["meta"]; qid = q["qid"]
        iid = item_id(m, qid); seen[iid] += 1
        label = r["targets"][qid]
        it = {
            "id": iid, "track": "candidate", "question": qid, "type": q["qtype"],
            "instructions": q["instructions"], "state": r["state"],
            "images": [f"images/{Path(p).name}" for p in r["images"]],
            "label": label, "site": m["site"], "goal_kind": m.get("goal_kind"), "step_id": m["step_id"],
            "url_before": m.get("url_before"), "url_after": m.get("url_after"),
            "elements_before": m.get("text_before", ""), "elements_after": m.get("text_after", ""),
            "_src_images": r["images"],
        }
        if q["qtype"] == "choice":
            it["criteria"] = q["criteria"]
        if "candidate" in m:
            it["candidate"] = m["candidate"]
        if "which" in m:
            it["which"] = m["which"]
        items.append(it)
    dups = {k: v for k, v in seen.items() if v > 1}
    assert not dups, f"duplicate item ids: {list(dups)[:5]}"
    return items


def convert_baselines(items):
    """results/state/judge_*.json rows carry meta but no ids; results/vision/v5*/eval_state.json items carry meta too."""
    by_key = {(it["step_id"], it["question"], str(it.get("candidate", "")), str(it.get("which", ""))): it["id"] for it in items}

    def key(meta, qid):
        return (meta["step_id"], qid, str(meta.get("candidate", "")), str(meta.get("which", "")))

    out = {}
    for name in ["jev", "claude"]:
        f = ROOT / "results/state" / f"judge_{name}_rec1b.validation.json"
        if not f.exists():
            continue
        d = json.load(open(f)); preds = []
        for qid, rows in d["rows"].items():
            for r in rows:
                iid = by_key.get(key(r["meta"], qid))
                if iid:
                    preds.append({"id": iid, "p": r["p"]})
        out[name] = preds
    for name, tag in [("v5", "v5-8b-state"), ("v5b", "v5b-8b-state")]:
        f = ROOT / "results/vision" / tag / "eval_state.json"
        if not f.exists():
            continue
        d = json.load(open(f))
        if "items" not in d:  # V5 kept per-item predictions in a separate file
            d = json.load(open(ROOT / "results/vision" / tag / "eval_state_items.json"))
        rows_ = d["items"] if isinstance(d, dict) else d
        preds = []
        for r in rows_:
            iid = by_key.get(key(r["meta"], r["qid"]))
            if iid:
                preds.append({"id": iid, "p": r["p"]} if r["qid"] != "ground" else {"id": iid, "choice": r.get("choice"), "p": r["p"]})
        out[name] = preds
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="rec1b.validation"); ap.add_argument("--name", default="v0")
    a = ap.parse_args()
    items = build_items(a.split)
    rel = ROOT / "release" / a.name; (rel / "images").mkdir(parents=True, exist_ok=True)
    copied = 0
    for it in items:
        for src, dst in zip(it.pop("_src_images"), it["images"]):
            d = rel / dst
            if not d.exists():
                shutil.copy2(src, d); copied += 1
    with open(rel / "items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    shutil.copy2(rel / "items.jsonl", ROOT / "bench" / f"{a.name}_items.jsonl")
    (ROOT / "bench/baselines").mkdir(exist_ok=True)
    for name, preds in convert_baselines(items).items():
        with open(ROOT / "bench/baselines" / f"{name}.jsonl", "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
        print(f"baseline {name}: {len(preds)} predictions")
    stats = Counter(it["question"] for it in items)
    print(f"{len(items)} items, {len({it['site'] for it in items})} sites, {copied} images copied, {dict(stats)}")


if __name__ == "__main__":
    main()
