"""Convert benchmark items (either track) back into the training/eval row format so model/eval_schema.py can score a
local checkpoint on them, and write the resulting per-item predictions as a bench submission.

python bench/items_to_rows.py --items bench/v1_items_pixel.jsonl --release release/v1 --out /tmp/v1_pixel.rows.jsonl
python model/eval_schema.py model/runs/v5b-8b-state/final --data /tmp/v1_pixel.rows.jsonl --temp 0.5 --out /tmp/v1_pixel.eval.json
python bench/items_to_rows.py --to-submission /tmp/v1_pixel.eval.json --items bench/v1_items_pixel.jsonl --out bench/baselines/v5b_pixel.jsonl
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def to_rows(items_path, release, out):
    n = 0
    with open(out, "w") as f:
        for l in open(items_path):
            it = json.loads(l)
            q = {"qid": it["question"], "qtype": it["type"], "instructions": it["instructions"]}
            if it["type"] == "choice":
                q["criteria"] = it["criteria"]
            row = {"images": [str((Path(release) / p).resolve()) for p in it["images"]], "state": it["state"],
                   "questions": [q], "targets": {it["question"]: it["label"]},
                   "meta": {"id": it["id"], "site": it["site"], "step_id": it["step_id"]}}
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); n += 1
    print(f"{n} rows -> {out}")


def to_submission(eval_json, items_path, out):
    d = json.load(open(eval_json)); items = d.get("items") or []
    ids = [json.loads(l)["id"] for l in open(items_path)]
    n = 0
    with open(out, "w") as f:
        for r in items:
            iid = (r.get("meta") or {}).get("id") or (ids[r["i"]] if "i" in r and r["i"] < len(ids) else None)
            if not iid:
                continue
            rec = {"id": iid, "p": r["p"]}
            if "choice" in r:
                rec["choice"] = r["choice"]
            f.write(json.dumps(rec) + "\n"); n += 1
    print(f"{n} predictions -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True); ap.add_argument("--release", default=None)
    ap.add_argument("--out", required=True); ap.add_argument("--to-submission", default=None, help="eval_schema output json")
    a = ap.parse_args()
    if a.to_submission:
        to_submission(a.to_submission, a.items, a.out)
    else:
        to_rows(a.items, a.release or ROOT / "release" / Path(a.items).stem.split("_")[0], a.out)


if __name__ == "__main__":
    main()
