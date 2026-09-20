"""Convert a probes/vision/judge_baselines.py output (rows with meta, no ids) into a bench submission keyed by item id.

python bench/convert_judge.py --judge results/state/judge_jev_rec2.validation.json --items bench/v1_items.jsonl --out bench/baselines/v1_jev.jsonl
"""

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True); ap.add_argument("--items", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    by_key = {}
    for l in open(a.items):
        it = json.loads(l)
        by_key[(it["step_id"], it["question"], str(it.get("candidate", "")), str(it.get("which", "")))] = it["id"]
    d = json.load(open(a.judge)); n = 0; miss = 0
    with open(a.out, "w") as f:
        for qid, rows in d["rows"].items():
            for r in rows:
                m = r["meta"]; iid = by_key.get((m["step_id"], qid, str(m.get("candidate", "")), str(m.get("which", ""))))
                if not iid:
                    miss += 1; continue
                f.write(json.dumps({"id": iid, "p": r["p"]}) + "\n"); n += 1
    print(f"{n} predictions -> {a.out} ({miss} unmatched)")


if __name__ == "__main__":
    main()
