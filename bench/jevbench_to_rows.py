"""JevBench public items -> our eval row format (text-only), so eval_schema.py and baseline_frozen_vl.py can score them.
python bench/jevbench_to_rows.py --dir <dir with easy/original/hard.jsonl> --out /workspace/bench/jevbench_public.rows.jsonl"""
import argparse, json
ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
n = 0
with open(a.out, "w") as f:
    for tier in ["easy", "original", "hard"]:
        for l in open(f"{a.dir}/{tier}.jsonl"):
            if not l.strip(): continue
            r = json.loads(l); q = dict(r["question"]); exp = str(r["expected"])
            state = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], indent=1, ensure_ascii=False)  # as serve.py renders it
            qq = {"qid": f"jb_{tier}", "qtype": q["type"], "instructions": q["instructions"]}
            if q["type"] == "noul":
                target = int(exp == "yes")
            elif q["type"] == "choice":
                qq["criteria"] = q["criteria"]; target = exp
            else:
                qq["criteria"] = q["criteria"]; target = int(exp)
            f.write(json.dumps({"images": [], "state": state, "questions": [qq], "targets": {qq["qid"]: target}, "meta": {"id": r["id"], "tier": tier, "family": r.get("family")}}, ensure_ascii=False) + "\n"); n += 1
print(n, "rows ->", a.out)
