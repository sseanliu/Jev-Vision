"""TypeSafe Jev on the V0b ground items, same candidates and state text as our model gets.

Jev's contract carries no screenshot, so this is the text-only view of the same decision:
task + history + the numbered candidates' DOM descriptions. One Choice question per item.

python teacher_jev.py --items ../../model/data/vision/m2w_items_j --limit 300
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from teacher_claude import metrics

ROOT = Path(__file__).resolve().parents[2]


def load_key():
    if os.environ.get("TYPESAFE_API_KEY"):
        return
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("TYPESAFE_API_KEY="):
            os.environ["TYPESAFE_API_KEY"] = line.split("=", 1)[1].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="../../model/data/vision/m2w_items_j")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    load_key()
    from typesafe_sdk import Choice, TypeSafeClient

    reqs = [json.loads(l) for l in open(Path(a.items) / "requests.jsonl")][: a.limit]
    client = TypeSafeClient(model=a.model, timeout=60.0)
    lock = threading.Lock(); per_item = {}; usage = {"in": 0, "out": 0}; lat = []

    def work(r):
        q = r["questions"][0]
        t0 = time.perf_counter()
        res = client.system_one(state=r["state"], questions={"ground": Choice(instructions=q["instructions"], criteria=q["criteria"])})
        ms = (time.perf_counter() - t0) * 1000
        ans = res.choices["ground"]
        with lock:
            usage["in"] += res.usage.input_tokens; usage["out"] += res.usage.output_tokens; lat.append(ms)
        return r["id"], {"gold": r["targets"]["ground"], "verbal": dict(ans.probabilities), "choice": ans.choice,
                         "confidence": float(ans.confidence), "model": res.model, "ms": round(ms, 1)}

    t0 = time.time(); errors = 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(work, r) for r in reqs]
        for k, f in enumerate(as_completed(futs), 1):
            try:
                iid, res = f.result(); per_item[iid] = res
            except Exception as e:
                errors += 1; print(f"  error {type(e).__name__}: {str(e)[:120]}", flush=True)
            if k % 50 == 0:
                print(f"  {k}/{len(reqs)} {time.time()-t0:.0f}s", flush=True)
    rows = []
    for r in per_item.values():
        p = r["verbal"]; top = max(p, key=p.get); rows.append({"conf": p[top], "hit": top == r["gold"]})
    rep = metrics(rows); lat.sort()
    rep.update({"model": next(iter(per_item.values()))["model"], "errors": errors, "usage": usage,
                "mean_ms": round(sum(lat) / len(lat), 1), "p50_ms": round(lat[len(lat) // 2], 1)})
    print("jev", rep)
    out = Path(a.out or f"../../results/vision/teacher_jev_m2w{len(reqs)}_j.json")
    out.write_text(json.dumps({"report": rep, "items": per_item}, indent=1)); print("saved", out)


if __name__ == "__main__":
    main()
