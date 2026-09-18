"""Collect Jev's probability distributions on our training requests as soft
labels (data condition (b): Jev-distilled).

python distill_jev.py --in ../model/data/jsonl --out ../model/data/jsonl_jev \
    --per-source 3000 --workers 6

Each output row keeps the original hard target under "targets_hard" and puts
Jev's distribution under "targets": choice/score -> {key: p}, noul -> p(yes).
Rows Jev rejects are skipped and counted. Resumable: existing output rows are
kept and their states are not re-queried.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common import call

lock = threading.Lock()


def to_payload(row: dict) -> dict:
    questions = {}
    for q in row["questions"]:
        entry = {"type": q["qtype"], "instructions": q["instructions"]}
        if q["qtype"] in ("choice", "score"):
            entry["criteria"] = q["criteria"]
        questions[q["qid"]] = entry
    return {"model": "jev-latest", "state": row["state"], "questions": questions}


def distill_row(row: dict) -> dict | None:
    res = call(to_payload(row))
    if res["status"] != 200:
        return None
    answers = res["body"]["answers"]
    soft = {}
    for q in row["questions"]:
        a = answers[q["qid"]]
        if q["qtype"] == "noul":
            soft[q["qid"]] = a["noul"]
        else:
            soft[q["qid"]] = a["probabilities"]
    return {**row, "targets_hard": row.get("targets"), "targets": soft,
            "jev_model": res["body"]["model"], "jev_request_id": res.get("request_id")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="../model/data/jsonl")
    ap.add_argument("--out", default="../model/data/jsonl_jev")
    ap.add_argument("--per-source", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    inp, out = Path(a.inp), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    total_ok = total_fail = 0
    t0 = time.time()
    for path in sorted(inp.glob("*.train.jsonl")):
        rows = [json.loads(l) for l in open(path)]
        rng.shuffle(rows)
        rows = rows[: a.per_source]
        dest = out / path.name
        done_states = set()
        if dest.exists():
            done_states = {json.loads(l)["state"] for l in open(dest)}
        todo = [r for r in rows if r["state"] not in done_states]
        ok = fail = 0
        with dest.open("a") as f, ThreadPoolExecutor(a.workers) as ex:
            futs = {ex.submit(distill_row, r): r for r in todo}
            for fut in as_completed(futs):
                r = fut.result()
                if r is None:
                    fail += 1
                    continue
                with lock:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                ok += 1
                if (ok + fail) % 500 == 0:
                    print(f"  {path.name}: {ok} ok, {fail} fail, {time.time()-t0:.0f}s", flush=True)
        total_ok += ok
        total_fail += fail
        print(f"{path.name:32s} kept={len(done_states)} new_ok={ok} fail={fail}", flush=True)
    print(f"done: {total_ok} ok, {total_fail} fail, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
