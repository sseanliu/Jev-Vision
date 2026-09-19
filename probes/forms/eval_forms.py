"""Same-decision comparison on Cua's synthetic form-filling test split (cua_s1.synth, 10,000 episodes, seed 2026).

Each row: a context (task / form / element) and options (document entities to fill, plus check/click/skip).
Providers score the same rows: cua-s1-forms (their 706K byte-level scorer), our decision server (text-only,
never trained on forms), and TypeSafe Jev. Metrics follow the model card split: judgment calls (gold fill/check/click)
vs no-ops (gold skip).

python eval_forms.py sample --n 1000
python eval_forms.py s1 --ckpt ../../model/data/cua_s1_forms/cua-s1-forms.safetensors   # run with the cua-s1 venv
python eval_forms.py ours --server http://127.0.0.1:8811
python eval_forms.py jev
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "model/data/cua_s1_synth"
SUBSET = DATA / "test_1k.jsonl"
OUT = ROOT / "results/forms"
INSTR = "Which option is the right action for this form element?"


def load_rows():
    return [json.loads(l) for l in open(SUBSET)]


def report(name, preds, extra=None):
    """preds: list of (row, chosen_index, top_prob, ms)."""
    by = defaultdict(list)
    rows = []
    for row, idx, p, ms in preds:
        gold = row["label"]; hit = idx == gold; act = row["meta"]["action"]
        by["all"].append(hit); by[act].append(hit); by["noop" if act == "skip" else "judgment"].append(hit)
        rows.append({"conf": p, "hit": hit, "ms": ms})
    n = len(rows)
    bins = [[] for _ in range(10)]
    for r in rows:
        bins[min(9, int(r["conf"] * 10))].append(r)
    ece = sum(len(b) / n * abs(sum(x["hit"] for x in b) / len(b) - sum(x["conf"] for x in b) / len(b)) for b in bins if b)
    pos = [r["conf"] for r in rows if r["hit"]]; neg = [r["conf"] for r in rows if not r["hit"]]
    auroc = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg)) if pos and neg else float("nan")
    ms = sorted(r["ms"] for r in rows)
    rep = {"provider": name, "n": n, "acc": round(sum(by["all"]) / n, 4),
           "by_gold": {k: {"n": len(v), "acc": round(sum(v) / len(v), 4)} for k, v in by.items() if k != "all"},
           "ece": round(ece, 3), "auroc": round(auroc, 3), "p50_ms": round(ms[n // 2], 1), "mean_ms": round(sum(ms) / n, 1)}
    if extra:
        rep.update(extra)
    print(json.dumps(rep, indent=1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps({"report": rep, "preds": [
        {"i": i, "gold": row["label"], "pred": idx, "p": p, "action": row["meta"]["action"]} for i, (row, idx, p, ms) in enumerate(preds)]}, indent=1))
    return rep


def cmd_sample(a):
    rows = [json.loads(l) for l in open(DATA / "test.jsonl")]
    rng = random.Random(a.seed); sub = rng.sample(rows, a.n)
    with open(SUBSET, "w") as f:
        for r in sub:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    acts = defaultdict(int)
    for r in sub:
        acts[r["meta"]["action"]] += 1
    print("wrote", len(sub), "rows", dict(acts), "mean options", round(sum(len(r["options"]) for r in sub) / len(sub), 1))


def cmd_s1(a):
    import torch
    from cua_s1.model import load_checkpoint, validate_example
    model, collator, cfg = load_checkpoint(a.ckpt, "cpu")
    rows = load_rows(); preds = []
    with torch.no_grad():
        for i in range(0, len(rows), 64):
            chunk = rows[i:i + 64]
            ex = [validate_example(r) for r in chunk]
            t0 = time.perf_counter(); logits = model(collator(ex)); ms = (time.perf_counter() - t0) * 1000 / len(chunk)
            for r, z in zip(chunk, logits):
                z = z[: len(r["options"])]; p = torch.softmax(z.float(), -1)
                idx = int(p.argmax()); preds.append((r, idx, float(p[idx]), ms))
    report("cua-s1-forms", preds, {"params": sum(x.numel() for x in model.parameters()), "config": cfg})


def _ours_one(server, r):
    import urllib.request
    crit = {str(i): o for i, o in enumerate(r["options"])}
    payload = {"model": "s1", "state": r["context"], "image": None,
               "questions": {"act": {"type": "choice", "instructions": INSTR, "criteria": crit}}}
    req = urllib.request.Request(server.rstrip("/") + "/v1/systemone", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); res = json.loads(urllib.request.urlopen(req, timeout=120).read()); ms = (time.perf_counter() - t0) * 1000
    ans = res["answers"]["act"]; return r, int(ans["choice"]), float(ans["probabilities"][ans["choice"]]), ms


def cmd_ours(a):
    rows = load_rows(); preds = []
    with ThreadPoolExecutor(a.workers) as ex:
        for k, f in enumerate(as_completed([ex.submit(_ours_one, a.server, r) for r in rows]), 1):
            preds.append(f.result())
            if k % 200 == 0:
                print(f"  {k}/{len(rows)}", flush=True)
    report(a.name, preds)


def cmd_jev(a):
    if not os.environ.get("TYPESAFE_API_KEY"):
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith("TYPESAFE_API_KEY="):
                os.environ["TYPESAFE_API_KEY"] = line.split("=", 1)[1].strip()
    from typesafe_sdk import Choice, TypeSafeClient
    client = TypeSafeClient(timeout=60.0); rows = load_rows(); preds = []; errors = 0

    def one(r):
        crit = {str(i): o for i, o in enumerate(r["options"])}
        t0 = time.perf_counter()
        res = client.system_one(state=r["context"], questions={"act": Choice(instructions=INSTR, criteria=crit)})
        ms = (time.perf_counter() - t0) * 1000; ans = res.answers["act"]
        return r, int(ans.choice), float(ans.probabilities[ans.choice]), ms

    with ThreadPoolExecutor(a.workers) as ex:
        for k, f in enumerate(as_completed([ex.submit(one, r) for r in rows]), 1):
            try:
                preds.append(f.result())
            except Exception as e:
                errors += 1; print("  error", type(e).__name__, str(e)[:100], flush=True)
            if k % 200 == 0:
                print(f"  {k}/{len(rows)}", flush=True)
    report("jev", preds, {"errors": errors})


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample"); s.add_argument("--n", type=int, default=1000); s.add_argument("--seed", type=int, default=0); s.set_defaults(f=cmd_sample)
    s = sub.add_parser("s1"); s.add_argument("--ckpt", required=True); s.set_defaults(f=cmd_s1)
    s = sub.add_parser("ours"); s.add_argument("--server", default="http://127.0.0.1:8811"); s.add_argument("--workers", type=int, default=4); s.add_argument("--name", default="ours-v3"); s.set_defaults(f=cmd_ours)
    s = sub.add_parser("jev"); s.add_argument("--workers", type=int, default=8); s.set_defaults(f=cmd_jev)
    a = ap.parse_args(); a.f(a)


if __name__ == "__main__":
    main()
