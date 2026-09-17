"""Is the billing token counter the model's tokenizer?

Server time grows linearly with state length (Hume, latency-rerun). If the
counter is the model's tokenizer, server time per *billed* token should be the
same for every text family. If the counter over-counts some text relative to
what the model actually sees, those families will show a shallower slope per
billed token, and their slopes should line up when plotted against a
frontier-style tokenizer count instead.

Families
- plain:     ordinary English prose (Jev count ~ Qwen count)
- latin_x2:  long English words Jev splits ~2x finer than Qwen
- cyrillic:  Cyrillic words Jev splits per character (~3x Qwen)
- cjk:       Chinese Jev splits per character (~1.5-2x Qwen)

Each family is scaled to the same set of target billed sizes using the API's
own counter, then requested REPEATS times in shuffled order, one at a time.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from datetime import datetime, timezone

from common import ROOT, call, save

QUESTION = {"q": {"type": "noul", "instructions": "Is the text mostly about payments?"}}
FAMILIES = {
    "plain": (
        "The customer wrote to say that the order arrived late and the box was "
        "slightly damaged, but the item inside was fine and they would like to keep "
        "it. They asked whether a partial refund for the shipping cost was possible. "
    ),
    "latin_x2": (
        "responsibility unbelievable tokenization probability understanding "
        "JavaScript TypeScript Kubernetes accountability sustainability "
    ),
    "cyrillic": "Привет спасибо пожалуйста хорошо сегодня завтра работа ",
    "cjk": "人工智能 机器学习 自然语言处理 深度学习 神经网络 概率分布 ",
}
TARGETS = [4000, 12000, 20000, 28000]
REPEATS = 8
SEED = 20260917


def billed(state: str) -> int:
    r = call({"model": "jev-latest", "state": state, "questions": QUESTION})
    return r["body"]["usage"]["input_tokens"]


def qwen_counter():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    return lambda s: len(tok.encode(s, add_special_tokens=False))


def main() -> None:
    qwen = qwen_counter()
    base = billed("")
    print("empty-state billed tokens:", base)
    # Calibrate tokens per unit for each family with the API's own counter.
    units = {}
    for fam, unit in FAMILIES.items():
        per = (billed(unit * 50) - base) / 50
        units[fam] = per
        print(f"{fam:10s} billed/unit={per:.2f} qwen/unit={qwen(unit * 50) / 50:.2f}")

    plan = []
    for fam, unit in FAMILIES.items():
        for target in TARGETS:
            n = max(1, round(target / units[fam]))
            state = unit * n
            plan.append({"family": fam, "target": target, "state": state,
                         "qwen_tokens": qwen(state)})
    rng = random.Random(SEED)
    trials = []
    order = [(p, r) for p in plan for r in range(REPEATS)]
    rng.shuffle(order)
    for i, (p, rep) in enumerate(order, 1):
        res = call({"model": "jev-latest", "state": p["state"], "questions": QUESTION})
        ok = res["status"] == 200
        t = {
            "family": p["family"], "target": p["target"], "repeat": rep,
            "qwen_tokens": p["qwen_tokens"],
            "billed_tokens": res["body"]["usage"]["input_tokens"] if ok else None,
            "server_ms": res.get("server_ms"), "wall_ms": res.get("wall_ms"),
            "status": res["status"], "request_id": res.get("request_id"),
        }
        trials.append(t)
        print(f"[{i:3d}/{len(order)}] {t['family']:10s} target={t['target']:5d} "
              f"billed={t['billed_tokens']} qwen={t['qwen_tokens']} "
              f"server={t['server_ms']}ms status={t['status']}", flush=True)
    out = {"schema": "jev-counter-vs-model.v1",
           "date": datetime.now(timezone.utc).isoformat(), "seed": SEED,
           "empty_state_billed": base, "billed_per_unit": units,
           "targets": TARGETS, "repeats": REPEATS, "trials": trials}
    path = save("counter_vs_model_trials.json", out)
    print("saved", path)
    summarize(trials)


def _fit(xs, ys):
    """Least-squares slope (ms per 1k tokens) and intercept."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope * 1000, my - slope * mx


def summarize(trials: list) -> None:
    ok = [t for t in trials if t["status"] == 200 and t["server_ms"] is not None]
    print("\nMedian server ms by family and target:")
    fams = sorted({t["family"] for t in ok})
    targets = sorted({t["target"] for t in ok})
    print("family      " + "".join(f"{tg:>9d}" for tg in targets))
    for fam in fams:
        cells = []
        for tg in targets:
            v = [t["server_ms"] for t in ok if t["family"] == fam and t["target"] == tg]
            cells.append(f"{statistics.median(v):9.0f}" if v else f"{'':>9}")
        print(f"{fam:10s}  " + "".join(cells))
    print("\nSlope of server ms per 1k tokens (median-per-size fit):")
    print(f"{'family':10s} {'vs billed':>10s} {'vs qwen':>10s}  {'billed/qwen':>11s}")
    for fam in fams:
        pts = []
        for tg in targets:
            v = [t for t in ok if t["family"] == fam and t["target"] == tg]
            if not v:
                continue
            pts.append((statistics.median(t["billed_tokens"] for t in v),
                        statistics.median(t["qwen_tokens"] for t in v),
                        statistics.median(t["server_ms"] for t in v)))
        if len(pts) < 2:
            continue
        sb, _ = _fit([p[0] for p in pts], [p[2] for p in pts])
        sq, _ = _fit([p[1] for p in pts], [p[2] for p in pts])
        ratio = sum(p[0] for p in pts) / sum(p[1] for p in pts)
        print(f"{fam:10s} {sb:10.2f} {sq:10.2f}  {ratio:11.2f}")
    print("\nIf the counter is the model tokenizer, the 'vs billed' column is flat "
          "across families; if not, the 'vs qwen' column is the flat one.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summarize":
        data = json.loads((ROOT / "results" / "counter_vs_model_trials.json").read_text())
        summarize(data["trials"])
    else:
        main()
