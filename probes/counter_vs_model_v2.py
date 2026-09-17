"""Paired version of counter_vs_model: the v1 fit was swamped by server jitter
(intercepts differed by 3x across families; plain slope SE > slope).

Design
- For each family, build a LONG state (~28k billed) and a SHORT state (~4k).
- Round-robin over families; for each family send LONG then SHORT back to
  back, so slow drift in server load cancels within the pair.
- PAIRS pairs per family. Statistic: delta_ms / delta_billed_tokens per pair,
  reported as median and bootstrap-free trimmed mean with SE.

Prediction
- counter == model tokenizer: delta per billed token equal across families.
- counter != model tokenizer: families the counter over-counts show a smaller
  delta per billed token, under-counted ones larger.
"""

from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime, timezone

from common import ROOT, call, save
from counter_vs_model import FAMILIES, QUESTION, billed, qwen_counter

LONG, SHORT = 28000, 4000
PAIRS = 24


def state_for(unit: str, per: float, target: int) -> str:
    return unit * max(1, round(target / per))


def main() -> None:
    qwen = qwen_counter()
    base = billed("")
    units = {f: (billed(u * 50) - base) / 50 for f, u in FAMILIES.items()}
    states = {f: (state_for(u, units[f], LONG), state_for(u, units[f], SHORT))
              for f, u in FAMILIES.items()}
    fams = list(FAMILIES)
    trials = []
    n = 0
    for pair in range(PAIRS):
        for f in fams:
            rec = {"family": f, "pair": pair}
            for tag, st in (("long", states[f][0]), ("short", states[f][1])):
                res = call({"model": "jev-latest", "state": st, "questions": QUESTION})
                n += 1
                rec[tag] = {"status": res["status"], "server_ms": res.get("server_ms"),
                            "billed": res["body"]["usage"]["input_tokens"] if res["status"] == 200 else None,
                            "qwen": qwen(st)}
            trials.append(rec)
            l, s = rec["long"], rec["short"]
            print(f"[{n:3d}/{PAIRS*len(fams)*2}] {f:10s} pair={pair:2d} "
                  f"long={l['server_ms']}ms short={s['server_ms']}ms "
                  f"delta={(l['server_ms'] or 0)-(s['server_ms'] or 0)}ms", flush=True)
    out = {"schema": "jev-counter-vs-model-paired.v1",
           "date": datetime.now(timezone.utc).isoformat(),
           "empty_state_billed": base, "billed_per_unit": units,
           "pairs": PAIRS, "trials": trials}
    path = save("counter_vs_model_v2_trials.json", out)
    print("saved", path)
    summarize(trials)


def summarize(trials) -> None:
    fams = sorted({t["family"] for t in trials})
    print("\nPaired slope: (ms_long - ms_short) per 1k tokens")
    print(f"{'family':10s} {'n':>3s} {'vs billed: median':>18s} {'mean ± SE':>14s} "
          f"{'vs qwen: median':>16s} {'billed/qwen':>12s}")
    for f in fams:
        ts = [t for t in trials if t["family"] == f
              and t["long"]["status"] == 200 and t["short"]["status"] == 200]
        sb, sq = [], []
        for t in ts:
            d = t["long"]["server_ms"] - t["short"]["server_ms"]
            sb.append(1000 * d / (t["long"]["billed"] - t["short"]["billed"]))
            sq.append(1000 * d / (t["long"]["qwen"] - t["short"]["qwen"]))
        if not sb:
            continue
        m = statistics.mean(sb)
        se = statistics.stdev(sb) / len(sb) ** 0.5 if len(sb) > 1 else float("nan")
        ratio = statistics.mean(t["long"]["billed"] / t["long"]["qwen"] for t in ts)
        print(f"{f:10s} {len(sb):3d} {statistics.median(sb):18.2f} {m:8.2f} ± {se:4.2f} "
              f"{statistics.median(sq):16.2f} {ratio:12.2f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summarize":
        data = json.loads((ROOT / "results" / "counter_vs_model_v2_trials.json").read_text())
        summarize(data["trials"])
    else:
        main()
