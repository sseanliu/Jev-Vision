"""K-sweep: does adding irrelevant options change the odds between the original
options in a way that depends on K (a list-dependent temperature) or on content
(genuine option interaction)?

Design
- Same state and instructions as Hume's odds_intervention (followup-trials.json).
- Original four options: bank, provider, customer, unknown.
- For K in KS, append K-4 irrelevant options drawn from one of two disjoint
  pools (content A / content B), so each K has two content variants.
- BLOCKS randomised blocks; within a block every (K, content) request is sent
  DUPS times so duplicate noise can be pooled, as in Hume's design.

Predictions
- Temperature-only: all pairwise log-odds among the four originals shrink by a
  common factor that is monotone in K and identical across content A/B.
- Content interaction: non-proportional shifts and A/B differences.
"""

from __future__ import annotations

import itertools
import math
import random
import sys
from datetime import datetime, timezone

from common import call, save

STATE = (
    "Ticket from Dana: Payouts to my bank have been failing since Tuesday. Your "
    "status page says all systems are fine, and my bank says nothing is blocked on "
    "their side, though they did change my account number last month after a merger. "
    "I updated it in your dashboard, I think. I run a small bakery and this is "
    "starting to hurt. If this isn't fixed by Friday I may have to look at other "
    "providers."
)
INSTRUCTIONS = "Who most likely caused the payout failures?"
ORIGINAL = {
    "bank": "The bank caused it",
    "provider": "The payment provider caused it",
    "customer": "The customer caused it",
    "unknown": "Cannot tell",
}
POOL_A = [
    ("weather", "Bad weather caused it"),
    ("birds", "Wild birds caused it"),
    ("moon", "The phase of the moon caused it"),
    ("volcano", "A volcanic eruption caused it"),
    ("football", "A football match caused it"),
    ("opera", "An opera performance caused it"),
    ("glacier", "A glacier caused it"),
    ("kite", "A kite festival caused it"),
    ("comet", "A passing comet caused it"),
    ("chess", "A chess tournament caused it"),
    ("tide", "The ocean tide caused it"),
    ("orchestra", "An orchestra rehearsal caused it"),
    ("marathon", "A marathon caused it"),
    ("sculpture", "A sculpture exhibition caused it"),
    ("aurora", "The northern lights caused it"),
    ("penguin", "A penguin colony caused it"),
    ("harvest", "The wheat harvest caused it"),
    ("ballet", "A ballet recital caused it"),
    ("meteor", "A meteor shower caused it"),
    ("cactus", "A cactus bloom caused it"),
    ("carnival", "A street carnival caused it"),
    ("lighthouse", "A lighthouse caused it"),
    ("poetry", "A poetry reading caused it"),
    ("canyon", "A canyon echo caused it"),
    ("violin", "A violin solo caused it"),
    ("sandstorm", "A desert sandstorm caused it"),
    ("puppet", "A puppet show caused it"),
    ("eclipse", "A solar eclipse caused it"),
]
POOL_B = [
    ("dragon", "A dragon caused it"),
    ("bicycle", "A bicycle race caused it"),
    ("pumpkin", "A giant pumpkin caused it"),
    ("jazz", "A jazz concert caused it"),
    ("iceberg", "An iceberg caused it"),
    ("circus", "A travelling circus caused it"),
    ("tulip", "A tulip garden caused it"),
    ("rainbow", "A double rainbow caused it"),
    ("yodel", "A yodelling contest caused it"),
    ("whale", "A migrating whale caused it"),
    ("origami", "An origami workshop caused it"),
    ("fossil", "A dinosaur fossil caused it"),
    ("lantern", "A lantern parade caused it"),
    ("hammock", "A hammock caused it"),
    ("bagpipe", "A bagpipe band caused it"),
    ("snowman", "A snowman caused it"),
    ("kayak", "A kayak trip caused it"),
    ("museum", "A museum opening caused it"),
    ("fireworks", "A fireworks display caused it"),
    ("scarecrow", "A scarecrow caused it"),
    ("trampoline", "A trampoline caused it"),
    ("waterfall", "A waterfall caused it"),
    ("accordion", "An accordion caused it"),
    ("hedgehog", "A hedgehog caused it"),
    ("quilt", "A quilting circle caused it"),
    ("sunflower", "A sunflower field caused it"),
    ("tango", "A tango lesson caused it"),
    ("windmill", "A windmill caused it"),
]
KS = [4, 5, 6, 8, 12, 16, 24, 32]
BLOCKS = 6
DUPS = 2
SEED = 20260917


def build(k: int, pool: list) -> dict:
    criteria = dict(ORIGINAL)
    for key, desc in pool[: k - 4]:
        criteria[key] = desc
    return {
        "model": "jev-latest",
        "state": STATE,
        "questions": {
            "decision": {
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": criteria,
            }
        },
    }


def main() -> None:
    rng = random.Random(SEED)
    conditions = []
    for k in KS:
        if k == 4:
            conditions.append((k, "A"))  # K=4 has no appended content
        else:
            conditions += [(k, "A"), (k, "B")]
    trials = []
    n_total = BLOCKS * len(conditions) * DUPS
    i = 0
    for block in range(BLOCKS):
        order = conditions[:]
        rng.shuffle(order)
        for k, content in order:
            pool = POOL_A if content == "A" else POOL_B
            payload = build(k, pool)
            for dup in range(DUPS):
                i += 1
                res = call(payload)
                trials.append(
                    {
                        "block": block,
                        "k": k,
                        "content": content,
                        "dup": dup,
                        "payload": payload,
                        "response": res,
                    }
                )
                ans = res.get("body", {}).get("answers", {}).get("decision", {})
                p = ans.get("probabilities", {})
                print(
                    f"[{i:3d}/{n_total}] block={block} K={k:2d} {content} dup={dup} "
                    f"status={res['status']} server={res.get('server_ms')}ms "
                    f"cust={p.get('customer')} unk={p.get('unknown')} "
                    f"prov={p.get('provider')} bank={p.get('bank')}",
                    flush=True,
                )
    out = {
        "schema": "jev-k-sweep.v1",
        "date": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "design": {"KS": KS, "BLOCKS": BLOCKS, "DUPS": DUPS},
        "trials": trials,
    }
    path = save("k_sweep_trials.json", out)
    print("saved", path)
    summarize(trials)


def summarize(trials: list) -> None:
    """Pool dups per (block, K, content); report pairwise log-odds among originals."""
    pairs = list(itertools.combinations(ORIGINAL, 2))
    pooled: dict[tuple, dict[str, list[float]]] = {}
    mass_irrelevant: dict[tuple, list[float]] = {}
    for t in trials:
        if t["response"]["status"] != 200:
            continue
        p = t["response"]["body"]["answers"]["decision"]["probabilities"]
        key = (t["block"], t["k"], t["content"])
        for a, b in pairs:
            pa, pb = max(p.get(a, 0), 0.005), max(p.get(b, 0), 0.005)
            pooled.setdefault(key, {}).setdefault(f"{a}/{b}", []).append(
                math.log(pa / pb)
            )
        mass_irrelevant.setdefault(key, []).append(
            sum(v for kk, v in p.items() if kk not in ORIGINAL)
        )
    print("\nMean log-odds among original options, by K and content (pooled over blocks and dups)")
    header = "K  cont " + " ".join(f"{a[:4]}/{b[:4]:<5}" for a, b in pairs) + "  irrelevant-mass"
    print(header)
    rows = {}
    for (block, k, content), d in pooled.items():
        rows.setdefault((k, content), {}).setdefault("_n", 0)
        rows[(k, content)]["_n"] += 1
        for pair, vals in d.items():
            rows[(k, content)].setdefault(pair, []).append(sum(vals) / len(vals))
        rows[(k, content)].setdefault("_mass", []).append(
            sum(mass_irrelevant[(block, k, content)]) / len(mass_irrelevant[(block, k, content)])
        )
    base = rows.get((4, "A"), {})
    for (k, content) in sorted(rows):
        r = rows[(k, content)]
        cells = []
        for a, b in pairs:
            v = r[f"{a}/{b}"]
            cells.append(f"{sum(v)/len(v):+.2f}     ")
        m = r["_mass"]
        print(f"{k:2d} {content}    " + " ".join(cells) + f"  {sum(m)/len(m):.3f}")
    # Proportionality check: ratio of each pair's log-odds to its K=4 value.
    print("\nRatio to K=4 (temperature-only predicts one common ratio per row):")
    for (k, content) in sorted(rows):
        if k == 4:
            continue
        r = rows[(k, content)]
        ratios = []
        for a, b in pairs:
            b0 = sum(base[f"{a}/{b}"]) / len(base[f"{a}/{b}"])
            v = sum(r[f"{a}/{b}"]) / len(r[f"{a}/{b}"])
            ratios.append(v / b0 if abs(b0) > 0.15 else float("nan"))
        shown = " ".join(f"{x:5.2f}" if x == x else "  n/a" for x in ratios)
        print(f"{k:2d} {content}    {shown}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summarize":
        import json
        from common import ROOT

        data = json.loads((ROOT / "results" / "k_sweep_trials.json").read_text())
        summarize(data["trials"])
    else:
        main()
