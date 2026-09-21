"""Leaderboard for the general track from eval JSONs (eval_schema.py or baseline_frozen_vl.py reports).

python bench/general_leaderboard.py results/vision/general/zeroshot_base.json:"Qwen3-VL-8B frozen, logit readout" results/vision/general/zeroshot_v5b.json:"V5b (ours)" > bench/general_leaderboard.md
"""

import json
import random
import sys

SOURCES = ["pope", "mme", "nlvr2", "aokvqa", "food101"]


def item_hits(r):
    """Per-item (correct, confidence) from the saved predictions, for CIs and high-confidence errors."""
    out = []
    for it in r.get("items", []):
        if "y" in it:
            p = it["p"]; out.append((int((p >= 0.5) == bool(it["y"])), max(p, 1 - p), it["qid"]))
        elif "pred" in it:
            out.append((int(it["pred"] == it["gold"]), it["p"], it["qid"]))
    return out


def ci95(hits, n_boot=2000, seed=0):
    if not hits:
        return None
    rng = random.Random(seed); n = len(hits); means = []
    for _ in range(n_boot):
        means.append(sum(hits[rng.randrange(n)] for _ in range(n)) / n)
    means.sort(); return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def main():
    rows = []
    for arg in sys.argv[1:]:
        path, _, name = arg.partition(":")
        r = json.load(open(path)); bq = r["by_question"]
        cells = []
        for s in SOURCES:
            e = bq.get(s)
            if not e:
                cells.append("-"); continue
            c = f"{e['acc']:.3f}"
            if e.get("auroc") is not None:
                c += f" / {e['auroc']:.2f}"
            c += f" (ECE {e['ece']:.3f})"
            cells.append(c)
        accs = [bq[s]["acc"] for s in SOURCES if s in bq]
        ih = item_hits(r); hits = [h for h, _, _ in ih]
        lo, hi = ci95(hits) if hits else (None, None)
        errs = [c for h, c, _ in ih if not h]; hi_err = sum(1 for c in errs if c >= 0.99)
        rows.append((name or path, cells, sum(accs) / len(accs), (lo, hi), (hi_err, len(errs)), r.get("mean_ms")))
    print("| system | POPE | MME | NLVR2 (2 images) | A-OKVQA (4-way) | Food-101 (20-way) | mean acc [95% CI] | errors stated at p>=0.99 | ms |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, cells, mean, (lo, hi), (he, ne), ms in rows:
        ci = f" [{lo:.3f}-{hi:.3f}]" if lo is not None else ""
        print(f"| {name} | " + " | ".join(cells) + f" | {mean:.3f}{ci} | {he} of {ne} ({100 * he / max(1, ne):.0f}%) | {ms} |")
    print("\nCells: accuracy / AUROC (yes-no sources) (ECE). CI: 2,000-resample item bootstrap over all 1,500 items. Errors stated at p>=0.99: wrong answers given with at least 0.99 confidence (the Solomon card's high-confidence-error column). ms: mean per request on one H100, batch 1; ours packs all questions of an image in one forward, the frozen baseline is one forward per question.")


if __name__ == "__main__":
    main()
