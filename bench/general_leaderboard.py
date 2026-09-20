"""Leaderboard for the general track from eval JSONs (eval_schema.py or baseline_frozen_vl.py reports).

python bench/general_leaderboard.py results/vision/general/zeroshot_base.json:"Qwen3-VL-8B frozen, logit readout" results/vision/general/zeroshot_v5b.json:"V5b (ours)" > bench/general_leaderboard.md
"""

import json
import sys

SOURCES = ["pope", "mme", "nlvr2", "aokvqa", "food101"]


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
        rows.append((name or path, cells, sum(accs) / len(accs), r.get("mean_ms")))
    print("| system | POPE | MME | NLVR2 (2 images) | A-OKVQA (4-way) | Food-101 (20-way) | mean acc | ms |")
    print("|---|---|---|---|---|---|---|---|")
    for name, cells, mean, ms in rows:
        print(f"| {name} | " + " | ".join(cells) + f" | {mean:.3f} | {ms} |")
    print("\nCells: accuracy / AUROC (yes-no sources) (ECE). ms: mean per request on one H100, batch 1; ours packs all questions of an image in one forward, the frozen baseline is one forward per question.")


if __name__ == "__main__":
    main()
