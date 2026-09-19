"""Evaluate a vision checkpoint on schema rows (any number of typed questions per screenshot)
and report per-question-id metrics. Used both for the V2 eval and as the zero-shot
"schema flexibility" probe: run it on a checkpoint that was trained on ground only and see
how well it answers question types it never saw.

python eval_schema.py runs/v1b-8b-m2w-jitter/final --data /workspace/m2w_items_j/schema.validation.jsonl --temp 0.5
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch

from eval_vl import auroc, load
from s1.dataset_vl import VLRequestDataset
from train import expected_calibration_error


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-pixels", type=int, default=1288 * 1000)
    ap.add_argument("--temp", type=float, default=1.0, help="global temperature applied to choice/score logits")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, packer = load(Path(a.checkpoint), device, a.max_pixels)
    ds = VLRequestDataset([Path(a.data)], packer, limit=a.limit)
    per = defaultdict(lambda: {"conf": [], "hit": [], "nll": [], "abs_err": [], "n_opts": []})
    ms = []
    for i in range(len(ds)):
        ex = ds.build(ds.rows[i])
        t0 = time.perf_counter()
        outs = model(ex.packed, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1000)
        for z, t, qtype, qid, keys in zip(outs, ex.targets, ex.packed.qtypes, [q.qid for q in ds_questions(ds.rows[i])], ex.packed.option_keys):
            z = z.float()
            if qtype == "noul":
                p = torch.sigmoid(z).item(); pred = p >= 0.5; gold = t >= 0.5
                per[qid]["conf"].append(max(p, 1 - p)); per[qid]["hit"].append(pred == gold)
                per[qid]["nll"].append(-math.log(max(p if gold else 1 - p, 1e-9))); per[qid]["n_opts"].append(2)
            else:
                pr = torch.softmax(z / a.temp, -1); gi = int(torch.tensor(t).argmax()) if isinstance(t, list) else int(t)
                per[qid]["conf"].append(pr.max().item()); per[qid]["hit"].append(int(pr.argmax()) == gi)
                per[qid]["nll"].append(-math.log(max(pr[gi].item(), 1e-9))); per[qid]["n_opts"].append(len(keys))
                if qtype == "score":
                    per[qid]["abs_err"].append(abs(float((pr * torch.arange(len(pr))).sum()) - gi))
    rep = {"n_rows": len(ds), "mean_ms": round(sum(ms) / len(ms), 1), "temp": a.temp, "by_question": {}}
    print(f"{'qid':12s} {'n':>5s} {'acc':>6s} {'chance':>7s} {'ece':>6s} {'auroc':>6s} {'nll':>6s} {'mae':>5s}")
    for qid, d in sorted(per.items()):
        n = len(d["hit"]); acc = sum(d["hit"]) / n; chance = sum(1 / k for k in d["n_opts"]) / n
        r = {"n": n, "acc": round(acc, 3), "chance": round(chance, 3), "ece": round(expected_calibration_error(d["conf"], d["hit"]), 3),
             "auroc": round(auroc(d["conf"], d["hit"]), 3), "nll": round(sum(d["nll"]) / n, 3),
             "mae": round(sum(d["abs_err"]) / len(d["abs_err"]), 2) if d["abs_err"] else None}
        rep["by_question"][qid] = r
        print(f"{qid:12s} {n:5d} {r['acc']:6.3f} {r['chance']:7.3f} {r['ece']:6.3f} {r['auroc']:6.3f} {r['nll']:6.3f} {r['mae'] if r['mae'] is not None else '':>5}")
    print(f"mean {rep['mean_ms']} ms per screenshot (all questions in one forward)")
    out = Path(a.out or (Path(a.checkpoint) / f"eval_schema_{Path(a.data).stem}.json"))
    out.write_text(json.dumps(rep, indent=1)); print("saved", out)


def ds_questions(row):
    from s1.packing import Question
    return [Question(q["qid"], q["qtype"], q["instructions"], q.get("criteria")) for q in row["questions"]]


if __name__ == "__main__":
    main()
