"""Held-out evaluation of a trained checkpoint.

python eval.py runs/s1/final --sets mmlu mmlu_pro boolq_val --limit 1000

Reports accuracy, ECE (10 bins), Brier and NLL per set, plus the permutation
sensitivity Jev exhibits: mean |p_top(original) - p_top(reversed options)|.
Held-out sets were never converted for training.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s1.model import DecisionModel  # noqa: E402
from s1.packing import Packer, Question  # noqa: E402
from train import expected_calibration_error  # noqa: E402


def load(path: Path, device):
    cfg = json.loads((path / "s1_config.json").read_text())
    tok = AutoTokenizer.from_pretrained(path / "backbone")
    model = DecisionModel(str(path / "backbone"), rank=cfg["rank"], attn_implementation="sdpa",
                          torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    heads = torch.load(path / "heads.pt", map_location="cpu")
    model.load_state_dict(heads, strict=False)
    packer = Packer(tok)
    return model.to(device).eval(), packer


def set_mmlu(limit, rng):
    ds = load_dataset("cais/mmlu", "all", split="test")
    idx = rng.sample(range(len(ds)), min(limit, len(ds)))
    items = []
    for i in idx:
        ex = ds[i]
        crit = {chr(97 + j): c for j, c in enumerate(ex["choices"])}
        items.append((ex["question"], "Which option correctly answers the question in the state?", crit,
                      chr(97 + ex["answer"])))
    return items


def set_mmlu_pro(limit, rng):
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    idx = rng.sample(range(len(ds)), min(limit, len(ds)))
    items = []
    for i in idx:
        ex = ds[i]
        crit = {chr(97 + j): c for j, c in enumerate(ex["options"])}
        items.append((ex["question"], "Which option correctly answers the question in the state?", crit,
                      ex["answer"].lower()))
    return items


def set_anli(limit, rng):
    ds = load_dataset("facebook/anli", split="test_r3")
    labels = {"entailment": "The hypothesis follows from the premise",
              "neutral": "The hypothesis may or may not be true given the premise",
              "contradiction": "The hypothesis contradicts the premise"}
    keys = list(labels)
    idx = rng.sample(range(len(ds)), min(limit, len(ds)))
    return [(f"Premise: {ds[i]['premise']}\nHypothesis: {ds[i]['hypothesis']}",
             "What is the relationship between the hypothesis and the premise?", labels, keys[ds[i]["label"]])
            for i in idx]


SETS = {"mmlu": set_mmlu, "mmlu_pro": set_mmlu_pro, "anli": set_anli}


@torch.no_grad()
def run_set(model, packer, items, device):
    confs, hits, briers, nlls, perm = [], [], [], [], []
    for state, instr, crit, gold in items:
        q = Question("q", "choice", instr, crit)
        z = model(packer.pack(state, [q]), device)[0]
        p = torch.softmax(z.float(), -1)
        keys = list(crit)
        gi = keys.index(gold)
        top = int(p.argmax())
        confs.append(p[top].item())
        hits.append(top == gi)
        onehot = torch.zeros_like(p)
        onehot[gi] = 1
        briers.append(((p - onehot) ** 2).sum().item())
        nlls.append(-torch.log(p[gi].clamp_min(1e-9)).item())
        # permutation sensitivity: reversed option order
        rq = Question("q", "choice", instr, dict(reversed(list(crit.items()))))
        zr = model(packer.pack(state, [rq]), device)[0]
        pr = torch.softmax(zr.float(), -1)
        perm.append(abs(p[gi].item() - pr[list(reversed(keys)).index(gold)].item()))
    n = len(items)
    return {"n": n, "acc": sum(hits) / n, "ece": expected_calibration_error(confs, hits),
            "brier": sum(briers) / n, "nll": sum(nlls) / n, "mean_top_p": sum(confs) / n,
            "perm_abs_delta_p_gold": sum(perm) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--sets", nargs="+", default=["mmlu", "mmlu_pro", "anli"])
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, packer = load(Path(a.checkpoint), device)
    rng = random.Random(a.seed)
    out = {}
    for name in a.sets:
        items = SETS[name](a.limit, rng)
        out[name] = run_set(model, packer, items, device)
        print(name, json.dumps(out[name]), flush=True)
    (Path(a.checkpoint) / "eval.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
