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


def load(path: Path, device, attn: str = "sdpa", dtype=None):
    cfg = json.loads((path / "s1_config.json").read_text())
    tok = AutoTokenizer.from_pretrained(path / "backbone")
    dtype = dtype or (torch.bfloat16 if device.type == "cuda" else torch.float32)
    adapter = path / "backbone" / "adapter_config.json"
    if adapter.exists():
        # LoRA checkpoint: rebuild base + resized vocab, then attach the adapter
        from peft import PeftModel
        base = json.loads(adapter.read_text())["base_model_name_or_path"]
        model = DecisionModel(base, rank=cfg["rank"], new_vocab_size=len(tok), attn_implementation=attn, torch_dtype=dtype)
        model.backbone = PeftModel.from_pretrained(model.backbone, str(path / "backbone"))
        model.lora = cfg.get("lora")
    else:
        model = DecisionModel(str(path / "backbone"), rank=cfg["rank"], attn_implementation=attn, torch_dtype=dtype)
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
def run_set(model, packer, items, device, temps=(1.0,)):
    """Also reports ECE/NLL under global temperature scaling for each T in temps
    (oracle per-set T shows how much of the miscalibration is mere sharpness)."""
    confs, hits, briers, nlls, perm = [], [], [], [], []
    by_T = {T: {"confs": [], "hits": [], "nlls": []} for T in temps}
    for state, instr, crit, gold in items:
        q = Question("q", "choice", instr, crit)
        z = model(packer.pack(state, [q]), device)[0]
        for T in temps:
            pT = torch.softmax(z.float() / T, -1)
            gi_ = list(crit).index(gold)
            by_T[T]["confs"].append(pT.max().item()); by_T[T]["hits"].append(int(pT.argmax()) == gi_)
            by_T[T]["nlls"].append(-torch.log(pT[gi_].clamp_min(1e-9)).item())
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
    # AUROC of top-p as a predictor of correctness: instance-level uncertainty, independent of
    # any global temperature (a format-locked constant confidence scores 0.5).
    pos = [c for c, h in zip(confs, hits) if h]; neg = [c for c, h in zip(confs, hits) if not h]
    auroc = (sum((p_ > n_) + 0.5 * (p_ == n_) for p_ in pos for n_ in neg) / (len(pos) * len(neg))) if pos and neg else float("nan")
    out = {"n": n, "acc": sum(hits) / n, "ece": expected_calibration_error(confs, hits), "auroc": auroc,
           "conf_std": (sum((c - sum(confs) / n) ** 2 for c in confs) / n) ** 0.5,
           "brier": sum(briers) / n, "nll": sum(nlls) / n, "mean_top_p": sum(confs) / n,
           "perm_abs_delta_p_gold": sum(perm) / n}
    if len(temps) > 1:
        out["by_temperature"] = {str(T): {"ece": expected_calibration_error(d["confs"], d["hits"]),
                                          "nll": sum(d["nlls"]) / n, "mean_top_p": sum(d["confs"]) / n}
                                 for T, d in by_T.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--sets", nargs="+", default=["mmlu", "mmlu_pro", "anli"])
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temps", default="1.0", help="comma-separated global temperatures to report, e.g. 1,1.5,2,3")
    a = ap.parse_args()
    temps = tuple(float(t) for t in a.temps.split(","))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, packer = load(Path(a.checkpoint), device)
    rng = random.Random(a.seed)
    out = {}
    for name in a.sets:
        items = SETS[name](a.limit, rng)
        out[name] = run_set(model, packer, items, device, temps)
        print(name, json.dumps(out[name]), flush=True)
    (Path(a.checkpoint) / ("eval.json" if temps == (1.0,) else "eval_temps.json")).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
