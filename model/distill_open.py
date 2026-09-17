"""Data condition (a): soft labels from an open-weight model.

Scores each option by the length-normalised log-likelihood of the option text
given state + question (standard multiple-choice scoring, as in lm-eval), with
a PMI correction against an empty state, then softmax at a temperature fitted
so that the mean top-probability matches the teacher's accuracy on a held-out
slice. Noul is scored as a two-option choice ("yes"/"no").

python distill_open.py --teacher Qwen/Qwen3-30B-A3B-Base --in data/jsonl \
    --out data/jsonl_open --per-source 3000 --bsz 32

Runs on the GPU pod after training; uses plain transformers (no vLLM) so it
works with the tree-mask-free base model as-is.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def render(state: str, q: dict, option_text: str) -> tuple[str, str]:
    prompt = f"{state}\n\nQuestion: {q['instructions']}\nAnswer:"
    return prompt, " " + option_text


@torch.no_grad()
def option_loglik(model, tok, prompts: list[str], continuations: list[str], device) -> list[float]:
    """Sum of log p(continuation tokens | prompt), batched with right padding."""
    enc_p = [tok.encode(p, add_special_tokens=False) for p in prompts]
    enc_c = [tok.encode(c, add_special_tokens=False) for c in continuations]
    seqs = [p + c for p, c in zip(enc_p, enc_c)]
    L = max(len(s) for s in seqs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        att[i, : len(s)] = 1
    logits = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits.float()
    logp = torch.log_softmax(logits, -1)
    out = []
    for i, (p, c) in enumerate(zip(enc_p, enc_c)):
        total = 0.0
        for j, tid in enumerate(c):
            pos = len(p) + j - 1  # logits at pos predict token at pos+1
            total += logp[i, pos, tid].item()
        out.append(total / max(len(c), 1))
    return out


def score_row(model, tok, row: dict, device, bsz: int) -> dict:
    soft = {}
    for q in row["questions"]:
        if q["qtype"] == "noul":
            keys, texts = ["yes", "no"], ["yes", "no"]
        elif q["qtype"] == "choice":
            keys = list(q["criteria"])
            texts = [f"{k}: {d}" for k, d in q["criteria"].items()]
        else:
            keys = [str(i) for i in range(len(q["criteria"]))]
            texts = [f"level {i}: {d}" for i, d in enumerate(q["criteria"])]
        prompts, conts, prompts0 = [], [], []
        for t in texts:
            p, c = render(row["state"], q, t)
            p0, _ = render("", q, t)
            prompts.append(p)
            conts.append(c)
            prompts0.append(p0)
        ll, ll0 = [], []
        for i in range(0, len(texts), bsz):
            ll += option_loglik(model, tok, prompts[i:i + bsz], conts[i:i + bsz], device)
            ll0 += option_loglik(model, tok, prompts0[i:i + bsz], conts[i:i + bsz], device)
        scores = [a - b for a, b in zip(ll, ll0)]  # PMI: remove option prior
        soft[q["qid"]] = {"keys": keys, "scores": scores}
    return soft


def fit_temperature(rows_scored: list[tuple[dict, dict]]) -> float:
    """Pick T so that mean top-probability ≈ top-1 accuracy against hard labels."""
    best, best_gap = 1.0, 1e9
    for T in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]:
        conf, acc, n = 0.0, 0.0, 0
        for row, soft in rows_scored:
            for q in row["questions"]:
                if q["qtype"] == "noul":
                    continue
                s = soft[q["qid"]]
                z = torch.tensor(s["scores"]) / T
                p = torch.softmax(z, -1)
                hard = row["targets"][q["qid"]]
                gold = s["keys"].index(hard if q["qtype"] == "choice" else str(hard))
                conf += p.max().item()
                acc += float(int(p.argmax()) == gold)
                n += 1
        gap = abs(conf / max(n, 1) - acc / max(n, 1))
        if gap < best_gap:
            best, best_gap = T, gap
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="Qwen/Qwen3-30B-A3B-Base")
    ap.add_argument("--in", dest="inp", default="data/jsonl")
    ap.add_argument("--out", default="data/jsonl_open")
    ap.add_argument("--per-source", type=int, default=3000)
    ap.add_argument("--bsz", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(a.teacher)
    model = AutoModelForCausalLM.from_pretrained(a.teacher, dtype=torch.bfloat16, device_map="cuda").eval()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    for path in sorted(Path(a.inp).glob("*.train.jsonl")):
        rows = [json.loads(l) for l in open(path)]
        rng.shuffle(rows)
        rows = rows[: a.per_source]
        scored = [(r, score_row(model, tok, r, device, a.bsz)) for r in rows]
        T = fit_temperature(scored[: 300])
        with (out / path.name).open("w") as f:
            for row, soft in scored:
                targets = {}
                for q in row["questions"]:
                    s = soft[q["qid"]]
                    p = torch.softmax(torch.tensor(s["scores"]) / T, -1).tolist()
                    if q["qtype"] == "noul":
                        targets[q["qid"]] = float(p[0])
                    else:
                        targets[q["qid"]] = {k: round(v, 4) for k, v in zip(s["keys"], p)}
                f.write(json.dumps({**row, "targets_hard": row["targets"], "targets": targets,
                                    "teacher": a.teacher, "temperature": T}, ensure_ascii=False) + "\n")
        print(f"{path.name:32s} n={len(scored)} T={T}", flush=True)


if __name__ == "__main__":
    main()
