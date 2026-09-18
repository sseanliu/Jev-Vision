"""Data condition (a): soft labels from an open-weight model.

Scores each option by the length-normalised log-likelihood of the option text
given state + question (standard multiple-choice scoring, as in lm-eval), with
a PMI correction against an empty state, then softmax at a temperature fitted
so that the mean top-probability matches the teacher's accuracy on a held-out
slice. Noul is scored as a two-option choice ("yes"/"no").

Throughput: all (row, question, option) sequences of a source are pooled,
sorted by length and scored in large batches; the empty-state prior depends
only on (instructions, option text) and is computed once per distinct pair.

python distill_open.py --teacher Qwen/Qwen3-30B-A3B-Base --in data/jsonl_jev \
    --out data/jsonl_open --per-source 3000 --bsz 64
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def render(state: str, q: dict, option_text: str) -> tuple[str, str]:
    prompt = f"{state}\n\nQuestion: {q['instructions']}\nAnswer:"
    return prompt, " " + option_text


def option_texts(q: dict) -> tuple[list[str], list[str]]:
    if q["qtype"] == "noul":
        return ["yes", "no"], ["yes", "no"]
    if q["qtype"] == "choice":
        return list(q["criteria"]), [f"{k}: {d}" for k, d in q["criteria"].items()]
    return [str(i) for i in range(len(q["criteria"]))], [f"level {i}: {d}" for i, d in enumerate(q["criteria"])]


@torch.no_grad()
def score_sequences(model, tok, pairs: list[tuple[str, str]], device, bsz: int, max_len: int = 1536,
                    token_budget: int = 24000) -> list[float]:
    """pairs: (prompt, continuation). Returns mean log p(continuation | prompt) per pair.
    Length-sorted batches capped by both sequence count and total tokens; the vocabulary
    log-softmax is taken only at continuation positions (a handful per sequence), so memory
    is O(batch_tokens x hidden) instead of O(batch_tokens x vocab)."""
    enc = []
    for p, c in pairs:
        pi = tok.encode(p, add_special_tokens=False)
        ci = tok.encode(c, add_special_tokens=False)
        if len(pi) + len(ci) > max_len:
            pi = pi[-(max_len - len(ci)):]
        enc.append((pi, ci))
    order = sorted(range(len(enc)), key=lambda i: len(enc[i][0]) + len(enc[i][1]))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out = [0.0] * len(enc)
    b = 0
    while b < len(order):
        idx, L = [], 0
        while b < len(order) and len(idx) < bsz:
            n = len(enc[order[b]][0]) + len(enc[order[b]][1])
            if idx and max(L, n) * (len(idx) + 1) > token_budget:
                break
            idx.append(order[b]); L = max(L, n); b += 1
        seqs = [enc[i][0] + enc[i][1] for i in idx]
        ids = torch.full((len(seqs), L), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for r, s in enumerate(seqs):
            ids[r, :len(s)] = torch.tensor(s)
            att[r, :len(s)] = 1
        logits = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits  # [B, L, V] bf16
        for r, i in enumerate(idx):
            pi, ci = enc[i]
            pos = torch.arange(len(pi) - 1, len(pi) - 1 + len(ci), device=logits.device)
            tgt = torch.tensor(ci, device=logits.device)
            lp = torch.log_softmax(logits[r, pos, :].float(), -1)  # [len(ci), V] only
            out[i] = lp.gather(1, tgt[:, None]).sum().item() / max(len(ci), 1)
        del logits
    return out


def score_source(model, tok, rows: list[dict], device, bsz: int) -> list[dict]:
    """Returns per-row {qid: {"keys": [...], "scores": [...]}} with PMI-corrected scores."""
    cond_pairs, cond_index = [], []          # (row_i, qid, option_j) -> position in cond_pairs
    prior_pairs, prior_key = [], {}          # distinct (instructions, option_text) -> position
    for ri, row in enumerate(rows):
        for q in row["questions"]:
            keys, texts = option_texts(q)
            for oj, t in enumerate(texts):
                p, c = render(row["state"], q, t)
                cond_index.append((ri, q["qid"], oj))
                cond_pairs.append((p, c))
                pk = (q["instructions"], t)
                if pk not in prior_key:
                    prior_key[pk] = len(prior_pairs)
                    prior_pairs.append(render("", q, t))
    t0 = time.time()
    cond = score_sequences(model, tok, cond_pairs, device, bsz)
    prior = score_sequences(model, tok, prior_pairs, device, bsz)
    print(f"    scored {len(cond_pairs)} conditional + {len(prior_pairs)} prior sequences in {time.time()-t0:.0f}s",
          flush=True)
    soft = [dict() for _ in rows]
    for (ri, qid, oj), ll in zip(cond_index, cond):
        row = rows[ri]
        q = next(qq for qq in row["questions"] if qq["qid"] == qid)
        keys, texts = option_texts(q)
        if qid not in soft[ri]:
            soft[ri][qid] = {"keys": keys, "scores": [0.0] * len(keys)}
        soft[ri][qid]["scores"][oj] = ll - prior[prior_key[(q["instructions"], texts[oj])]]
    return soft


def fit_temperature(rows, soft) -> float:
    """Pick T so that mean top-probability ≈ top-1 accuracy against hard labels."""
    best, best_gap = 1.0, 1e9
    for T in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]:
        conf, acc, n = 0.0, 0.0, 0
        for row, s_row in zip(rows, soft):
            for q in row["questions"]:
                if q["qtype"] == "noul":
                    continue
                s = s_row[q["qid"]]
                p = torch.softmax(torch.tensor(s["scores"]) / T, -1)
                hard = row.get("targets_hard", row["targets"])[q["qid"]]
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
    ap.add_argument("--bsz", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.teacher)
    model = AutoModelForCausalLM.from_pretrained(
        a.teacher, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="cuda" if device.type == "cuda" else None).eval()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    for path in sorted(Path(a.inp).glob("*.train.jsonl")):
        dest = out / path.name
        if dest.exists() and sum(1 for _ in open(dest)) >= min(a.per_source, sum(1 for _ in open(path))):
            print(f"{path.name:32s} exists, skipping", flush=True)
            continue
        rows = [json.loads(l) for l in open(path)]
        rng.shuffle(rows)
        rows = rows[: a.per_source]
        print(f"{path.name:32s} scoring {len(rows)} rows", flush=True)
        soft = score_source(model, tok, rows, device, a.bsz)
        T = fit_temperature(rows[:300], soft[:300])
        with dest.open("w") as f:
            for row, s_row in zip(rows, soft):
                targets = {}
                for q in row["questions"]:
                    s = s_row[q["qid"]]
                    p = torch.softmax(torch.tensor(s["scores"]) / T, -1).tolist()
                    targets[q["qid"]] = float(p[0]) if q["qtype"] == "noul" else {k: round(v, 4) for k, v in zip(s["keys"], p)}
                f.write(json.dumps({**row, "targets_hard": row.get("targets_hard", row["targets"]), "targets": targets,
                                    "teacher": a.teacher, "temperature": T}, ensure_ascii=False) + "\n")
        print(f"{path.name:32s} n={len(rows)} T={T}", flush=True)


if __name__ == "__main__":
    main()
