"""Frozen-VLM baseline for typed questions: prompt the base model and read the answer from the next-token logits.

No fine-tuning, no heads: `noul` reads P(yes) = softmax over the "yes"/"no" first tokens; `choice` reads a softmax
over the option letters. Same row format as model/eval_schema.py (images, state, questions[{qid,qtype,instructions,
criteria}], targets) and the same report shape (by_question acc/ece/auroc, per-item predictions), so the two are
directly comparable. This is what a "prompt-only open vision Jev" gets from the same backbone.

python baseline_frozen_vl.py --model Qwen/Qwen3-VL-8B-Instruct --data /workspace/bench/general/general.rows.jsonl --out /workspace/base_general.json
"""

import argparse
import json
import math
import time

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def ece(conf, hit, bins=10):
    n = len(conf); tot = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if idx:
            tot += len(idx) / n * abs(sum(hit[i] for i in idx) / len(idx) - sum(conf[i] for i in idx) / len(idx))
    return tot


def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]; neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct"); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--max-pixels", type=int, default=1288 * 1000)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(a.model, max_pixels=a.max_pixels)
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map=device).eval()
    tok = proc.tokenizer
    yes_ids = [tok.encode(" yes")[-1], tok.encode("yes")[-1], tok.encode(" Yes")[-1], tok.encode("Yes")[-1]]
    no_ids = [tok.encode(" no")[-1], tok.encode("no")[-1], tok.encode(" No")[-1], tok.encode("No")[-1]]
    letter_ids = {L: [tok.encode(" " + L)[-1], tok.encode(L)[-1]] for L in LETTERS}
    rows = [json.loads(l) for l in open(a.data)]
    if a.limit:
        rows = rows[: a.limit]
    per = {}; items = []; ms = []
    for i, r in enumerate(rows):
        imgs = [Image.open(p).convert("RGB") for p in (r.get("images") or [])]
        for q in r["questions"]:
            qid = q["qid"]; gold = r["targets"][qid]
            content = [{"type": "image"} for _ in imgs]
            if q["qtype"] == "noul":
                text = f"{r['state']}\n{q['instructions']}\nAnswer with yes or no."
            else:
                keys = list(q["criteria"].keys()); letters = LETTERS[: len(keys)]
                opts = "\n".join(f"{L}. {q['criteria'][k]}" for L, k in zip(letters, keys))
                text = f"{r['state']}\n{q['instructions']}\n{opts}\nAnswer with the letter only."
            content.append({"type": "text", "text": text})
            msgs = [{"role": "user", "content": content}]
            prompt = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            inputs = (proc(text=[prompt], images=imgs, return_tensors="pt") if imgs else proc(text=[prompt], return_tensors="pt")).to(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                logits = model(**inputs).logits[0, -1].float()
            if device == "cuda":
                torch.cuda.synchronize()
            ms.append((time.perf_counter() - t0) * 1000)
            d = per.setdefault(qid, {"conf": [], "hit": [], "p": [], "y": [], "n_opts": 0})
            if q["qtype"] == "noul":
                ly = torch.logsumexp(logits[yes_ids], 0); ln = torch.logsumexp(logits[no_ids], 0)
                p = torch.sigmoid(ly - ln).item(); pred = p >= 0.5; g = bool(gold)
                d["conf"].append(max(p, 1 - p)); d["hit"].append(pred == g); d["p"].append(p); d["y"].append(int(g)); d["n_opts"] = 2
                items.append({"i": i, "qid": qid, "p": round(p, 4), "y": int(g), "meta": r.get("meta", {})})
            else:
                lg = torch.stack([torch.logsumexp(logits[letter_ids[L]], 0) for L in letters])
                pr = torch.softmax(lg, 0); k = int(pr.argmax()); pred_key = keys[k]
                d["conf"].append(pr.max().item()); d["hit"].append(pred_key == gold); d["n_opts"] = len(keys)
                items.append({"i": i, "qid": qid, "pred": pred_key, "gold": gold, "p": round(pr.max().item(), 4), "meta": r.get("meta", {})})
    rep = {"model": a.model, "n_rows": len(rows), "mean_ms": round(sum(ms) / max(1, len(ms)), 1), "by_question": {}}
    for qid, d in sorted(per.items()):
        n = len(d["hit"]); acc = sum(d["hit"]) / n
        e = {"n": n, "acc": round(acc, 3), "chance": round(1 / d["n_opts"], 3), "ece": round(ece(d["conf"], d["hit"]), 3)}
        if d["p"]:
            e["auroc"] = round(auroc(d["p"], d["y"]), 3) if auroc(d["p"], d["y"]) is not None else None
            pos = [h for h, y in zip(d["hit"], d["y"]) if y]; neg = [h for h, y in zip(d["hit"], d["y"]) if not y]
            e["acc_pos"] = round(sum(pos) / len(pos), 3) if pos else None; e["acc_neg"] = round(sum(neg) / len(neg), 3) if neg else None
        rep["by_question"][qid] = e
        print(f"{qid:10} n={n:4} acc={acc:.3f} ece={e['ece']:.3f}" + (f" auroc={e.get('auroc')}" if d["p"] else ""))
    rep["items"] = items
    print("mean", rep["mean_ms"], "ms per question")
    json.dump(rep, open(a.out, "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
