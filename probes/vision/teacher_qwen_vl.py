"""Self-hosted logprob teacher: Qwen3-VL on the same Mind2Web `ground` items.

The distribution over candidate keys is read directly from the next-token logits after an
answer prefix (no sampling, no verbalised numbers): p(key) = softmax over the first token of
each key ("1".."9", "none"). One forward pass per item.

python teacher_qwen_vl.py --model Qwen/Qwen3-VL-32B-Instruct --items /workspace/m2w_items \
    --out /workspace/teacher_qwen3vl32b_m2w300.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

SYSTEM = ("You are grounding a web task to a UI element. The screenshot has numbered boxes drawn on candidate "
          "elements. Given the task and the actions already taken, decide which numbered element must be acted on "
          "next. If none of the boxed elements is right, answer none. Reply with the candidate key only.")


def prompt(item: dict) -> str:
    hist = "\n".join(f"  - {h}" for h in item["history"]) or "  (none)"
    cands = "\n".join(f"  {k}: {v}" for k, v in item["criteria"].items())
    return f"Task: {item['task']}\nActions already taken:\n{hist}\nCandidates:\n{cands}\nAnswer with the key only."


def metrics(rows):
    n = len(rows)
    acc = sum(r["hit"] for r in rows) / n
    bins = [[] for _ in range(10)]
    for r in rows:
        bins[min(9, int(r["conf"] * 10))].append(r)
    ece = sum(len(b) / n * abs(sum(x["hit"] for x in b) / len(b) - sum(x["conf"] for x in b) / len(b)) for b in bins if b)
    pos = [r["conf"] for r in rows if r["hit"]]; neg = [r["conf"] for r in rows if not r["hit"]]
    auroc = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg)) if pos and neg else float("nan")
    return {"n": n, "acc": round(acc, 3), "ece": round(ece, 3), "auroc": round(auroc, 3),
            "mean_top_p": round(sum(r["conf"] for r in rows) / n, 3)}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-32B-Instruct")
    ap.add_argument("--items", default="/workspace/m2w_items")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--max-pixels", type=int, default=1288 * 1000)
    ap.add_argument("--out", default="/workspace/teacher_qwen_vl.json")
    ap.add_argument("--temperature", type=float, default=1.0, help="global T applied to the stored soft targets")
    ap.add_argument("--requests-out", default=None,
                    help="also write s1 VL request rows with the (temperature-scaled) teacher distribution as targets")
    a = ap.parse_args()
    root = Path(a.items)
    items = [json.loads(l) for l in open(root / "items.jsonl")][: a.limit]
    proc = AutoProcessor.from_pretrained(a.model, max_pixels=a.max_pixels)
    model = Qwen3VLForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    tok = proc.tokenizer
    # first token id of each key as it would appear right after the assistant turn starts
    key_tok = {}
    for k in list(items[0]["criteria"]):
        ids = tok.encode(k, add_special_tokens=False)
        key_tok[k] = ids[0]
    print("key -> first token:", {k: tok.decode([t]) for k, t in key_tok.items()}, flush=True)
    per_item, rows = {}, []
    t0 = time.time(); ms = []
    for i, it in enumerate(items, 1):
        img = Image.open(root / it["image"]).convert("RGB")
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt(it)}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
        torch.cuda.synchronize(); s = time.perf_counter()
        logits = model(**inputs).logits[0, -1].float()
        torch.cuda.synchronize(); ms.append((time.perf_counter() - s) * 1000)
        keys = list(it["criteria"])
        z = torch.stack([logits[key_tok[k]] for k in keys])
        p = torch.softmax(z, 0).tolist()
        dist = dict(zip(keys, p))
        # how much probability mass the full vocab puts on the candidate tokens (sanity: should be high)
        full = torch.softmax(logits, 0)
        mass = float(sum(full[key_tok[k]] for k in keys))
        top = max(dist, key=dist.get)
        per_item[it["id"]] = {"gold": it["gold"], "logprob": dist, "cand_mass": round(mass, 3)}
        rows.append({"conf": dist[top], "hit": top == it["gold"]})
        if i % 25 == 0:
            print(f"  {i}/{len(items)} {time.time()-t0:.0f}s  mean fwd {sum(ms)/len(ms):.0f} ms  running acc {sum(r['hit'] for r in rows)/len(rows):.3f}", flush=True)
    rep = metrics(rows)
    rep.update({"model": a.model, "mean_forward_ms": round(sum(ms) / len(ms), 1),
                "mean_cand_mass": round(sum(r["cand_mass"] for r in per_item.values()) / len(per_item), 3),
                "visual_tokens_est": int(inputs["input_ids"].shape[1])})
    print("logprob", rep, flush=True)
    Path(a.out).write_text(json.dumps({"report": rep, "items": per_item}, indent=1))
    print("saved", a.out)
    if a.requests_out:
        # soft-labelled request rows: same image/state/question as requests.jsonl, targets = teacher p^(1/T)
        req = {json.loads(l)["id"]: json.loads(l) for l in open(root / "requests.jsonl")} if (root / "requests.jsonl").exists() else {}
        n = 0
        with open(a.requests_out, "w") as f:
            for it in items:
                r = req.get(it["id"])
                if r is None:
                    continue
                p = per_item[it["id"]]["logprob"]
                z = {k: math.log(max(v, 1e-9)) / a.temperature for k, v in p.items()}
                m = max(z.values()); e = {k: math.exp(v - m) for k, v in z.items()}; s = sum(e.values())
                r = {**r, "targets": {"ground": {k: round(v / s, 5) for k, v in e.items()}}, "teacher": a.model, "teacher_T": a.temperature}
                f.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
        print(f"wrote {n} soft-labelled request rows to {a.requests_out}")


if __name__ == "__main__":
    main()
