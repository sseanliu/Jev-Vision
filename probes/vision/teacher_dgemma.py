"""DiffusionGemma-as-Jev baseline on our `ground` items, via transformers (no vLLM patch).

Same trick as mmastrac's vLLM PR #57250: encode the prompt (screenshot + task + lettered
candidates) with the encoder, seed the diffusion canvas with the answer template
("ground: @") where @ is an arbitrary placeholder token, run ONE denoising step read-only,
and read the logits at the slot restricted to the label tokens. One forward per item, no
generation, exact distribution over the candidates.

python teacher_dgemma.py --model google/diffusiongemma-26B-A4B-it --items /workspace/m2w_items_j --limit 300 --out /workspace/teacher_dgemma_m2w300_j.json
python teacher_dgemma.py --model google/diffusiongemma-26B-A4B-it --items /workspace/m2w_items_j --probe 2   # print what the model generates, to check the template
"""

from __future__ import annotations

import argparse
import json
import math
import string
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

SYSTEM = ("You are grounding a web task to a UI element. The screenshot has numbered boxes drawn on candidate "
          "elements. Given the task and the actions already taken, decide which candidate must be acted on next. "
          "Answer with the candidate letter only.")
LETTERS = string.ascii_uppercase


def build_prompt(item: dict):
    keys = list(item["criteria"])
    letter = {k: LETTERS[i] for i, k in enumerate(keys)}
    hist = "\n".join(f"  - {h}" for h in item["history"]) or "  (none)"
    cands = "\n".join(f"  {letter[k]}: {v}" for k, v in item["criteria"].items())
    text = (f"Task: {item['task']}\nActions already taken:\n{hist}\nCandidates (letter: box number and element):\n{cands}\n"
            f"Which candidate should be acted on next? Reply with one line: ground: <letter>")
    return text, keys, letter


def metrics(rows):
    n = len(rows); acc = sum(r["hit"] for r in rows) / n
    bins = [[] for _ in range(10)]
    for r in rows:
        bins[min(9, int(r["conf"] * 10))].append(r)
    ece = sum(len(b) / n * abs(sum(x["hit"] for x in b) / len(b) - sum(x["conf"] for x in b) / len(b)) for b in bins if b)
    pos = [r["conf"] for r in rows if r["hit"]]; neg = [r["conf"] for r in rows if not r["hit"]]
    auroc = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg)) if pos and neg else float("nan")
    return {"n": n, "acc": round(acc, 3), "ece": round(ece, 3), "auroc": round(auroc, 3), "mean_top_p": round(sum(r["conf"] for r in rows) / n, 3)}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/diffusiongemma-26B-A4B-it")
    ap.add_argument("--items", default="/workspace/m2w_items_j")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--canvas", type=int, default=32)
    ap.add_argument("--prefix", default="<|channel>thought\n<channel|>", help="assistant-turn prefix to seed before the template")
    ap.add_argument("--probe", type=int, default=0, help="generate freely for N items and print the raw answer (template check)")
    ap.add_argument("--out", default="/workspace/teacher_dgemma.json")
    a = ap.parse_args()
    root = Path(a.items)
    items = [json.loads(l) for l in open(root / "items.jsonl")][: a.limit]
    proc = AutoProcessor.from_pretrained(a.model)
    tok = proc.tokenizer
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    dev = model.device

    def encode(item):
        text, keys, letter = build_prompt(item)
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [{"type": "image", "image": Image.open(root / item["image"]).convert("RGB")}, {"type": "text", "text": text}]}]
        enc = proc.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True,
                                       chat_template_kwargs={"enable_thinking": False})
        return {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in enc.items()}, keys, letter

    if a.probe:
        for it in items[: a.probe]:
            enc, keys, letter = encode(it)
            out = model.generate(**enc, max_new_tokens=32)
            print("GOLD", letter.get(it["gold"]), "| RAW:", repr(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=False)[:200]))
        return

    # canvas template: prefix + "ground: " + placeholder, padded to canvas length
    prefix_ids = tok.encode(a.prefix, add_special_tokens=False) if a.prefix else []
    tmpl_ids = tok.encode("ground:", add_special_tokens=False)
    slot = len(prefix_ids) + len(tmpl_ids)
    placeholder = tok.encode(" Mell", add_special_tokens=False)[0]
    pad = tok.pad_token_id or 0
    canvas = prefix_ids + tmpl_ids + [placeholder] + tok.encode("\n", add_special_tokens=False)
    canvas = (canvas + [pad] * a.canvas)[: a.canvas]
    canvas_t = torch.tensor([canvas], device=dev)
    # label token ids: both " A" and "A" spellings
    label_ids = {L: sorted({tok.encode(" " + L, add_special_tokens=False)[-1], tok.encode(L, add_special_tokens=False)[-1]}) for L in LETTERS[:26]}
    print("slot", slot, "canvas", tok.decode(canvas[:slot + 2]), "| label ids sample", {k: label_ids[k] for k in "AB"}, flush=True)

    per, rows, ms = {}, [], []
    t0 = time.time()
    for i, it in enumerate(items, 1):
        enc, keys, letter = encode(it)
        L = enc["input_ids"].shape[1]
        att = enc.get("attention_mask", torch.ones_like(enc["input_ids"])).bool()
        torch.cuda.synchronize(); s = time.perf_counter()
        out = model(**{k: v for k, v in enc.items() if k != "attention_mask"}, attention_mask=att,
                    decoder_input_ids=canvas_t, decoder_position_ids=torch.arange(L, L + a.canvas, device=dev).unsqueeze(0),
                    decoder_attention_mask=torch.nn.functional.pad(att, (0, a.canvas), value=True))
        torch.cuda.synchronize(); ms.append((time.perf_counter() - s) * 1000)
        logits = out.logits[0, slot].float()
        z = torch.stack([torch.logsumexp(logits[label_ids[letter[k]]], 0) for k in keys])
        p = torch.softmax(z, 0).tolist()
        dist = dict(zip(keys, p)); top = max(dist, key=dist.get)
        full = torch.softmax(logits, 0); mass = float(sum(full[j] for k in keys for j in label_ids[letter[k]]))
        per[it["id"]] = {"gold": it["gold"], "logprob": dist, "cand_mass": round(mass, 3)}
        rows.append({"conf": dist[top], "hit": top == it["gold"]})
        if i % 25 == 0:
            print(f"  {i}/{len(items)} {time.time()-t0:.0f}s  mean fwd {sum(ms)/len(ms):.0f} ms  running acc {sum(r['hit'] for r in rows)/len(rows):.3f}  cand mass {sum(v['cand_mass'] for v in per.values())/len(per):.2f}", flush=True)
    rep = metrics(rows); rep.update({"model": a.model, "mean_forward_ms": round(sum(ms) / len(ms), 1),
                                     "mean_cand_mass": round(sum(v["cand_mass"] for v in per.values()) / len(per), 3)})
    print("logprob", rep, flush=True)
    Path(a.out).write_text(json.dumps({"report": rep, "items": per}, indent=1)); print("saved", a.out)


if __name__ == "__main__":
    main()
