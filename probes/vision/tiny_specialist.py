"""CUA-S1-style tiny specialist as a baseline on our Mind2Web `ground` task.

Same recipe family as cua-ai/cua-s1-forms: byte-level embeddings, a 2-layer Transformer
encoder over the context and (separately) over each option's text, an attention readout
(option query attends to context tokens), one logit per option, softmax over the live
option count. ~0.7-1M parameters, trains on a laptop CPU in minutes. It sees only text
(task, history, element descriptions) — no screenshot — so it measures how much of our
task is solvable without vision.

python tiny_specialist.py --train /workspace/m2w_train_j/items.jsonl --test /workspace/m2w_items_j/items.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time

import torch
from torch import nn

CTX_BYTES, OPT_BYTES, D, HEADS, LAYERS = 224, 96, 128, 4, 2


def state_text(it):
    hist = " | ".join(it["history"]) or "(none)"
    return f"TASK {it['task']}\nHISTORY {hist}"


def enc(s: str, n: int) -> list[int]:
    b = s.encode("utf-8")[:n]
    return list(b) + [0] * (n - len(b))


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(256, D)
        self.pos_c = nn.Parameter(torch.zeros(CTX_BYTES, D)); self.pos_o = nn.Parameter(torch.zeros(OPT_BYTES, D))
        layer = lambda: nn.TransformerEncoder(nn.TransformerEncoderLayer(D, HEADS, 4 * D, batch_first=True, dropout=0.1), LAYERS)
        self.ctx_enc, self.opt_enc = layer(), layer()
        self.q = nn.Linear(D, D); self.k = nn.Linear(D, D); self.v = nn.Linear(D, D); self.out = nn.Linear(D, 1)

    def forward(self, ctx, opts, opt_mask):
        # ctx [B, C]; opts [B, K, O]; opt_mask [B, K] (1 = live option)
        B, K, O = opts.shape
        h_c = self.ctx_enc(self.emb(ctx) + self.pos_c)                       # [B, C, D]
        h_o = self.opt_enc(self.emb(opts.view(B * K, O)) + self.pos_o).mean(1).view(B, K, D)
        q = self.q(h_o); k = self.k(h_c); v = self.v(h_c)
        att = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(D), -1)        # [B, K, C]
        pooled = att @ v                                                      # [B, K, D]
        logit = self.out(torch.tanh(pooled * h_o)).squeeze(-1)               # [B, K]
        return logit.masked_fill(opt_mask == 0, -1e4)


def batchify(items, idx, device):
    K = max(len(items[i]["criteria"]) for i in idx)
    ctx = torch.tensor([enc(state_text(items[i]), CTX_BYTES) for i in idx], device=device)
    opts = torch.zeros((len(idx), K, OPT_BYTES), dtype=torch.long, device=device)
    mask = torch.zeros((len(idx), K), device=device); gold = torch.zeros(len(idx), dtype=torch.long, device=device)
    for b, i in enumerate(idx):
        keys = list(items[i]["criteria"])
        for j, kk in enumerate(keys):
            opts[b, j] = torch.tensor(enc(f"{kk}: {items[i]['criteria'][kk]}", OPT_BYTES)); mask[b, j] = 1
        gold[b] = keys.index(items[i]["gold"])
    return ctx, opts, mask, gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--bsz", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = random.Random(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = [json.loads(l) for l in open(a.train)]; test = [json.loads(l) for l in open(a.test)]
    model = Tiny().to(device)
    print("params:", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(train) / a.bsz)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
    t0 = time.time()
    for ep in range(a.epochs):
        model.train(); order = list(range(len(train))); rng.shuffle(order); tot = 0.0
        for s in range(0, len(order), a.bsz):
            ctx, opts, mask, gold = batchify(train, order[s:s + a.bsz], device)
            loss = nn.functional.cross_entropy(model(ctx, opts, mask), gold)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); tot += loss.item()
        model.eval(); confs, hits = [], []
        with torch.no_grad():
            for s in range(0, len(test), 128):
                ctx, opts, mask, gold = batchify(test, list(range(s, min(len(test), s + 128))), device)
                p = torch.softmax(model(ctx, opts, mask), -1)
                confs += p.max(-1).values.tolist(); hits += (p.argmax(-1) == gold).tolist()
        pos = [c for c, h in zip(confs, hits) if h]; neg = [c for c, h in zip(confs, hits) if not h]
        au = sum((x > y) + 0.5 * (x == y) for x in pos for y in neg) / (len(pos) * len(neg)) if pos and neg else float("nan")
        print(f"epoch {ep+1}: train loss {tot / max(1, len(order) // a.bsz):.3f}  test acc {sum(hits)/len(hits):.3f}  top-p {sum(confs)/len(confs):.3f}  AUROC {au:.3f}  {time.time()-t0:.0f}s", flush=True)
    chance = sum(1 / len(it["criteria"]) for it in test) / len(test)
    print(f"chance {chance:.3f}")


if __name__ == "__main__":
    main()
