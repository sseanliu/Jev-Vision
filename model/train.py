"""Train the decision model.

python train.py --base Qwen/Qwen3-1.7B-Base --data data/jsonl --out runs/s1-1.7b \
    --epochs 1 --bsz 4 --lr 2e-5 --max-tokens 2048

Loss: cross-entropy over option logits (choice/score), BCE (noul). Both are
proper scoring rules, so the objective is the calibration objective.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from s1.dataset import RequestDataset, collate
from s1.model import DecisionModel
from s1.packing import Packer


class Collate:
    """Picklable collate for DataLoader workers."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, examples):
        return collate(examples, self.pad_id)


def loss_for(model: DecisionModel, ids, seg, pos, examples, device):
    h = model.hidden(ids.to(device), seg.to(device), pos.to(device))
    total, n = 0.0, 0
    for b, ex in enumerate(examples):
        logits = model.readout(h[b], ex.packed)
        for z, t, qtype in zip(logits, ex.targets, ex.packed.qtypes):
            if qtype == "noul":
                total = total + F.binary_cross_entropy_with_logits(z.float(), torch.tensor(t, device=device))
            else:
                total = total + F.cross_entropy(z.float()[None], torch.tensor([t], device=device))
            n += 1
    return total / max(n, 1), n


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    model.eval()
    tot, n, correct, nq = 0.0, 0, 0, 0
    confs, hits = [], []
    for i, (ids, seg, pos, examples) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        h = model.hidden(ids.to(device), seg.to(device), pos.to(device))
        for b, ex in enumerate(examples):
            logits = model.readout(h[b], ex.packed)
            for z, t, qtype in zip(logits, ex.targets, ex.packed.qtypes):
                if qtype == "noul":
                    p = torch.sigmoid(z.float()).item()
                    pred = p >= 0.5
                    hit = pred == (t >= 0.5)
                    confs.append(max(p, 1 - p))
                    tot += F.binary_cross_entropy_with_logits(z.float(), torch.tensor(float(t), device=device)).item()
                else:
                    pr = torch.softmax(z.float(), -1)
                    hit = int(pr.argmax()) == t
                    confs.append(pr.max().item())
                    tot += F.cross_entropy(z.float()[None], torch.tensor([t], device=device)).item()
                hits.append(hit)
                correct += hit
                nq += 1
                n += 1
    model.train()
    ece = expected_calibration_error(confs, hits)
    return {"loss": tot / max(n, 1), "acc": correct / max(nq, 1), "ece": ece, "n": nq}


def expected_calibration_error(confs, hits, bins=10):
    if not confs:
        return float("nan")
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confs) if (lo <= c < hi) or (b == bins - 1 and c == 1.0)]
        if not idx:
            continue
        acc = sum(hits[i] for i in idx) / len(idx)
        conf = sum(confs[i] for i in idx) / len(idx)
        ece += len(idx) / len(confs) * abs(acc - conf)
    return ece


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--data", default="data/jsonl")
    ap.add_argument("--out", default="runs/s1")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="cap train rows (smoke tests)")
    ap.add_argument("--tiny", action="store_true", help="random tiny backbone for CPU smoke tests")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.base)
    packer = Packer(tok)
    if a.tiny:
        from transformers import Qwen3Config
        cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096)
        model = DecisionModel(cfg, rank=32, attn_implementation="eager")
    else:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        model = DecisionModel(a.base, rank=256, new_vocab_size=len(tok), attn_implementation=a.attn, torch_dtype=dtype)
        if a.grad_ckpt:
            model.backbone.gradient_checkpointing_enable()
    model.to(device)

    data = Path(a.data)
    train_paths = sorted(data.glob("*.train.jsonl"))
    val_paths = sorted(data.glob("*.validation.jsonl"))
    train_ds = RequestDataset(train_paths, packer, a.max_tokens, augment=True, seed=a.seed)
    if a.limit:
        train_ds.rows = random.Random(a.seed).sample(train_ds.rows, min(a.limit, len(train_ds.rows)))
    val_ds = RequestDataset(val_paths, packer, a.max_tokens, augment=False, seed=1)
    val_ds.rows = random.Random(1).sample(val_ds.rows, min(400, len(val_ds.rows)))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    col = Collate(pad)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=a.bsz, shuffle=True, collate_fn=col, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=a.bsz, shuffle=False, collate_fn=col)
    print(f"train rows={len(train_ds)} val rows={len(val_ds)} device={device}")

    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    bb_params = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": bb_params, "lr": a.lr}, {"params": head_params, "lr": a.head_lr}],
                            weight_decay=0.01, betas=(0.9, 0.95))
    steps_per_epoch = math.ceil(len(train_loader) / a.grad_accum)
    total_steps = a.max_steps or int(steps_per_epoch * a.epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "log.jsonl").open("a")

    model.train()
    step, micro, t0 = 0, 0, time.time()
    running = 0.0
    done = False
    while not done:
        for ids, seg, pos, examples in train_loader:
            loss, _ = loss_for(model, ids, seg, pos, examples, device)
            (loss / a.grad_accum).backward()
            running += loss.item()
            micro += 1
            if micro % a.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    rec = {"step": step, "loss": running / (10 * a.grad_accum), "lr": sched.get_last_lr()[0],
                           "elapsed": round(time.time() - t0)}
                    print(json.dumps(rec), flush=True)
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                    running = 0.0
                if step % a.eval_every == 0 or step >= total_steps:
                    ev = evaluate(model, val_loader, device)
                    rec = {"step": step, "eval": ev}
                    print(json.dumps(rec), flush=True)
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                    save(model, tok, out / f"step{step}")
                if step >= total_steps:
                    done = True
                    break
    save(model, tok, out / "final")
    print("done", out / "final")


def save(model: DecisionModel, tok, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(path / "backbone")
    tok.save_pretrained(path / "backbone")
    torch.save({k: v.cpu() for k, v in model.state_dict().items() if not k.startswith("backbone.")},
               path / "heads.pt")
    (path / "s1_config.json").write_text(json.dumps({"rank": model.rank}))


if __name__ == "__main__":
    main()
