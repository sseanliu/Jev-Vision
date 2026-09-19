"""Train the vision decision model (Qwen3-VL backbone + s1 heads, LoRA on the LM).

python train_vl.py --base Qwen/Qwen3-VL-8B-Instruct --data data/vision/m2w_train --out runs/v1-8b \
    --lora-r 64 --epochs 1 --bsz 4 --grad-accum 4 --lr 1e-4 --head-lr 1e-3 --max-pixels 1288000
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
from transformers import AutoProcessor

from s1.dataset_vl import VLRequestDataset
from s1.packing import SPECIAL_TOKENS
from s1.vl import VLDecisionModel, VLPacker, collate_vl
from train import expected_calibration_error, prune_checkpoints


class CollateVL:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, examples):
        return collate_vl(examples, self.pad_id)


def batch_hidden(model, batch, device):
    ids, seg, pos, pix, grid, slens, mm, examples = batch
    h = model.hidden_vl(ids.to(device), seg.to(device), pos.to(device), pix.to(device, model.backbone.dtype),
                        grid.to(device), slens, mm)
    return h, examples


def loss_for(model, batch, device):
    h, examples = batch_hidden(model, batch, device)
    total, n = 0.0, 0
    for b, ex in enumerate(examples):
        for z, t, qtype in zip(model.readout(h[b], ex.packed), ex.targets, ex.packed.qtypes):
            if qtype == "noul":
                total = total + F.binary_cross_entropy_with_logits(z.float(), torch.tensor(t, device=device))
            elif isinstance(t, list):
                tt = torch.tensor(t, device=device); tt = tt / tt.sum().clamp_min(1e-9)
                total = total - (tt * F.log_softmax(z.float(), -1)).sum()
            else:
                total = total + F.cross_entropy(z.float()[None], torch.tensor([t], device=device))
            n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    confs, hits, nll = [], [], 0.0
    for batch in loader:
        h, examples = batch_hidden(model, batch, device)
        for b, ex in enumerate(examples):
            for z, t, qtype in zip(model.readout(h[b], ex.packed), ex.targets, ex.packed.qtypes):
                if qtype == "noul":
                    p = torch.sigmoid(z.float()).item(); hit = (p >= 0.5) == (t >= 0.5); confs.append(max(p, 1 - p))
                else:
                    pr = torch.softmax(z.float(), -1); ti = int(torch.tensor(t).argmax()) if isinstance(t, list) else t
                    hit = int(pr.argmax()) == ti; confs.append(pr.max().item()); nll += -math.log(max(pr[ti].item(), 1e-9))
                hits.append(hit)
    model.train()
    pos = [c for c, h_ in zip(confs, hits) if h_]; neg = [c for c, h_ in zip(confs, hits) if not h_]
    auroc = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg)) if pos and neg else float("nan")
    return {"n": len(hits), "acc": sum(hits) / max(1, len(hits)), "ece": expected_calibration_error(confs, hits),
            "auroc": auroc, "nll": nll / max(1, len(hits)), "mean_top_p": sum(confs) / max(1, len(confs))}


def save(model, proc, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(path / "backbone")
    proc.save_pretrained(path / "backbone")
    torch.save({k: v.cpu() for k, v in model.state_dict().items() if not k.startswith("backbone.")}, path / "heads.pt")
    (path / "s1_config.json").write_text(json.dumps({"rank": model.rank, "lora": model.lora, "vl": True}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--data", required=True, help="dir with *.train.jsonl and *.validation.jsonl (+ images)")
    ap.add_argument("--extra-data", nargs="*", default=[])
    ap.add_argument("--out", default="runs/v1")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--val-limit", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-pixels", type=int, default=1288 * 1000)
    ap.add_argument("--multires", default=None, help="comma-separated pixel budgets sampled per training example, e.g. 1288000,640000,320000")
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=0)
    ap.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--init-adapter", default=None, help="LoRA checkpoint dir (…/final/backbone) to continue from")
    ap.add_argument("--init-heads", default=None, help="heads.pt to continue from")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--keep", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    proc = AutoProcessor.from_pretrained(a.base)
    packer = VLPacker(proc, max_pixels=a.max_pixels)
    tok = proc.tokenizer
    new_ids = [tok.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS]
    lora = {"r": a.lora_r, "alpha": a.lora_alpha or 2 * a.lora_r, "targets": a.lora_targets.split(",")} if a.lora_r > 0 else None
    if a.init_adapter:
        from peft import PeftModel
        model = VLDecisionModel(a.base, rank=256, new_vocab_size=len(tok), attn_implementation=a.attn, torch_dtype=dtype)
        model.backbone = PeftModel.from_pretrained(model.backbone, a.init_adapter, is_trainable=True)
        model.lora = json.loads((Path(a.init_adapter).parent / "s1_config.json").read_text()).get("lora")
        print(f"loaded adapter from {a.init_adapter}")
    else:
        model = VLDecisionModel(a.base, rank=256, new_vocab_size=len(tok), attn_implementation=a.attn, torch_dtype=dtype,
                                lora=lora, trainable_token_ids=new_ids if lora else None)
    if model.lora:
        model.backbone.print_trainable_parameters()
    if a.init_heads:
        missing, unexpected = model.load_state_dict(torch.load(a.init_heads, map_location="cpu"), strict=False)
        assert not unexpected, unexpected
        print(f"loaded heads from {a.init_heads}")
    if a.grad_ckpt:
        model.backbone.gradient_checkpointing_enable()
    model.to(device)

    data = Path(a.data)
    train_paths = sorted(data.glob("*.train.jsonl")) + [p for d in a.extra_data for p in sorted(Path(d).glob("*.train.jsonl"))]
    val_paths = sorted(data.glob("*.validation.jsonl"))
    budgets = [int(x) for x in a.multires.split(",")] if a.multires else None
    train_ds = VLRequestDataset(train_paths, packer, seed=a.seed, limit=a.limit, budgets=budgets)
    val_ds = VLRequestDataset(val_paths, packer, seed=1, limit=a.val_limit)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    col = CollateVL(pad)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=a.bsz, shuffle=True, collate_fn=col, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=a.bsz, shuffle=False, collate_fn=col, num_workers=2)
    print(f"train rows={len(train_ds)} val rows={len(val_ds)} device={device}", flush=True)

    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    bb_params = [p for n, p in model.named_parameters() if n.startswith("backbone.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": bb_params, "lr": a.lr}, {"params": head_params, "lr": a.head_lr}],
                            weight_decay=0.01, betas=(0.9, 0.95))
    total_steps = int(math.ceil(len(train_loader) / a.grad_accum) * a.epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = (out / "log.jsonl").open("a")
    model.train()
    step, micro, running, t0, done = 0, 0, 0.0, time.time(), False
    while not done:
        for batch in train_loader:
            loss = loss_for(model, batch, device)
            (loss / a.grad_accum).backward()
            running += loss.item(); micro += 1
            if micro % a.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 10 == 0:
                    rec = {"step": step, "loss": running / (10 * a.grad_accum), "lr": sched.get_last_lr()[0], "elapsed": round(time.time() - t0)}
                    print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush(); running = 0.0
                if step % a.eval_every == 0 or step >= total_steps:
                    rec = {"step": step, "eval": evaluate(model, val_loader, device)}
                    print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
                    save(model, proc, out / f"step{step}"); prune_checkpoints(out, a.keep)
                if step >= total_steps:
                    done = True; break
    save(model, proc, out / "final")
    print("done", out / "final")


if __name__ == "__main__":
    main()
