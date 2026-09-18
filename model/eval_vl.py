"""Evaluate a vision decision checkpoint on request rows; report acc / ECE / AUROC / NLL and
per-decision latency, optionally next to teacher result files on the same items.

python eval_vl.py runs/v1-8b/final --data /workspace/m2w_items/requests.jsonl --teachers /workspace/teacher_qwen3vl32b_m2w300.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoProcessor

from s1.dataset_vl import VLRequestDataset
from s1.vl import VLDecisionModel, VLPacker
from train import expected_calibration_error


def load(path: Path, device, max_pixels: int, attn="sdpa"):
    cfg = json.loads((path / "s1_config.json").read_text())
    proc = AutoProcessor.from_pretrained(path / "backbone")
    tok = proc.tokenizer
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    adapter = path / "backbone" / "adapter_config.json"
    if adapter.exists():
        from peft import PeftModel
        base = json.loads(adapter.read_text())["base_model_name_or_path"]
        model = VLDecisionModel(base, rank=cfg["rank"], new_vocab_size=len(tok), attn_implementation=attn, torch_dtype=dtype)
        model.backbone = PeftModel.from_pretrained(model.backbone, str(path / "backbone"))
    else:
        model = VLDecisionModel(str(path / "backbone"), rank=cfg["rank"], attn_implementation=attn, torch_dtype=dtype)
    model.load_state_dict(torch.load(path / "heads.pt", map_location="cpu"), strict=False)
    return model.to(device).eval(), VLPacker(proc, max_pixels=max_pixels)


def auroc(confs, hits):
    pos = [c for c, h in zip(confs, hits) if h]; neg = [c for c, h in zip(confs, hits) if not h]
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg)) if pos and neg else float("nan")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--data", required=True, help="requests.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-pixels", type=int, default=1288 * 1000)
    ap.add_argument("--temps", default="1,1.5,2,3,4")
    ap.add_argument("--teachers", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, packer = load(Path(a.checkpoint), device, a.max_pixels)
    ds = VLRequestDataset([Path(a.data)], packer, limit=a.limit)
    temps = [float(t) for t in a.temps.split(",")]
    per, ms = {}, []
    for i in range(len(ds)):
        ex = ds.build(ds.rows[i])
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        z = model(ex.packed, device)[0].float()
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1000)
        keys = ex.packed.option_keys[0]
        gold = ex.targets[0]
        per[ds.rows[i].get("id", str(i))] = {"logits": z.tolist(), "keys": keys, "gold": keys[gold] if isinstance(gold, int) else gold}
    rep = {"n": len(per), "mean_ms": round(sum(ms) / len(ms), 1), "p50_ms": round(sorted(ms)[len(ms) // 2], 1), "by_temperature": {}}
    for T in temps:
        confs, hits, nll = [], [], 0.0
        for r in per.values():
            p = torch.softmax(torch.tensor(r["logits"]) / T, -1); gi = r["keys"].index(r["gold"])
            confs.append(p.max().item()); hits.append(int(p.argmax()) == gi); nll += -math.log(max(p[gi].item(), 1e-9))
        rep["by_temperature"][str(T)] = {"acc": round(sum(hits) / len(hits), 3), "ece": round(expected_calibration_error(confs, hits), 3),
                                         "auroc": round(auroc(confs, hits), 3), "nll": round(nll / len(hits), 3), "mean_top_p": round(sum(confs) / len(confs), 3)}
    print(json.dumps(rep, indent=1))
    for tpath in a.teachers:
        t = json.load(open(tpath))
        items = t["items"]; ch = "logprob" if "logprob" in next(iter(items.values())) else "verbal"
        shared = [k for k in per if k in items]
        agree = sum(max(items[k][ch], key=items[k][ch].get) == per[k]["keys"][int(torch.tensor(per[k]["logits"]).argmax())] for k in shared)
        print(f"teacher {Path(tpath).name}: {t['report'].get('acc', t['report'].get(ch, {}).get('acc'))} acc on its run; "
              f"student argmax agreement on {len(shared)} shared items: {agree / max(1, len(shared)):.3f}")
    out = Path(a.out or (Path(a.checkpoint) / "eval_vl.json"))
    out.write_text(json.dumps({"report": rep, "items": per}, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
