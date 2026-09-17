"""Run the Jev behavioural fingerprints against a local checkpoint so the
results are directly comparable with results/k_sweep_*_trials.json and
results/capability_jev-1.13.0.json.

python fingerprint.py runs/s1-1.7b/final --out results/fingerprint_s1-1.7b.json

Sections
- k_sweep (unknown + concrete variants): pairwise log-odds among the original
  options as absurd options are appended. Jev: pairs involving "Cannot tell"
  move ~2 nats and saturate by K~12-16; concrete pairs stable.
- isolation: a question's distribution is unchanged by siblings (by
  construction here; reported for completeness).
- capability: Bob two-games puzzle (2 phrasings x 1, deterministic), two-step
  word problems, modular exponentiation, 3-digit products, same generator
  and seed as probes/capability.py.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "probes"))
from eval import load  # noqa: E402
from s1.packing import Question  # noqa: E402

import capability as cap  # noqa: E402  (probes/capability.py)
import k_sweep as ks  # noqa: E402      (probes/k_sweep.py)


@torch.no_grad()
def choice_probs(model, packer, state, instructions, criteria, device):
    q = Question("q", "choice", instructions, criteria)
    z = model(packer.pack(state, [q]), device)[0]
    p = torch.softmax(z.float(), -1).tolist()
    return dict(zip(criteria, p))


def run_k_sweep(model, packer, device, variant):
    original = dict(ks.ORIGINAL)
    if variant == "concrete":
        original.pop("unknown")
        original["hacker"] = "A hacker caused it"
    pairs = list(itertools.combinations(original, 2))
    rows = {}
    for k in ks.KS:
        for content, pool in (("A", ks.POOL_A), ("B", ks.POOL_B)):
            if k == 4 and content == "B":
                continue
            crit = dict(original)
            for key, desc in pool[: k - 4]:
                crit[key] = desc
            p = choice_probs(model, packer, ks.STATE, ks.INSTRUCTIONS, crit, device)
            lo = {f"{a}/{b}": math.log(max(p[a], 0.005) / max(p[b], 0.005)) for a, b in pairs}
            lo["irrelevant_mass"] = sum(v for kk, v in p.items() if kk not in original)
            rows[f"{k}{content}"] = lo
    base = rows["4A"]
    print(f"\nK-sweep ({variant}): log-odds among original options")
    print("K    " + " ".join(f"{a[:4]}/{b[:4]:<5}" for a, b in pairs) + "  irrelevant")
    for key, lo in rows.items():
        print(f"{key:<4} " + " ".join(f"{lo[f'{a}/{b}']:+.2f}     " for a, b in pairs)
              + f"  {lo['irrelevant_mass']:.3f}")
    print("Change vs K=4 for pairs involving the 4th option:")
    fourth = list(original)[3]
    for key, lo in rows.items():
        deltas = [lo[f"{a}/{b}"] - base[f"{a}/{b}"] for a, b in pairs if fourth in (a, b)]
        print(f"  {key:<4} mean delta {sum(deltas)/len(deltas):+.2f}")
    return rows


def run_capability(model, packer, device):
    rng = random.Random(cap.SEED)
    results = []
    for text, crit, correct in cap.BOB:
        p = choice_probs(model, packer, text, "Which option should be chosen?", crit, device)
        top = max(p, key=lambda kk: p[kk])
        results.append({"family": "bob", "is_correct": top == correct, "top_prob": p[top], "p_correct": p[correct]})
    for fam, gen, spread in (("word_2step", cap.gen_word_2step, 40), ("modexp", cap.gen_modexp, 6),
                             ("mult3x3", cap.gen_mult, 900)):
        for text, ans in gen(rng):
            payload, correct = cap.numeric_question(rng, text, ans, spread)
            crit = payload["questions"]["a"]["criteria"]
            p = choice_probs(model, packer, text, payload["questions"]["a"]["instructions"], crit, device)
            top = max(p, key=lambda kk: p[kk])
            results.append({"family": fam, "is_correct": top == correct, "top_prob": p[top], "p_correct": p[correct]})
    print("\nCapability")
    print(f"{'family':12s} {'n':>4s} {'accuracy':>9s} {'mean top-p':>11s} {'mean p(correct)':>16s}")
    for fam in ("bob", "word_2step", "modexp", "mult3x3"):
        rs = [r for r in results if r["family"] == fam]
        print(f"{fam:12s} {len(rs):4d} {sum(r['is_correct'] for r in rs)/len(rs):9.2f} "
              f"{sum(r['top_prob'] for r in rs)/len(rs):11.2f} {sum(r['p_correct'] for r in rs)/len(rs):16.2f}")
    return results


@torch.no_grad()
def run_isolation(model, packer, device):
    q1 = Question("who", "choice", ks.INSTRUCTIONS, ks.ORIGINAL)
    q2 = Question("urgent", "noul", "Does this message require urgent attention?")
    alone = torch.softmax(model(packer.pack(ks.STATE, [q1]), device)[0].float(), -1)
    with_sib = torch.softmax(model(packer.pack(ks.STATE, [q2, q1]), device)[1].float(), -1)
    d = (alone - with_sib).abs().max().item()
    print(f"\nIsolation: max |dp| with a sibling question = {d:.2e}")
    return {"max_abs_dp": d}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, packer = load(Path(a.checkpoint), device)
    out = {"checkpoint": a.checkpoint,
           "k_sweep_unknown": run_k_sweep(model, packer, device, "unknown"),
           "k_sweep_concrete": run_k_sweep(model, packer, device, "concrete"),
           "isolation": run_isolation(model, packer, device),
           "capability": run_capability(model, packer, device)}
    path = Path(a.out or (Path(a.checkpoint) / "fingerprint.json"))
    path.write_text(json.dumps(out, indent=1))
    print("saved", path)


if __name__ == "__main__":
    main()
