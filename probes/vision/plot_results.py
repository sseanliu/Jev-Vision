"""Figures for the release: (1) parameters vs grounding accuracy on V0b, (2) escalation curves.

python plot_results.py --out ../../results/vision/figures
Reads the result JSON/TXT files under results/vision; missing entries are skipped.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[2] / "results" / "vision"


def acc_from(path, channel):
    p = R / path
    if not p.exists():
        return None
    d = json.load(open(p))
    rep = d["report"]
    if "acc" in rep:
        return rep["acc"]
    return rep.get(channel, {}).get("acc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(R / "figures"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # --- 1. params vs accuracy (V0b, 300 jittered Mind2Web test_domain items) ---
    points = [  # (label, params, acc, trained_on_task, marker)
        ("tiny text-only specialist (0.7M)", 0.7e6, 0.397, True),
        ("Qwen3-VL-32B zero-shot", 32e9, acc_from("v1b-8b-m2w-jitter/teacher_qwen3vl32b_m2w300_j.json", "logprob"), False),
        ("DiffusionGemma 26B-A4B zero-shot", 26e9, acc_from("baselines/teacher_dgemma_m2w300_j.json", "logprob"), False),
        ("Gemini 3.8 Flash (thinking) zero-shot", None, acc_from("teacher_gemini-3.8-flash-high_m2w300_j.json", "verbal"), False),
        ("Claude Opus 5 (thinking) zero-shot", None, acc_from("teacher_claude-opus-5-thinking_m2w300_j.json", "verbal"), False),
        ("V1b student 8B (ours)", 8e9, 0.847, True),
    ]
    v2 = R / "v2-8b-schema" / "eval_v0b.json"
    if v2.exists():
        points.append(("V2 student 8B schema-mix (ours)", 8e9, json.load(open(v2))["report"]["by_temperature"]["0.5"]["acc"], True))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, params, acc, trained in points:
        if acc is None:
            continue
        x = params if params else 300e9  # API models: unknown size, plotted at the right edge
        ax.scatter([x], [acc], s=70, marker="o" if trained else "s", color="#1f4e79" if trained else "#8c8c8c", zorder=3)
        ax.annotate(label, (x, acc), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.axhline(0.10, color="#bbbbbb", lw=1, ls="--"); ax.annotate("chance (10 options)", (1e6, 0.11), fontsize=8, color="#888888")
    ax.set_xscale("log"); ax.set_xlim(3e5, 1e12); ax.set_ylim(0, 1)
    ax.set_xlabel("parameters (API models at right edge: size unknown)"); ax.set_ylabel("grounding accuracy, V0b (300 items)")
    ax.set_title("Trained on the task (circles) vs zero-shot (squares)")
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out / "params_vs_accuracy.png", dpi=160); print("saved", out / "params_vs_accuracy.png")

    # --- 2. escalation curves from the txt outputs of escalation.py ---
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for txt in sorted((R / "baselines").glob("escalation_*.txt")):
        xs, ys = [], []
        for line in open(txt):
            parts = line.split()
            if len(parts) == 5 and parts[0].replace(".", "").isdigit():
                xs.append(float(parts[1])); ys.append(float(parts[2]))
        if xs:
            ax.plot(xs, ys, marker="o", label=txt.stem.replace("escalation_", ""))
    ax.set_xlabel("fraction of decisions escalated to the teacher"); ax.set_ylabel("combined accuracy")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "escalation.png", dpi=160); print("saved", out / "escalation.png")

    # --- 3. latency vs accuracy (per decision; API latencies are wall-clock means from the teacher runs) ---
    lat = [  # (label, ms, acc, trained)
        ("tiny text-only specialist", 1, 0.397, True),
        ("V1b / V2 8B (ours, both 0.847)", 153, 0.847, True),
        ("DiffusionGemma 26B-A4B", 154, acc_from("baselines/teacher_dgemma_m2w300_j.json", "logprob"), False),
        ("Qwen3-VL-32B logprob", 600, acc_from("v1b-8b-m2w-jitter/teacher_qwen3vl32b_m2w300_j.json", "logprob"), False),
        ("Gemini 3.8 Flash thinking-high", 3000, acc_from("teacher_gemini-3.8-flash-high_m2w300_j.json", "verbal"), False),
        ("Claude Opus 5 thinking", 7800, acc_from("teacher_claude-opus-5-thinking_m2w300_j.json", "verbal"), False),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, ms, acc, trained in lat:
        if acc is None:
            continue
        ax.scatter(ms, acc, marker="o" if trained else "s", s=60, color="black" if trained else "gray")
        ax.annotate(label, (ms, acc), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("latency per decision (ms, log)"); ax.set_ylabel("grounding accuracy, V0b (300 items)")
    ax.set_title("Accuracy vs latency: trained 8B (circles) vs zero-shot (squares)"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out / "latency_vs_accuracy.png", dpi=160); print("saved", out / "latency_vs_accuracy.png")

    # --- 4. desktop transfer: V2 zero-shot at step 0, then V3 desktop stage-2 evals by step ---
    v3log = R / "v3-8b-desktop" / "log.jsonl"; zs = R / "v3-8b-desktop" / "eval_desktop_zeroshot.json"
    if v3log.exists() and zs.exists():
        z = json.load(open(zs))["report"]["by_temperature"]["0.5"]
        steps, accs, aurocs = [0], [z["acc"]], [z["auroc"]]
        for line in open(v3log):
            r = json.loads(line)
            if "eval" in r:
                steps.append(r["step"]); accs.append(r["eval"]["acc"]); aurocs.append(r["eval"]["auroc"])
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        ax.plot(steps, accs, marker="o", color="black", label="accuracy")
        ax.plot(steps, aurocs, marker="s", color="gray", label="AUROC (confidence vs correctness)")
        ax.annotate("V2, never saw a desktop", (0, accs[0]), textcoords="offset points", xytext=(6, -12), fontsize=8)
        ax.set_xlabel("desktop stage-2 steps (bsz 16; 3k desktop rows + 4k web replay)"); ax.set_ylabel("macOS grounding, OS-Atlas validation (360)")
        ax.set_ylim(0.78, 1.0); ax.grid(alpha=0.25); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / "desktop_transfer.png", dpi=160); print("saved", out / "desktop_transfer.png")


if __name__ == "__main__":
    main()
