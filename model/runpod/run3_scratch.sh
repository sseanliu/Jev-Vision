#!/usr/bin/env bash
# Run 3 (from scratch, replaces the lost run-1 checkpoint):
#   Qwen3-1.7B-Base on hard labels (123k) + entropy-selected Jev labels (8k)
#   + synthetic Jev-labelled workflows (if present). One epoch.
# Compare against run 1 (hard only): does an uncertainty-rich soft-label mix fix OOD calibration?
# Everything lives on the network volume (/workspace). Logs to /workspace/run3.out.
set -euo pipefail
cd /workspace/jev-probes/model
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false PIP_BREAK_SYSTEM_PACKAGES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/workspace/hf
RUN_NAME=${RUN_NAME:-s1-1.7b-run3}
EXTRA="data/jsonl_jev_hard"
[ -f data/jsonl_synth_jev/synth.train.jsonl ] && EXTRA="$EXTRA data/jsonl_synth_jev"
for d in $EXTRA; do for f in data/jsonl/*.validation.jsonl; do ln -sf "$PWD/$f" "$d/"; done; done
echo "=== $RUN_NAME: hard + $EXTRA $(date -u) ==="
python train.py --base Qwen/Qwen3-1.7B-Base --data data/jsonl --extra-data $EXTRA --out "runs/$RUN_NAME" \
  --epochs 1 --bsz 8 --grad-accum 2 --lr 2e-5 --head-lr 1e-3 --max-tokens 2048 --eval-every 500 --grad-ckpt --keep 1 --attn sdpa
echo "=== eval $(date -u) ==="
python eval.py "runs/$RUN_NAME/final" --sets mmlu mmlu_pro anli --limit 500 2>&1 | grep -E "^(mmlu|anli)"
python fingerprint.py "runs/$RUN_NAME/final" --out "runs/$RUN_NAME/final/fingerprint.json" 2>&1 | grep -v -E "Warning|warn|^\s*$" | tail -45
echo "=== run3 done $(date -u) ==="
