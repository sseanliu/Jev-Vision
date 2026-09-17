#!/usr/bin/env bash
# Flagship: Qwen3-30B-A3B-Base + LoRA on attention projections, readout heads
# fully trained, experts frozen. Intended for one H200 (141 GB): bf16 weights
# ~61 GB + LoRA/optimizer + 2048-token activations with grad checkpointing.
#
# Env knobs:
#   SOFT_DIRS   extra data dirs with soft labels, space separated
#               (default: data/jsonl_jev; set to data/jsonl_open or both after run 2)
#   RUN_NAME    default s1-30b-a3b
#   EPOCHS      default 1
set -euo pipefail
cd /workspace/jev-probes/model
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false PIP_BREAK_SYSTEM_PACKAGES=1
python -m pip install -q peft
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /workspace | tail -1

RUN_NAME=${RUN_NAME:-s1-30b-a3b}
SOFT_DIRS=${SOFT_DIRS:-data/jsonl_jev}
EPOCHS=${EPOCHS:-1}

if [ ! -f data/jsonl/mnli.train.jsonl ]; then
  python data/convert.py --limit-train 20000 --limit-val 1000
fi
for d in $SOFT_DIRS; do
  for f in data/jsonl/*.validation.jsonl; do ln -sf "$PWD/$f" "$d/"; done
done

echo "=== flagship $RUN_NAME on Qwen3-30B-A3B-Base $(date -u) ==="
python train.py \
  --base Qwen/Qwen3-30B-A3B-Base --attn sdpa \
  --lora-r 32 --lora-alpha 64 --lora-targets q_proj,k_proj,v_proj,o_proj \
  --data data/jsonl --extra-data $SOFT_DIRS \
  --out "runs/$RUN_NAME" --epochs "$EPOCHS" \
  --bsz 4 --grad-accum 4 --lr 1e-4 --head-lr 1e-3 --warmup 100 \
  --max-tokens 2048 --eval-every 300 --grad-ckpt --keep 1

echo "=== eval $(date -u) ==="
python eval.py "runs/$RUN_NAME/final" --sets mmlu mmlu_pro anli --limit 500 2>&1 | grep -E "^(mmlu|anli)"
python fingerprint.py "runs/$RUN_NAME/final" --out "runs/$RUN_NAME/final/fingerprint.json" 2>&1 | grep -v -E "Warning|warn|^\s*$" | tail -45
echo "=== flagship done $(date -u) ==="
