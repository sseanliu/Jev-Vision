#!/usr/bin/env bash
# Run 3c: backbone-size test. Same data as run 3b (hard 123k + 8k entropy-selected Jev
# + 9.8k synthetic Jev-labelled), but Qwen3-8B-Base with LoRA r=64 on all linear
# projections (full FT of 8B does not fit one 80 GB card with AdamW). Heads fully trained.
# Question: does ANLI ECE (0.35 at 1.7B in runs 1/3a/3b) drop once the backbone is larger?
set -euo pipefail
cd /workspace/jev-probes/model
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false PIP_BREAK_SYSTEM_PACKAGES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/workspace/hf
python -m pip install -q peft datasets
RUN_NAME=${RUN_NAME:-s1-8b-run3c}
EXTRA="data/jsonl_jev_hard data/jsonl_synth_jev"
for d in $EXTRA; do for f in data/jsonl/*.validation.jsonl; do ln -sf "$PWD/$f" "$d/"; done; done
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h /workspace | tail -1
echo "=== $RUN_NAME: Qwen3-8B-Base LoRA r64, hard + $EXTRA $(date -u) ==="
python train.py --base Qwen/Qwen3-8B-Base --attn sdpa \
  --lora-r 64 --lora-alpha 128 --lora-targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --data data/jsonl --extra-data $EXTRA --out "runs/$RUN_NAME" \
  --epochs 1 --bsz 8 --grad-accum 2 --lr 1e-4 --head-lr 1e-3 --warmup 100 --max-tokens 2048 --eval-every 1000 --grad-ckpt --keep 1
echo "=== eval $(date -u) ==="
python eval.py "runs/$RUN_NAME/final" --sets mmlu mmlu_pro anli --limit 500 --temps 1,1.5,2,3,4 2>&1 | grep -E "^(mmlu|anli)"
python fingerprint.py "runs/$RUN_NAME/final" --out "runs/$RUN_NAME/final/fingerprint.json" 2>&1 | grep -v -E "Warning|warn|^\s*$" | tail -45
echo "=== run3c done $(date -u) ==="
