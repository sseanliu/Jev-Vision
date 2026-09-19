#!/usr/bin/env bash
# V1: student ground model on Mind2Web.
#   1. build ~8k training items from the Mind2Web train split (10% none-gold), set-of-mark crops
#   2. Qwen3-VL-32B logprob teacher -> soft targets (T=4), uncertain subset (top-p < 0.8) as extra rows
#   3. train Qwen3-VL-8B-Instruct + s1 heads, LoRA r64 on the LM, hard rows + soft rows
#   4. eval on the V0 300 test_domain items next to the three teachers
# Everything under /workspace (network volume). Log: /workspace/v1.out
set -euo pipefail
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false PIP_BREAK_SYSTEM_PACKAGES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/workspace/hf
python -m pip install -q transformers datasets accelerate peft pillow duckdb 2>&1 | grep -v WARNING || true
python -c "import torchvision" 2>/dev/null || python -m pip install -q torchvision 2>&1 | grep -v WARNING || true
command -v hf >/dev/null || (curl -LsSf https://hf.co/cli/install.sh | bash -s >/dev/null 2>&1; export PATH="$HOME/.local/bin:$PATH")
export PATH="$HOME/.local/bin:$PATH"
RUN_NAME=${RUN_NAME:-v1-8b-m2w}
N_TRAIN=${N_TRAIN:-8000}
cd /workspace/jev-probes

if [ ! -f /workspace/m2w_train/requests.jsonl ]; then
  echo "=== build train items $(date -u) ==="
  (cd probes/vision && python build_m2w_items.py --split train --files 27 --n $N_TRAIN --none-frac 0.1 --out /workspace/m2w_train)
fi
# validation = the V0 300 items, with absolute image paths so they resolve from any directory
python - <<'PY'
import json
rows = [json.loads(l) for l in open("/workspace/m2w_items/requests.jsonl")]
with open("/workspace/m2w_train/v0.validation.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps({**r, "image": "/workspace/m2w_items/" + r["image"]}) + "\n")
print("validation rows", len(rows))
PY
if [ ! -f /workspace/m2w_train_soft/soft.train.jsonl ]; then
  echo "=== teacher soft labels $(date -u) ==="
  mkdir -p /workspace/m2w_train_soft
  python probes/vision/teacher_qwen_vl.py --model Qwen/Qwen3-VL-32B-Instruct --items /workspace/m2w_train --limit 100000 \
    --out /workspace/teacher_qwen3vl32b_train.json --temperature 4 --soft-max-top 0.8 \
    --requests-out /workspace/m2w_train_soft/soft.train.jsonl
fi
# the teacher reads items.jsonl + requests.jsonl; the trainer globs *.train.jsonl
cp -n /workspace/m2w_train/requests.jsonl /workspace/m2w_train/m2w.train.jsonl
wc -l /workspace/m2w_train/*.jsonl /workspace/m2w_train_soft/*.jsonl

echo "=== train $RUN_NAME $(date -u) ==="
cd model
python train_vl.py --base Qwen/Qwen3-VL-8B-Instruct --data /workspace/m2w_train --extra-data /workspace/m2w_train_soft \
  --out "runs/$RUN_NAME" --epochs 1 --bsz 4 --grad-accum 4 --lr 1e-4 --head-lr 1e-3 --warmup 50 \
  --eval-every 100 --val-limit 300 --max-pixels 1288000 --lora-r 64 --grad-ckpt --keep 1
echo "=== eval $(date -u) ==="
python eval_vl.py "runs/$RUN_NAME/final" --data /workspace/m2w_train/v0.validation.jsonl \
  --teachers /workspace/teacher_qwen3vl32b_m2w300.json --out "runs/$RUN_NAME/final/eval_v0.json"
echo "=== v1 done $(date -u) ==="
