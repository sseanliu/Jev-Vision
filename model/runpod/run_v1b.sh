#!/usr/bin/env bash
# V1b: same as run_v1.sh but with --jitter crops (random viewport offset, distractors from a wider pool),
# which removes the centred-target cue a trained student could exploit. Builds a fresh 300-item
# test_domain eval set (V0b) with the same jitter, re-runs the Qwen teacher on it, then trains and evals.
set -euo pipefail
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false PIP_BREAK_SYSTEM_PACKAGES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/workspace/hf
export PATH="$HOME/.local/bin:$PATH"
RUN_NAME=${RUN_NAME:-v1b-8b-m2w-jitter}
N_TRAIN=${N_TRAIN:-8000}
EVAL=/workspace/m2w_items_j
TRAIN=/workspace/m2w_train_j
SOFT=/workspace/m2w_train_j_soft
cd /workspace/jev-probes

if [ ! -f $EVAL/requests.jsonl ]; then
  echo "=== build V0b eval items (jitter) $(date -u) ==="
  (cd probes/vision && python build_m2w_items.py --split test_domain --files 3 --n 300 --jitter --seed 1 --out $EVAL)
fi
if [ ! -f /workspace/teacher_qwen3vl32b_m2w300_j.json ]; then
  echo "=== teacher on V0b $(date -u) ==="
  python probes/vision/teacher_qwen_vl.py --model Qwen/Qwen3-VL-32B-Instruct --items $EVAL --limit 300 --out /workspace/teacher_qwen3vl32b_m2w300_j.json
fi
if [ ! -f $TRAIN/requests.jsonl ]; then
  echo "=== build train items (jitter) $(date -u) ==="
  (cd probes/vision && python build_m2w_items.py --split train --files 27 --n $N_TRAIN --none-frac 0.1 --jitter --seed 2 --out $TRAIN)
fi
python - <<PY
import json
rows = [json.loads(l) for l in open("$EVAL/requests.jsonl")]
with open("$TRAIN/v0b.validation.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps({**r, "image": "$EVAL/" + r["image"]}) + "\n")
print("validation rows", len(rows))
PY
if [ ! -f $SOFT/soft.train.jsonl ]; then
  echo "=== teacher soft labels (jitter) $(date -u) ==="
  mkdir -p $SOFT
  python probes/vision/teacher_qwen_vl.py --model Qwen/Qwen3-VL-32B-Instruct --items $TRAIN --limit 100000 \
    --out /workspace/teacher_qwen3vl32b_train_j.json --temperature 4 --soft-max-top 0.8 --requests-out $SOFT/soft.train.jsonl
fi
cp -n $TRAIN/requests.jsonl $TRAIN/m2w.train.jsonl
wc -l $TRAIN/*.jsonl $SOFT/*.jsonl

echo "=== train $RUN_NAME $(date -u) ==="
cd model
python train_vl.py --base Qwen/Qwen3-VL-8B-Instruct --data $TRAIN --extra-data $SOFT \
  --out "runs/$RUN_NAME" --epochs 1 --bsz 4 --grad-accum 4 --lr 1e-4 --head-lr 1e-3 --warmup 50 \
  --eval-every 100 --val-limit 300 --max-pixels 1288000 --lora-r 64 --grad-ckpt --keep 1
echo "=== eval $(date -u) ==="
python eval_vl.py "runs/$RUN_NAME/final" --data $TRAIN/v0b.validation.jsonl --temps 0.5,0.75,1,1.5,2 \
  --teachers /workspace/teacher_qwen3vl32b_m2w300_j.json --out "runs/$RUN_NAME/final/eval_v0b.json"
# the V1 (centred) student on the jittered eval set: how much of V1's score was the cue?
python eval_vl.py runs/v1-8b-m2w/final --data $TRAIN/v0b.validation.jsonl --temps 0.5,0.75,1 --out runs/v1-8b-m2w/final/eval_v0b.json
echo "=== v1b done $(date -u) ==="
