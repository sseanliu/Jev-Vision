#!/usr/bin/env bash
# V2: schema flexibility. Same jittered Mind2Web data as V1b, but training rows carry 2-3 randomly
# sampled typed questions (ground + act/final/progress/needs_text/tag/history_len) with randomised
# phrasing and option sets, so the student learns to fill in whatever schema the caller sends.
#   0. rebuild items with the new fields (same seeds -> same screenshots, so V1b's teacher labels still apply)
#   1. zero-shot probe: V1b (ground-only) on the V0b schema rows  -> how much flexibility the backbone kept
#   2. stage-2 from V1b on schema rows + V1b's soft ground rows (replay)
#   3. eval V2 on V0b schema rows (per question) and on the plain V0b ground rows (no regression check)
set -euo pipefail
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false PIP_BREAK_SYSTEM_PACKAGES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/workspace/hf
export PATH="$HOME/.local/bin:$PATH"
python -m pip install -q transformers datasets accelerate peft pillow duckdb 2>&1 | grep -v WARNING || true
RUN_NAME=${RUN_NAME:-v2-8b-schema}
EVAL=/workspace/m2w_items_j
TRAIN=/workspace/m2w_train_j
SOFT=/workspace/m2w_train_j_soft
INIT=/workspace/jev-probes/model/runs/v1b-8b-m2w-jitter/final
cd /workspace/jev-probes

if ! grep -q '"n_steps"' $EVAL/items.jsonl; then
  echo "=== rebuild items with step fields $(date -u) ==="
  (cd probes/vision && python build_m2w_items.py --split test_domain --files 3 --n 300 --jitter --seed 1 --out $EVAL \
     && python build_m2w_items.py --split train --files 27 --n 8000 --none-frac 0.1 --jitter --seed 2 --out $TRAIN)
  cp -n $TRAIN/requests.jsonl $TRAIN/m2w.train.jsonl || true
fi
echo "=== schema rows $(date -u) ==="
mkdir -p /workspace/m2w_schema
(cd probes/vision && python build_schema_rows.py --items $TRAIN --out /workspace/m2w_schema/schema.train.jsonl --per-item 2 --seed 0 \
   && python build_schema_rows.py --items $EVAL --out /workspace/m2w_schema/v0b_schema.validation.jsonl --per-item 1 --seed 1)
wc -l /workspace/m2w_schema/*.jsonl

cd model
echo "=== zero-shot schema probe: V1b $(date -u) ==="
python eval_schema.py $INIT --data /workspace/m2w_schema/v0b_schema.validation.jsonl --temp 0.5 --out $INIT/eval_schema_v0b_zeroshot.json

echo "=== train $RUN_NAME (stage-2 from V1b) $(date -u) ==="
python train_vl.py --base Qwen/Qwen3-VL-8B-Instruct --data /workspace/m2w_schema --extra-data $SOFT \
  --init-adapter $INIT/backbone --init-heads $INIT/heads.pt \
  --out "runs/$RUN_NAME" --epochs 1 --bsz 4 --grad-accum 4 --lr 5e-5 --head-lr 5e-4 --warmup 50 \
  --eval-every 200 --val-limit 300 --max-pixels 1288000 --grad-ckpt --keep 1
echo "=== eval $(date -u) ==="
python eval_schema.py "runs/$RUN_NAME/final" --data /workspace/m2w_schema/v0b_schema.validation.jsonl --temp 0.5 --out "runs/$RUN_NAME/final/eval_schema_v0b.json"
python eval_vl.py "runs/$RUN_NAME/final" --data $TRAIN/v0b.validation.jsonl --temps 0.5,0.75,1 --out "runs/$RUN_NAME/final/eval_v0b.json"
echo "=== v2 done $(date -u) ==="
