#!/usr/bin/env bash
# V4: multi-resolution stage on top of V3. Same data mix as V3 (desktop schema rows + Mind2Web replay),
# each training example packed at a pixel budget drawn from {full, 640k, 320k}, so the model can be served
# at half or a quarter of the visual tokens without the zero-shot collapse seen in the sweep.
set -euo pipefail
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false PIP_BREAK_SYSTEM_PACKAGES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/workspace/hf
RUN_NAME=${RUN_NAME:-v4-8b-multires}
INIT=/workspace/jev-probes/model/runs/v3-8b-desktop/final
cd /workspace/jev-probes/model
echo "=== train $RUN_NAME (stage-2 from V3, multires) $(date -u) ==="
python train_vl.py --base Qwen/Qwen3-VL-8B-Instruct --data /workspace/v3_data \
  --init-adapter $INIT/backbone --init-heads $INIT/heads.pt \
  --out "runs/$RUN_NAME" --epochs 1 --bsz 4 --grad-accum 4 --lr 3e-5 --head-lr 3e-4 --warmup 20 \
  --eval-every 100 --val-limit 360 --multires 1288000,640000,320000 --grad-ckpt --keep 1
F="runs/$RUN_NAME/final"
echo "=== eval $(date -u) ==="
for px in 1288000 640000 320000; do
  echo "=== v4 web px=$px ==="
  python eval_vl.py $F --data /workspace/m2w_train_j/v0b.validation.jsonl --max-pixels $px --temps 0.5,1 --out $F/eval_v0b_px$px.json 2>&1 | grep -v Warning | tail -14
  echo "=== v4 desktop px=$px ==="
  python eval_vl.py $F --data /workspace/v3_data/desktop.validation.jsonl --max-pixels $px --temps 0.5,1 --out $F/eval_desktop_px$px.json 2>&1 | grep -v Warning | tail -14
done
python eval_schema.py $F --data /workspace/m2w_schema/v0b_schema.validation.jsonl --temp 0.5 --out $F/eval_schema_v0b.json 2>&1 | grep -v Warning
echo "=== v4 done $(date -u) ==="
