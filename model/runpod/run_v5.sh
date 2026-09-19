#!/usr/bin/env bash
# V5: state questions (skip / effect / done) on top of V3, with web + desktop replay.
# Data is synced from the Mac to /workspace/vision (triplets/*, state_rows/*) and the jsonl image paths are
# rewritten from the Mac prefix to the pod prefix before training.
#   1. stage-2 from V3 on state rows (recorded sites + synthetic pages + Mind2Web-derived) + replay
#   2. eval: state questions on the held-out sites (eval_schema), web V0b schema, desktop ground
set -euo pipefail
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false PIP_BREAK_SYSTEM_PACKAGES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=${HF_HOME:-/workspace/hf}
RUN_NAME=${RUN_NAME:-v5-8b-state}
REPO=${REPO:-/workspace/jev-probes}
INIT=$REPO/model/runs/v3-8b-desktop/final
V=/workspace/vision                      # synced: $V/triplets/{run1,synth1}/img, $V/state_rows/*.jsonl
D=/workspace/v5_data; mkdir -p $D
MAC_PREFIX=${MAC_PREFIX:-/Users/xiaoanliu/Github/typesafe/model/data/vision}
cd $REPO
echo "=== data $(date -u) ==="
for f in $V/state_rows/*.train.jsonl; do sed "s#$MAC_PREFIX#$V#g" "$f" > $D/$(basename $f); done
sed "s#$MAC_PREFIX#$V#g" $V/state_rows/rec1b.validation.jsonl > $D/state.validation.jsonl
# replay: 4k web schema rows + 1.5k desktop schema rows keep grounding and the other heads
python - <<'PY'
import json, random
rng = random.Random(5)
web = [l for l in open("/workspace/m2w_schema/schema.train.jsonl")]; open("/workspace/v5_data/replay_web.train.jsonl", "w").writelines(rng.sample(web, 4000))
desk = [l for l in open("/workspace/v3_data/osatlas_schema.train.jsonl")]; open("/workspace/v5_data/replay_desktop.train.jsonl", "w").writelines(rng.sample(desk, min(1500, len(desk))))
PY
wc -l $D/*.jsonl
cd model
echo "=== zero-shot: V3 on state questions $(date -u) ==="
python eval_schema.py $INIT --data $D/state.validation.jsonl --temp 1.0 --out $INIT/eval_state_zeroshot.json 2>&1 | grep -v Warning
echo "=== train $RUN_NAME (stage-2 from V3) $(date -u) ==="
python train_vl.py --base Qwen/Qwen3-VL-8B-Instruct --data $D \
  --init-adapter $INIT/backbone --init-heads $INIT/heads.pt \
  --out "runs/$RUN_NAME" --epochs 1 --bsz 4 --grad-accum 4 --lr 4e-5 --head-lr 4e-4 --warmup 30 \
  --eval-every 150 --val-limit 600 --grad-ckpt --keep 1
F="runs/$RUN_NAME/final"
echo "=== eval $(date -u) ==="
python eval_schema.py $F --data $D/state.validation.jsonl --temp 1.0 --out $F/eval_state.json 2>&1 | grep -v Warning
python eval_schema.py $F --data /workspace/m2w_schema/v0b_schema.validation.jsonl --temp 0.5 --out $F/eval_schema_v0b.json 2>&1 | grep -v Warning
python eval_vl.py $F --data /workspace/m2w_train_j/v0b.validation.jsonl --temps 0.5,1 --out $F/eval_v0b.json 2>&1 | grep -v Warning | tail -8
python eval_vl.py $F --data /workspace/v3_data/desktop.validation.jsonl --temps 0.5,1 --out $F/eval_desktop.json 2>&1 | grep -v Warning | tail -8
echo "=== v5 done $(date -u) ==="
