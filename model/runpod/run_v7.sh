#!/usr/bin/env bash
# V7: general vision mix. Stage-3 from V6 (candidate + pixel + ops) on typed questions from public vision train
# splits (A-OKVQA, Food-101, VQAv2 yes/no, GQA yes/no, NLVR2) plus screen replay (state rows, pixel rows, ops rows,
# Mind2Web schema, desktop grounding), so one model serves general images and screens. Evals: general track (1,500),
# state validation, v1 candidate + pixel, ops validation, V0b, desktop.
#   RUN_NAME=v7-8b-general LR=2e-5 HEAD_LR=2e-4 bash run_v7.sh
set -euo pipefail
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false PIP_BREAK_SYSTEM_PACKAGES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=${HF_HOME:-/workspace/hf}
RUN_NAME=${RUN_NAME:-v7-8b-general}; LR=${LR:-2e-5}; HEAD_LR=${HEAD_LR:-2e-4}
REPLAY_STATE=${REPLAY_STATE:-3000}; REPLAY_PIXEL=${REPLAY_PIXEL:-2500}; REPLAY_OPS=${REPLAY_OPS:-1500}; REPLAY_WEB=${REPLAY_WEB:-3000}; REPLAY_DESK=${REPLAY_DESK:-1500}
REPO=${REPO:-/workspace/jev-probes}
INIT=${INIT:-$REPO/model/runs/v6-8b-pixel/final}
V=/workspace/vision; D=/workspace/v7_data; mkdir -p $D; rm -f $D/*.jsonl
cd $REPO
echo "=== data $(date -u) ==="
for f in $V/general_rows/*.train.jsonl; do cp "$f" $D/general_$(basename $f); done
cp /workspace/v6_data/state.validation.jsonl $D/state.validation.jsonl
REPLAY_STATE=$REPLAY_STATE REPLAY_PIXEL=$REPLAY_PIXEL REPLAY_OPS=$REPLAY_OPS REPLAY_WEB=$REPLAY_WEB REPLAY_DESK=$REPLAY_DESK python - <<'PY'
import random, os, glob
rng = random.Random(7); D = "/workspace/v7_data"; S = "/workspace/v6_data"
def take(paths, n, out):
    rows = [l for p in paths for l in open(p)]
    open(out, "w").writelines(rng.sample(rows, min(n, len(rows)))); print(out, min(n, len(rows)))
take([f"{S}/rec1b.train.jsonl", f"{S}/rec2train.train.jsonl", f"{S}/rec1.train.jsonl"], int(os.environ["REPLAY_STATE"]), f"{D}/replay_state.train.jsonl")
take([f"{S}/rec2train_pixel.train.jsonl"], int(os.environ["REPLAY_PIXEL"]), f"{D}/replay_pixel.train.jsonl")
take([f"{S}/rec2train_ops.train.jsonl"], int(os.environ["REPLAY_OPS"]), f"{D}/replay_ops.train.jsonl")
take([f"{S}/replay_web.train.jsonl"], int(os.environ["REPLAY_WEB"]), f"{D}/replay_web.train.jsonl")
take([f"{S}/replay_desktop.train.jsonl"], int(os.environ["REPLAY_DESK"]), f"{D}/replay_desktop.train.jsonl")
PY
python - <<'PY'
import json, glob, os
missing = total = 0
for f in glob.glob("/workspace/v7_data/*.train.jsonl"):
    for l in open(f):
        r = json.loads(l)
        for p in (r.get("images") or [r.get("image")]):
            total += 1; missing += (p is not None and not os.path.exists(p))
print(f"image paths checked {total}, missing {missing}"); assert missing == 0
PY
wc -l $D/*.jsonl
cd model
echo "=== train $RUN_NAME (stage-3 from $INIT, general mix + screen replay) $(date -u) ==="
python train_vl.py --data $D --init-adapter $INIT/backbone --init-heads $INIT/heads.pt \
  --out "runs/$RUN_NAME" --epochs 1 --bsz 2 --grad-accum 8 --grad-ckpt --lr $LR --head-lr $HEAD_LR --warmup 30 \
  --eval-every 200 --val-limit 600 --keep 1 2>&1 | grep -v Warning
F="runs/$RUN_NAME/final"
echo "=== eval $(date -u) ==="
python eval_schema.py $F --data /workspace/bench/general/general.rows.jsonl --temp 0.5 --out $F/eval_general.json 2>&1 | grep -v Warning | tail -n 8
python eval_schema.py $F --data $D/state.validation.jsonl --temp 0.5 --out $F/eval_state.json 2>&1 | grep -v Warning | tail -n 8
for T in candidate pixel; do
  [ -f /workspace/bench/v1.$T.rows.jsonl ] && python eval_schema.py $F --data /workspace/bench/v1.$T.rows.jsonl --temp 0.5 --out $F/eval_v1_$T.json 2>&1 | grep -v Warning | tail -n 8
done
python eval_schema.py $F --data /workspace/v6_data/ops.validation.jsonl --temp 0.5 --out $F/eval_ops.json 2>&1 | grep -v Warning | tail -n 8
python eval_schema.py $F --data /workspace/m2w_schema/v0b_schema.validation.jsonl --temp 0.5 --out $F/eval_schema_v0b.json 2>&1 | grep -v Warning | tail -n 10
python eval_vl.py $F --data /workspace/v3_data/desktop.validation.jsonl --temps 0.5,1 --out $F/eval_desktop.json 2>&1 | grep -v Warning | tail -n 4
echo "=== v7 done $(date -u) ==="
