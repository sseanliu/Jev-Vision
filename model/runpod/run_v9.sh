#!/usr/bin/env bash
# V9: stage-3 from V7 with the CORRECTED operation rows (gold from the task, not the recorded policy). Same recipe as V8 otherwise.
# question, history-bearing ones doubled) and the yes/no head's saturation on general images (label smoothing, fresh
# GQA yes/no rows). Replay keeps general + screen skills. Evals: general, state, v1 both tracks, ops, V0b, desktop.
#   RUN_NAME=v8-8b-decider LR=1.5e-5 HEAD_LR=1.5e-4 SMOOTH=0.05 bash run_v8.sh
set -euo pipefail
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false PIP_BREAK_SYSTEM_PACKAGES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=${HF_HOME:-/workspace/hf}
RUN_NAME=${RUN_NAME:-v9-8b-ops}; LR=${LR:-1.5e-5}; HEAD_LR=${HEAD_LR:-1.5e-4}; SMOOTH=${SMOOTH:-0.02}
GEN_SAMPLE=${GEN_SAMPLE:-3000}; GQA_SAMPLE=${GQA_SAMPLE:-6000}; REPLAY_PIXEL=${REPLAY_PIXEL:-3000}; REPLAY_STATE=${REPLAY_STATE:-2000}; REPLAY_WEB=${REPLAY_WEB:-1500}; REPLAY_DESK=${REPLAY_DESK:-1000}
REPO=${REPO:-/workspace/jev-probes}; INIT=${INIT:-$REPO/model/runs/v7-8b-general/final}
V=/workspace/vision; D=/workspace/v9_data; mkdir -p $D; rm -f $D/*.jsonl
MAC_PREFIX=${MAC_PREFIX:-/Users/xiaoanliu/Github/typesafe/model/data/vision}
cd $REPO
echo "=== data $(date -u) ==="
for f in rec2train_ops_full.train rec2train_pixel.train rec1b.train rec2train.train rec1.train; do sed "s#$MAC_PREFIX#$V#g" $V/state_rows/$f.jsonl > $D/src_$f.jsonl; done
sed "s#$MAC_PREFIX#$V#g" $V/state_rows/rec1b.validation.jsonl > $D/state.validation.jsonl
sed "s#$MAC_PREFIX#$V#g" $V/state_rows/rec2_ops.validation.jsonl > $D/ops.validation.jsonl
GEN_SAMPLE=$GEN_SAMPLE GQA_SAMPLE=$GQA_SAMPLE REPLAY_PIXEL=$REPLAY_PIXEL REPLAY_STATE=$REPLAY_STATE REPLAY_WEB=$REPLAY_WEB REPLAY_DESK=$REPLAY_DESK python - <<'PY'
import random, os, json
rng = random.Random(8); D = "/workspace/v9_data"; V = "/workspace/vision"
def take(paths, n, out):
    rows = [l for p in paths for l in open(p)]
    open(out, "w").writelines(rng.sample(rows, min(n, len(rows)))); print(out, min(n, len(rows)))
# ops: every row, and history-bearing target rows twice
ops = [l for l in open(f"{D}/src_rec2train_ops_full.train.jsonl")]
dup = [l for l in ops if json.loads(l)["questions"][0]["qid"].endswith("_target") and '"recent_actions": [\n  {' in json.loads(l)["state"]]
open(f"{D}/ops.train.jsonl", "w").writelines(ops + dup); print("ops rows", len(ops), "+ doubled history targets", len(dup))
take([f"{V}/general_rows/general.train.jsonl"], int(os.environ["GEN_SAMPLE"]), f"{D}/general.train.jsonl")
take([f"{V}/general_rows/gqa_yn.train.jsonl"], int(os.environ["GQA_SAMPLE"]), f"{D}/gqa.train.jsonl")
take([f"{D}/src_rec2train_pixel.train.jsonl"], int(os.environ["REPLAY_PIXEL"]), f"{D}/replay_pixel.train.jsonl")
take([f"{D}/src_rec1b.train.jsonl", f"{D}/src_rec2train.train.jsonl", f"{D}/src_rec1.train.jsonl"], int(os.environ["REPLAY_STATE"]), f"{D}/replay_state.train.jsonl")
take(["/workspace/m2w_schema/schema.train.jsonl"], int(os.environ["REPLAY_WEB"]), f"{D}/replay_web.train.jsonl")
take(["/workspace/v3_data/osatlas_schema.train.jsonl"], int(os.environ["REPLAY_DESK"]), f"{D}/replay_desktop.train.jsonl")
PY
rm -f $D/src_*.jsonl
python - <<'PY'
import json, glob, os
missing = total = 0
for f in glob.glob("/workspace/v9_data/*.train.jsonl"):
    for l in open(f):
        r = json.loads(l)
        for p in (r.get("images") or [r.get("image")]):
            total += 1; missing += (p is not None and not os.path.exists(p))
print(f"image paths checked {total}, missing {missing}"); assert missing == 0
PY
wc -l $D/*.jsonl
cd model
echo "=== train $RUN_NAME (stage-3 from $INIT, smooth $SMOOTH) $(date -u) ==="
python train_vl.py --data $D --init-adapter $INIT/backbone --init-heads $INIT/heads.pt --noul-smooth $SMOOTH \
  --out "runs/$RUN_NAME" --epochs 1 --bsz 2 --grad-accum 8 --grad-ckpt --lr $LR --head-lr $HEAD_LR --warmup 30 \
  --eval-every 200 --val-limit 600 --keep 1 2>&1 | grep -v Warning
F="runs/$RUN_NAME/final"
echo "=== eval $(date -u) ==="
python eval_schema.py $F --data /workspace/bench/general/general.rows.jsonl --temp 0.5 --out $F/eval_general.json 2>&1 | grep -v Warning | tail -n 8
python eval_schema.py $F --data $D/ops.validation.jsonl --temp 0.5 --out $F/eval_ops.json 2>&1 | grep -v Warning | tail -n 8
python eval_schema.py $F --data $D/state.validation.jsonl --temp 0.5 --out $F/eval_state.json 2>&1 | grep -v Warning | tail -n 8
for T in candidate pixel; do
  [ -f /workspace/bench/v1.$T.rows.jsonl ] && python eval_schema.py $F --data /workspace/bench/v1.$T.rows.jsonl --temp 0.5 --out $F/eval_v1_$T.json 2>&1 | grep -v Warning | tail -n 8
done
python eval_schema.py $F --data /workspace/m2w_schema/v0b_schema.validation.jsonl --temp 0.5 --out $F/eval_schema_v0b.json 2>&1 | grep -v Warning | tail -n 10
python eval_vl.py $F --data /workspace/v3_data/desktop.validation.jsonl --temps 0.5,1 --out $F/eval_desktop.json 2>&1 | grep -v Warning | tail -n 4
echo "=== v9 done $(date -u) ==="
