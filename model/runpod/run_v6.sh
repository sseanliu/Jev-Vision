#!/usr/bin/env bash
# V6: V5b recipe + the new 138-site recording in both renderings (candidate track and pixel track), so one model
# serves both tracks, plus jev-ultrafast-format `operation` / `*_target` rows (rules text, DONE at real terminal states)
# so the decider head sees the same question the harness sends at run time. Stage-2 from V3, full V5 state rows + rec2train candidate rows (subsampled) + rec2train pixel
# rows + web/desktop replay. Evals: v0 state validation, v1 candidate + pixel tracks, V0b schema, desktop.
#   RUN_NAME=v6-8b-pixel REPLAY_WEB=8000 REPLAY_DESK=3072 LR=3e-5 HEAD_LR=3e-4 CAND_FRAC=0.5 bash run_v6.sh
set -euo pipefail
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false PIP_BREAK_SYSTEM_PACKAGES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=${HF_HOME:-/workspace/hf}
RUN_NAME=${RUN_NAME:-v6-8b-pixel}
REPLAY_WEB=${REPLAY_WEB:-8000}; REPLAY_DESK=${REPLAY_DESK:-3072}; LR=${LR:-3e-5}; HEAD_LR=${HEAD_LR:-3e-4}; CAND_FRAC=${CAND_FRAC:-0.5}
REPO=${REPO:-/workspace/jev-probes}
INIT=$REPO/model/runs/v3-8b-desktop/final
V=/workspace/vision
D=/workspace/v6_data; mkdir -p $D
MAC_PREFIX=${MAC_PREFIX:-/Users/xiaoanliu/Github/typesafe/model/data/vision}
cd $REPO
echo "=== data $(date -u) ==="
for f in $V/state_rows/*.train.jsonl; do sed "s#$MAC_PREFIX#$V#g; s#$V/m2w_train_j#/workspace/m2w_train_j#g" "$f" > $D/$(basename $f); done
sed "s#$MAC_PREFIX#$V#g" $V/state_rows/rec1b.validation.jsonl > $D/state.validation.jsonl
sed "s#$MAC_PREFIX#$V#g" $V/state_rows/rec2_ops.validation.jsonl > $D/ops.validation.jsonl
CAND_FRAC=$CAND_FRAC REPLAY_WEB=$REPLAY_WEB REPLAY_DESK=$REPLAY_DESK python - <<'PY'
import json, random, os
rng = random.Random(6); D = "/workspace/v6_data"
# subsample the rec2train candidate rows (they overlap rec1b in kind); keep every pixel row
rows = [l for l in open(f"{D}/rec2train.train.jsonl")]; k = int(len(rows) * float(os.environ["CAND_FRAC"]))
open(f"{D}/rec2train.train.jsonl", "w").writelines(rng.sample(rows, k)); print("rec2train candidate rows kept", k)
nw = int(os.environ["REPLAY_WEB"]); nd = int(os.environ["REPLAY_DESK"])
web = [l for l in open("/workspace/m2w_schema/schema.train.jsonl")]; open(f"{D}/replay_web.train.jsonl", "w").writelines(rng.sample(web, min(nw, len(web))))
desk = [l for l in open("/workspace/v3_data/osatlas_schema.train.jsonl")]; open(f"{D}/replay_desktop.train.jsonl", "w").writelines(rng.sample(desk, min(nd, len(desk))))
PY
# sanity: every image path in the training rows must exist
python - <<'PY'
import json, glob, os
missing = 0; total = 0
for f in glob.glob("/workspace/v6_data/*.train.jsonl"):
    for l in open(f):
        r = json.loads(l); imgs = r.get("images") or [r.get("image")]
        for p in imgs:
            total += 1
            if p and not os.path.exists(p):
                missing += 1
print(f"image paths checked {total}, missing {missing}")
assert missing == 0, "missing images"
PY
wc -l $D/*.jsonl
cd model
echo "=== train $RUN_NAME (stage-2 from V3, candidate + pixel) $(date -u) ==="
python train_vl.py --data $D --init-adapter $INIT/backbone --init-heads $INIT/heads.pt \
  --out "runs/$RUN_NAME" --epochs 1 --bsz 4 --grad-accum 4 --lr $LR --head-lr $HEAD_LR --warmup 30 \
  --eval-every 200 --val-limit 600 --keep 1 2>&1 | grep -v Warning
F="runs/$RUN_NAME/final"
echo "=== eval $(date -u) ==="
python eval_schema.py $F --data $D/state.validation.jsonl --temp 0.5 --out $F/eval_state.json 2>&1 | grep -v Warning | tail -n 8
python eval_schema.py $F --data $D/ops.validation.jsonl --temp 0.5 --out $F/eval_ops.json 2>&1 | grep -v Warning | tail -n 8
for T in candidate pixel; do
  [ -f /workspace/bench/v1.$T.rows.jsonl ] && python eval_schema.py $F --data /workspace/bench/v1.$T.rows.jsonl --temp 0.5 --out $F/eval_v1_$T.json 2>&1 | grep -v Warning | tail -n 8
done
python eval_schema.py $F --data /workspace/m2w_schema/v0b_schema.validation.jsonl --temp 0.5 --out $F/eval_schema_v0b.json 2>&1 | grep -v Warning | tail -n 10
python eval_vl.py $F --data /workspace/m2w_train_j/v0b.validation.jsonl --temps 0.5,1 --out $F/eval_v0b.json 2>&1 | grep -v Warning | tail -n 4
python eval_vl.py $F --data /workspace/v3_data/desktop.validation.jsonl --temps 0.5,1 --out $F/eval_desktop.json 2>&1 | grep -v Warning | tail -n 4
echo "=== v6 done $(date -u) ==="
