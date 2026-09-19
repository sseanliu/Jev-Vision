#!/usr/bin/env bash
# V3: second format (macOS desktop, OS-Atlas) on top of V2.
#   1. zero-shot: V2 (web-trained) on the desktop ground validation set  -> web->desktop transfer without training
#   2. stage-2 from V2 on desktop schema rows + a Mind2Web schema replay
#   3. eval V3 on desktop (ground + schema) and on the web V0b schema rows (no regression)
# Waits for the DiffusionGemma baseline to release the GPU first.
set -euo pipefail
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false PIP_BREAK_SYSTEM_PACKAGES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/workspace/hf
RUN_NAME=${RUN_NAME:-v3-8b-desktop}
INIT=/workspace/jev-probes/model/runs/v2-8b-schema/final
OSA=/workspace/osatlas_items
cd /workspace/jev-probes
until grep -q "dgemma done" /workspace/dgemma.out 2>/dev/null; do sleep 60; done
until [ -f $OSA/osatlas.validation.jsonl ] && [ "$(ls $OSA/img | wc -l)" -ge 1896 ]; do sleep 30; done

echo "=== schema rows (desktop) $(date -u) ==="
mkdir -p /workspace/v3_data
# desktop items -> schema rows (ground + act/tag/needs_text/history_len); Mind2Web replay: 4k rows
(cd probes/vision && python - <<'PY'
import json, random
rng = random.Random(3)
rows = [l for l in open("/workspace/m2w_schema/schema.train.jsonl")]
open("/workspace/v3_data/m2w_replay.train.jsonl", "w").writelines(rng.sample(rows, 4000))
PY
python - <<'PY'
import json, shutil
# schema rows need items.jsonl + img/ under one dir; the builder resolves image paths relative to --items
src = "/workspace/osatlas_items"
shutil.copyfile(f"{src}/items.train.jsonl", f"{src}/items.jsonl")
PY
python build_schema_rows.py --items $OSA --out /workspace/v3_data/osatlas_schema.train.jsonl --per-item 2 --seed 5)
# validation: desktop ground rows with absolute image paths
python - <<PY
import json
rows = [json.loads(l) for l in open("$OSA/osatlas.validation.jsonl")]
with open("/workspace/v3_data/desktop.validation.jsonl", "w") as f:
    for r in rows: f.write(json.dumps({**r, "image": "$OSA/" + r["image"]}) + "\n")
print("desktop validation rows", len(rows))
PY
wc -l /workspace/v3_data/*.jsonl

cd model
echo "=== zero-shot: V2 on desktop ground $(date -u) ==="
python eval_vl.py $INIT --data /workspace/v3_data/desktop.validation.jsonl --temps 0.5,1 --out $INIT/eval_desktop_zeroshot.json
echo "=== train $RUN_NAME (stage-2 from V2) $(date -u) ==="
python train_vl.py --base Qwen/Qwen3-VL-8B-Instruct --data /workspace/v3_data \
  --init-adapter $INIT/backbone --init-heads $INIT/heads.pt \
  --out "runs/$RUN_NAME" --epochs 1 --bsz 4 --grad-accum 4 --lr 5e-5 --head-lr 5e-4 --warmup 30 \
  --eval-every 100 --val-limit 360 --max-pixels 1288000 --grad-ckpt --keep 1
echo "=== eval $(date -u) ==="
python eval_vl.py "runs/$RUN_NAME/final" --data /workspace/v3_data/desktop.validation.jsonl --temps 0.5,1 --out "runs/$RUN_NAME/final/eval_desktop.json"
python eval_schema.py "runs/$RUN_NAME/final" --data /workspace/m2w_schema/v0b_schema.validation.jsonl --temp 0.5 --out "runs/$RUN_NAME/final/eval_schema_v0b.json"
echo "=== v3 done $(date -u) ==="
