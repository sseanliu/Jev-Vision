#!/usr/bin/env bash
# Bootstrap a fresh pod (no network volume) for V5 training: packages, base model, Mind2Web + OS-Atlas rebuild.
# Code, the V3 adapter, triplet images and state rows are rsync'd from the Mac by scripts/sync_v5_data.sh.
set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HOME=/workspace/hf PATH="$HOME/.local/bin:$PATH"
echo "=== pip $(date -u)"; python -m pip install -q transformers==5.17.* peft accelerate pillow duckdb 2>&1 | tail -1
curl -LsSf https://hf.co/cli/install.sh | bash -s 2>&1 | tail -1
echo "=== base model $(date -u)"; hf download Qwen/Qwen3-VL-8B-Instruct 2>&1 | tail -1
mkdir -p /workspace && cd /workspace/jev-probes/probes/vision
echo "=== mind2web items (same seeds as V1b) $(date -u)"
python build_m2w_items.py --split test_domain --files 3 --n 300 --jitter --seed 1 --out /workspace/m2w_items_j 2>&1 | tail -1
python build_m2w_items.py --split train --files 27 --n 8000 --none-frac 0.1 --jitter --seed 2 --out /workspace/m2w_train_j 2>&1 | tail -1
echo "=== schema rows $(date -u)"
mkdir -p /workspace/m2w_schema
python build_schema_rows.py --items /workspace/m2w_train_j --out /workspace/m2w_schema/schema.train.jsonl --per-item 2 --seed 0 2>&1 | tail -1
python build_schema_rows.py --items /workspace/m2w_items_j --out /workspace/m2w_schema/v0b_schema.validation.jsonl --per-item 1 --seed 1 2>&1 | tail -1
python - <<'PY'
import json
rows = [json.loads(l) for l in open("/workspace/m2w_items_j/requests.jsonl")]
with open("/workspace/m2w_train_j/v0b.validation.jsonl", "w") as f:
    for r in rows: f.write(json.dumps({**r, "image": "/workspace/m2w_items_j/" + r["image"]}) + "\n")
print("v0b validation rows", len(rows))
PY
echo "=== osatlas items from HF $(date -u)"
hf download SeanLiu/typesafe-cua-data --repo-type dataset --include "osatlas_items/*" --local-dir /workspace 2>&1 | tail -1
mkdir -p /workspace/v3_data && cp /workspace/osatlas_items/items.train.jsonl /workspace/osatlas_items/items.jsonl
python build_schema_rows.py --items /workspace/osatlas_items --out /workspace/v3_data/osatlas_schema.train.jsonl --per-item 2 --seed 5 2>&1 | tail -1
python - <<'PY'
import json
rows = [json.loads(l) for l in open("/workspace/osatlas_items/osatlas.validation.jsonl")]
with open("/workspace/v3_data/desktop.validation.jsonl", "w") as f:
    for r in rows: f.write(json.dumps({**r, "image": "/workspace/osatlas_items/" + r["image"]}) + "\n")
print("desktop validation rows", len(rows))
PY
echo "=== bootstrap done $(date -u)"
