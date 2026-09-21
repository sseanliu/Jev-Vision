#!/usr/bin/env bash
# Bootstrap a fresh pod (no network volume) for V8: packages, base model, Mind2Web + OS-Atlas rebuild, benchmark
# rows (v1 both tracks, general track) from the private HF datasets, general-vision training rows (A-OKVQA, Food-101,
# NLVR2, GQA yes/no). Code, recordings (run2_train, run2_val raw), state rows and the V7 adapter arrive by rsync from
# the Mac (scripts/sync_v8_pod.sh). Needs the HF token at /workspace/hf/token (copied by the sync script).
set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HOME=/workspace/hf PATH="$HOME/.local/bin:$PATH"
echo "=== pip $(date -u)"; python -m pip install -q transformers==5.17.* peft accelerate pillow duckdb datasets 2>&1 | tail -1
curl -LsSf https://hf.co/cli/install.sh | bash -s 2>&1 | tail -1
echo "=== base model $(date -u)"; hf download Qwen/Qwen3-VL-8B-Instruct 2>&1 | tail -1
cd /workspace/jev-probes/probes/vision
echo "=== mind2web items $(date -u)"
python build_m2w_items.py --split test_domain --files 3 --n 300 --jitter --seed 1 --out /workspace/m2w_items_j 2>&1 | tail -1
python build_m2w_items.py --split train --files 27 --n 8000 --none-frac 0.1 --jitter --seed 2 --out /workspace/m2w_train_j 2>&1 | tail -1
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
echo "=== osatlas $(date -u)"
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
echo "=== benchmark rows $(date -u)"
mkdir -p /workspace/release /workspace/bench
hf download SeanLiu/step-verifier-bench-v1 --repo-type dataset --local-dir /workspace/release/v1 2>&1 | tail -1
cd /workspace/jev-probes
python bench/items_to_rows.py --items bench/v1_items.jsonl --release /workspace/release/v1 --out /workspace/bench/v1.candidate.rows.jsonl 2>&1 | tail -1
python bench/items_to_rows.py --items bench/v1_items_pixel.jsonl --release /workspace/release/v1 --out /workspace/bench/v1.pixel.rows.jsonl 2>&1 | tail -1
hf download SeanLiu/jev-vision-bench-general --repo-type dataset --local-dir /workspace/bench/general 2>&1 | tail -1
python - <<'PY'
import json
out = open("/workspace/bench/general/general.rows.jsonl", "w"); n = 0
for l in open("/workspace/bench/general/items.jsonl"):
    it = json.loads(l); q = {"qid": it["question"], "qtype": it["type"], "instructions": it["instructions"]}
    if it["type"] == "choice": q["criteria"] = it["criteria"]
    out.write(json.dumps({"images": ["/workspace/bench/general/" + p for p in it["images"]], "state": it["state"] or "A photograph.",
                          "questions": [q], "targets": {it["question"]: it["label"]}, "meta": {"id": it["id"], "source": it["source"]}}, ensure_ascii=False) + "\n"); n += 1
print("general rows", n)
PY
echo "=== general training rows $(date -u)"
cd /workspace/jev-probes/probes/vision
python build_general_rows.py --out /workspace/vision/general_rows 2>&1 | grep -v Warning | tail -n 2
python build_gqa_rows.py --out /workspace/vision/general_rows --n 16000 --per-image 3 --seed 9 2>&1 | grep -v Warning | tail -n 2
echo "=== bootstrap done $(date -u)"
