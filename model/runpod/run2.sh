#!/usr/bin/env bash
# Stage-2 experiments from the run-1 checkpoint (same 33k states, different labels):
#   2b: Jev soft labels            (data/jsonl_jev)
#   2a: open-model soft labels     (data/jsonl_open, scored here with Qwen3-30B-A3B-Base)
# then held-out eval + fingerprints for both. Logs to /workspace/run2.out.
set -euo pipefail
cd /workspace/jev-probes/model
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false PIP_BREAK_SYSTEM_PACKAGES=1
INIT=runs/s1-1.7b/step6900
COMMON="--base $INIT/backbone --init-heads $INIT/heads.pt --epochs 1 --bsz 8 --grad-accum 2 --lr 1e-5 --head-lr 3e-4 --warmup 50 --max-tokens 2048 --eval-every 400 --grad-ckpt --keep 1"

echo "=== 2b: Jev soft labels $(date -u) ==="
python train.py $COMMON --data data/jsonl_jev --out runs/s1-1.7b-2b-jev

echo "=== scoring open-model soft labels with Qwen3-30B-A3B-Base $(date -u) ==="
python distill_open.py --teacher Qwen/Qwen3-30B-A3B-Base --in data/jsonl_jev --out data/jsonl_open --per-source 3000 --bsz 32
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Base

echo "=== 2a: open soft labels $(date -u) ==="
python train.py $COMMON --data data/jsonl_open --out runs/s1-1.7b-2a-open

for r in s1-1.7b-2b-jev s1-1.7b-2a-open; do
  echo "=== eval $r $(date -u) ==="
  python eval.py runs/$r/final --sets mmlu mmlu_pro anli --limit 500 2>&1 | grep -E "^(mmlu|anli)"
  python fingerprint.py runs/$r/final --out runs/$r/final/fingerprint.json 2>&1 | grep -v -E "Warning|warn|^\s*$" | tail -45
done
echo "=== run2 done $(date -u) ==="
