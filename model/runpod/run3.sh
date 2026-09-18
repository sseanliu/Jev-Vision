#!/usr/bin/env bash
# Run 3: does an uncertainty-rich Jev-labelled distribution fix OOD calibration?
# Stage-2 fine-tunes from run-1 step6900, same recipe as run 2b, different data:
#   3a: entropy-selected Jev rows only            (data/jsonl_jev_hard, ~8k)
#   3b: entropy-selected + synthetic Jev-labelled (data/jsonl_jev_hard + data/jsonl_synth_jev)
# Then held-out eval + fingerprints. Logs to /workspace/run3.out.
set -euo pipefail
cd /workspace/jev-probes/model
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false PIP_BREAK_SYSTEM_PACKAGES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
INIT=runs/s1-1.7b/step6900
COMMON="--base $INIT/backbone --init-heads $INIT/heads.pt --epochs ${EPOCHS:-2} --bsz 8 --grad-accum 2 --lr 1e-5 --head-lr 3e-4 --warmup 50 --max-tokens 2048 --eval-every 400 --grad-ckpt --keep 1"
for d in data/jsonl_jev_hard data/jsonl_synth_jev; do
  mkdir -p "$d"; for f in data/jsonl/*.validation.jsonl; do ln -sf "$PWD/$f" "$d/"; done
done

if [ ! -f runs/s1-1.7b-3a-hard/final/heads.pt ]; then
  echo "=== 3a: entropy-selected Jev rows $(date -u) ==="
  python train.py $COMMON --data data/jsonl_jev_hard --out runs/s1-1.7b-3a-hard
fi
if [ -f data/jsonl_synth_jev/synth.train.jsonl ] && [ ! -f runs/s1-1.7b-3b-synth/final/heads.pt ]; then
  echo "=== 3b: entropy-selected + synthetic $(date -u) ==="
  python train.py $COMMON --data data/jsonl_jev_hard --extra-data data/jsonl_synth_jev --out runs/s1-1.7b-3b-synth
fi
for r in s1-1.7b-3a-hard s1-1.7b-3b-synth; do
  [ -f runs/$r/final/heads.pt ] || continue
  echo "=== eval $r $(date -u) ==="
  python eval.py runs/$r/final --sets mmlu mmlu_pro anli --limit 500 2>&1 | grep -E "^(mmlu|anli)"
  python fingerprint.py runs/$r/final --out runs/$r/final/fingerprint.json 2>&1 | grep -v -E "Warning|warn|^\s*$" | tail -45
done
echo "=== run3 done $(date -u) ==="
