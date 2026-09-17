#!/usr/bin/env bash
# Bootstrap a RunPod pod for training. Expects env: GH_TOKEN (repo read),
# optional HF_TOKEN, RUN_NAME, BASE, TRAIN_ARGS. Logs to /workspace/bootstrap.log.
set -euo pipefail
exec > >(tee -a /workspace/bootstrap.log) 2>&1
echo "=== bootstrap $(date -u) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd /workspace
if [ ! -d jev-probes ]; then
  git clone -q "https://${GH_TOKEN}@github.com/sseanliu/jev-probes.git"
fi
cd jev-probes && git pull -q && cd model

python -m pip install -q --upgrade pip
python -m pip install -q "torch>=2.4" transformers datasets accelerate tiktoken requests numpy
python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__)"

export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
if [ ! -f data/jsonl/mnli.train.jsonl ]; then
  echo "=== converting data ==="
  python data/convert.py --limit-train "${LIMIT_TRAIN:-20000}" --limit-val 1000
fi

echo "=== training ${RUN_NAME:-s1} on ${BASE:-Qwen/Qwen3-1.7B-Base} ==="
python train.py --base "${BASE:-Qwen/Qwen3-1.7B-Base}" --data data/jsonl --out "runs/${RUN_NAME:-s1}" \
  ${TRAIN_ARGS:---epochs 1 --bsz 8 --grad-accum 2 --lr 2e-5 --max-tokens 2048 --eval-every 300 --grad-ckpt}
echo "=== done $(date -u) ==="
