#!/bin/bash
# Push code, recordings, state rows, the V7 adapter and the HF token to a fresh pod, then start the bootstrap.
# Usage: bash scripts/sync_v8_pod.sh   (reads POD_HOST / POD_PORT from scripts/pod.env)
set -e
cd "$(dirname "$0")/.."; . scripts/pod.env
E="ssh -i $HOME/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -p $POD_PORT"
R="rsync -az --no-owner --no-group -e"
$E root@$POD_HOST 'mkdir -p /workspace/jev-probes /workspace/vision/triplets/run2_train /workspace/vision/triplets/run2_val /workspace/vision/state_rows /workspace/hf /workspace/jev-probes/model/runs/v7-8b-general'
$R "$E" --exclude '.venv' --exclude '__pycache__' --exclude 'runs' --exclude 'data' --exclude 'release' --exclude 'results' --exclude '.git' ./ root@$POD_HOST:/workspace/jev-probes/
$R "$E" model/runs/v7-8b-general/final/ root@$POD_HOST:/workspace/jev-probes/model/runs/v7-8b-general/final/
$R "$E" model/data/vision/state_rows/ root@$POD_HOST:/workspace/vision/state_rows/
cat ~/.cache/huggingface/token | $E root@$POD_HOST 'cat > /workspace/hf/token && chmod 600 /workspace/hf/token'
$E root@$POD_HOST 'cd /workspace && (nohup bash /workspace/jev-probes/model/runpod/bootstrap_v8_pod.sh > /workspace/bootstrap.out 2>&1 < /dev/null &) ; echo bootstrap started' || true
# recordings (big): img + pixel + raw for run2_train, raw for run2_val
$R "$E" model/data/vision/triplets/run2_train/img model/data/vision/triplets/run2_train/pixel model/data/vision/triplets/run2_train/raw root@$POD_HOST:/workspace/vision/triplets/run2_train/
$R "$E" model/data/vision/triplets/run2_val/raw root@$POD_HOST:/workspace/vision/triplets/run2_val/
echo "sync done $(date -u)"
