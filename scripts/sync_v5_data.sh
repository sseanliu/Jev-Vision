#!/bin/bash
# Push V5 training data (recorded triplets, synthetic pages, Mind2Web-derived rows) to the training pod.
# Usage: scripts/sync_v5_data.sh   (reads POD_HOST/POD_PORT from scripts/pod.env)
set -e
cd "$(dirname "$0")/.."
source scripts/pod.env
SSH="ssh -i $HOME/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -p $POD_PORT"
$SSH root@$POD_HOST 'mkdir -p /workspace/vision/triplets /workspace/vision/state_rows'
for d in run1 synth1; do
  rsync -az --partial --no-owner --no-group -e "$SSH" model/data/vision/triplets/$d root@$POD_HOST:/workspace/vision/triplets/ 2>&1 | grep -v chown || true
done
rsync -az --no-owner --no-group -e "$SSH" model/data/vision/state_rows/ root@$POD_HOST:/workspace/vision/state_rows/ 2>&1 | grep -v chown || true
# Mind2Web-derived rows point at model/data/vision/m2w_train_j/img on the Mac; on the volume pod those images live at /workspace/m2w_train_j/img
$SSH root@$POD_HOST 'cd /workspace/vision/state_rows && sed -i "s#/Users/xiaoanliu/Github/typesafe/model/data/vision/m2w_train_j#/workspace/m2w_train_j#g" m2w.train.jsonl; ls -la /workspace/vision/state_rows; du -sh /workspace/vision/triplets/*'
