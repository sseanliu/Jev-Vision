#!/bin/bash
# Pull V5 evals, log and adapter from the training pod (host/port from scripts/pod.env).
set -e
cd "$(dirname "$0")/.."
source scripts/pod.env
E="ssh -i $HOME/.ssh/id_ed25519 -p $POD_PORT"
R=/workspace/jev-probes/model/runs/v5-8b-state
mkdir -p results/vision/v5-8b-state model/runs/v5-8b-state
rsync -az -e "$E" "root@$POD_HOST:$R/final/eval_*.json" "root@$POD_HOST:$R/log.jsonl" results/vision/v5-8b-state/
rsync -az -e "$E" "root@$POD_HOST:/workspace/jev-probes/model/runs/v3-8b-desktop/final/eval_state_zeroshot.json" results/vision/v5-8b-state/eval_state_zeroshot_v3.json
rsync -az --partial -e "$E" "root@$POD_HOST:$R/final" model/runs/v5-8b-state/
ls -la results/vision/v5-8b-state; du -sh model/runs/v5-8b-state/final
