#!/bin/bash
# Pull V2 results from the pod, restart the server on V2, rerun the live bench, commit.
set -e
cd "$(dirname "$0")/.."
source scripts/pod.env
SSH="ssh -i $HOME/.ssh/id_ed25519 -p $POD_PORT root@$POD_HOST"
R=/workspace/jev-probes/model/runs/v2-8b-schema
mkdir -p results/vision/v2-8b-schema
rsync -az --no-owner --no-group -e "ssh -i $HOME/.ssh/id_ed25519 -p $POD_PORT" \
  root@$POD_HOST:$R/final/eval_schema_v0b.json root@$POD_HOST:$R/final/eval_v0b.json root@$POD_HOST:$R/log.jsonl \
  results/vision/v2-8b-schema/ 2>&1 | grep -v chown || true
$SSH "rsync -a $R/final/ $R/final_bak/ 2>/dev/null; true"
$SSH "pkill -f '[s]erve.py' ; sleep 2; cd /workspace/jev-probes/model && setsid nohup python serve.py runs/v2-8b-schema/final --port 8811 --temp 0.5 > /workspace/serve.out 2>&1 < /dev/null & disown" || true
echo "pulled; server restarting on V2"
