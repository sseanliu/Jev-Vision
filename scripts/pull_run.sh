#!/usr/bin/env bash
# Pull a finished run's final checkpoint + results from the pod to local.
# usage: scripts/pull_run.sh <run_name>      (weights go to model/runs/<run_name>/final, gitignored)
set -euo pipefail
cd "$(dirname "$0")/.."; . scripts/pod.env
RS="ssh -i $HOME/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -p $POD_PORT"
run=$1; mkdir -p "model/runs/$run" "results/$run"
rsync -rlt --no-o --no-g --no-p -z --info=progress2 -e "$RS" "root@$POD_HOST:/workspace/jev-probes/model/runs/$run/final/" "model/runs/$run/final/"
rsync -rlt --no-o --no-g --no-p -z -e "$RS" "root@$POD_HOST:/workspace/jev-probes/model/runs/$run/log.jsonl" "results/$run/"
for f in eval.json fingerprint.json; do rsync -rlt --no-o --no-g --no-p -z -e "$RS" "root@$POD_HOST:/workspace/jev-probes/model/runs/$run/final/$f" "results/$run/" 2>/dev/null || true; done
du -sh "model/runs/$run/final"; ls "results/$run"
