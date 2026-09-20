#!/bin/bash
# Score a checkpoint on both benchmark tracks on the pod and write per-item predictions.
# Usage (on the pod): bash /workspace/pod_eval_tracks.sh v1 runs/v5b-8b-state/final
# Expects /workspace/bench/<name>.candidate.rows.jsonl and <name>.pixel.rows.jsonl (from bench/items_to_rows.py, image
# paths rewritten to /workspace/release/<name>/...).
set -e
NAME=$1; CKPT=$2
cd /workspace/jev-probes/model
for TRACK in candidate pixel; do
  ROWS=/workspace/bench/$NAME.$TRACK.rows.jsonl
  [ -f "$ROWS" ] || { echo "missing $ROWS"; continue; }
  echo "=== eval $NAME $TRACK $(date -u) ==="
  python eval_schema.py "$CKPT" --data "$ROWS" --temp 0.5 --out /workspace/bench/$NAME.$TRACK.eval.json 2>&1 | grep -v Warning | tail -n 12
done
echo "=== tracks done $(date -u) ==="
