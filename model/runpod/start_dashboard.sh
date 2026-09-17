#!/usr/bin/env bash
# Start (or restart) the live dashboard on port 8888. Uses a pidfile so the
# restart never pattern-matches its own shell.
cd /workspace/jev-probes/model
PID=/workspace/dashboard.pid
if [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; then kill "$(cat "$PID")"; sleep 1; fi
nohup python runpod/dashboard.py --runs runs --port 8888 > /workspace/dashboard.out 2>&1 &
echo $! > "$PID"
sleep 2
echo "pid $(cat "$PID") alive=$(kill -0 "$(cat "$PID")" 2>/dev/null && echo yes || echo no)"
curl -s -o /dev/null -w "local page HTTP %{http_code}\n" localhost:8888/
curl -s localhost:8888/api/runs | head -c 200; echo
tail -3 /workspace/dashboard.out
