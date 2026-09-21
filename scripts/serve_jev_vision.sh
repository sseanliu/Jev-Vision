#!/usr/bin/env bash
# One-command Jev-Vision server on a fresh CUDA box (RunPod / Lambda / any Ubuntu + NVIDIA GPU with >= 24 GB).
# Installs the deps, downloads the base model and the adapter, serves /v1/systemone on $PORT (default 8811).
#   curl -fsSL https://raw.githubusercontent.com/sseanliu/Jev-Vision/main/scripts/serve_jev_vision.sh | bash
#   REPO=SeanLiu/Jev-Vision PORT=8811 TEMP=0.5 bash scripts/serve_jev_vision.sh
set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HOME=${HF_HOME:-$HOME/hf} PATH="$HOME/.local/bin:$PATH"
REPO=${REPO:-SeanLiu/Jev-Vision}; PORT=${PORT:-8811}; TEMP=${TEMP:-0.5}; DIR=${DIR:-$HOME/jev-vision}
mkdir -p "$DIR"; cd "$DIR"
[ -d Jev-Vision ] || git clone -q https://github.com/sseanliu/Jev-Vision.git
python -m pip install -q "transformers==5.17.*" peft accelerate pillow 2>&1 | tail -n 1
command -v hf >/dev/null || curl -LsSf https://hf.co/cli/install.sh | bash -s 2>&1 | tail -n 1
hf download Qwen/Qwen3-VL-8B-Instruct 2>&1 | tail -n 1
hf download "$REPO" --local-dir "$DIR/Jev-Vision/model/runs/release" 2>&1 | tail -n 1
cd "$DIR/Jev-Vision/model"
echo "serving $REPO on port $PORT"
exec python serve.py runs/release --port "$PORT" --temp "$TEMP"
