#!/usr/bin/env bash
# Run 3d: stage-2 on the 8B LoRA checkpoint (run 3c) with same-format adversarial NLI rows
# labelled by Jev (HANS + unseen MNLI, data/jsonl_nli_adv_jev) plus a small replay of the
# original mix so nothing else is forgotten. Question: does ANLI mean top-p finally move off 0.74?
set -euo pipefail
cd /workspace/jev-probes/model
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false PIP_BREAK_SYSTEM_PACKAGES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/workspace/hf
python -m pip install -q peft datasets 2>&1 | grep -v WARNING || true
RUN_NAME=${RUN_NAME:-s1-8b-run3d}
INIT=runs/s1-8b-run3c/final
# replay: 4k templated hard + 1k entropy-selected + 1k synthetic (fixed seed)
python - <<'PY'
import json, random
from pathlib import Path
rng = random.Random(0)
out = Path("data/jsonl_replay"); out.mkdir(exist_ok=True)
def sample(glob, n, name):
    rows = [l for p in sorted(Path("data").glob(glob)) for l in open(p)]
    rows = rng.sample(rows, min(n, len(rows)))
    (out / f"{name}.train.jsonl").write_text("".join(rows)); print(name, len(rows))
sample("jsonl/*.train.jsonl", 4000, "replay_hard")
sample("jsonl_jev_hard/*.train.jsonl", 1000, "replay_jev_hard")
sample("jsonl_synth_jev/*.train.jsonl", 1000, "replay_synth")
PY
for d in data/jsonl_nli_adv_jev data/jsonl_replay; do for f in data/jsonl/*.validation.jsonl; do ln -sf "$PWD/$f" "$d/"; done; done
wc -l data/jsonl_nli_adv_jev/*.train.jsonl data/jsonl_replay/*.train.jsonl
echo "=== $RUN_NAME: stage-2 from $INIT on nli_adv_jev + replay $(date -u) ==="
python train.py --base Qwen/Qwen3-8B-Base --attn sdpa \
  --init-adapter "$INIT/backbone" --init-heads "$INIT/heads.pt" \
  --data data/jsonl_nli_adv_jev --extra-data data/jsonl_replay --out "runs/$RUN_NAME" \
  --epochs 1 --bsz 8 --grad-accum 2 --lr 5e-5 --head-lr 3e-4 --warmup 50 --max-tokens 2048 --eval-every 300 --grad-ckpt --keep 1
echo "=== eval $(date -u) ==="
python eval.py "runs/$RUN_NAME/final" --sets mmlu mmlu_pro anli --limit 500 --temps 1,1.5,2,3,4 2>&1 | grep -E "^(mmlu|anli)|Traceback|Error" || true
python fingerprint.py "runs/$RUN_NAME/final" --out "runs/$RUN_NAME/final/fingerprint.json" 2>&1 | grep -v -E "Warning|warn|^\s*$" | tail -45 || true
echo "=== run3d done $(date -u) ==="
