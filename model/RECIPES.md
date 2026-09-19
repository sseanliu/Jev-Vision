# Training recipes (reproduction sheet)

Everything needed to re-run any stage on a new base model: data builders, exact training commands,
hyper-parameters, and where the results live. Narrative and interpretation are in the Obsidian notes
(`~/Obsidian/Projects/TypeSafe/`); this file is the code-facing checklist.

Base model for all vision stages: `Qwen/Qwen3-VL-8B-Instruct` (Apache 2.0). Adapter: LoRA r=64, alpha=128 on
q/k/v/o/gate/up/down of the language model (~175M trainable), frozen vision tower. Heads: pointer readout for
choice/score, sigmoid for noul (`s1/model.py`). Requests: shared state prefix + isolated question branches
(`s1/packing.py`, `s1/mask.py`, `s1/vl.py`). All runs: one epoch, bf16, gradient checkpointing, one H100.

Data locations: Mac `model/data/vision/...` (gitignored except the small jsonl listed below); training pod network
volume `/workspace/...` (US-CA-2, volume `4tnkxpg1go`); private HF staging under `SeanLiu/typesafe-*`.

## Text line (Qwen3-8B-Base, `train.py`) — finished, conclusions only

| run | init | data | key flags | result |
|---|---|---|---|---|
| s1-8b-run3c | scratch | `data/jsonl` (+ `jsonl_jev`, `jsonl_synth_jev`) | lr 1e-4, head-lr 1e-3, bsz 8x2, warmup 100, max-tokens 2048 | MMLU 0.720/ECE 0.079, ANLI 0.470/0.274 |
| s1-8b-run3d | 3c | `jsonl_nli_adv_jev` + `jsonl_replay` | lr 5e-5, head-lr 3e-4, warmup 50 | ANLI 0.546/0.177, AUROC 0.53 -> 0.63 |

Scripts: `runpod/run3c.sh`, `runpod/run3d.sh`; data builders in `../probes/` (`gen_nli_adv.py`, `gen_workflows.py`).

## Vision line (`train_vl.py`)

### Data builders (`../probes/vision/`)
```
# V0b eval set (300 cue-free Mind2Web test_domain items) and the training items
python build_m2w_items.py --split test_domain --files 3 --n 300 --jitter --seed 1 --out $EVAL
python build_m2w_items.py --split train --files 27 --n 8000 --none-frac 0.1 --jitter --seed 2 --out $TRAIN
# soft labels from the 32B teacher (temperature ~4 gives ECE 0.08 on V0b)
python teacher_qwen_vl.py --model Qwen/Qwen3-VL-32B-Instruct --items $TRAIN --limit 100000 --requests-out $SOFT/soft.train.jsonl
# schema-mix rows (ground/act/final/progress/needs_text/tag/history_len, randomised phrasing)
python build_schema_rows.py --items $TRAIN --out /workspace/m2w_schema/schema.train.jsonl --per-item 2 --seed 0
python build_schema_rows.py --items $EVAL  --out /workspace/m2w_schema/v0b_schema.validation.jsonl --per-item 1 --seed 1
# desktop (OS-Atlas macOS), pooled a11y candidates, image-level split: 1,536 train / 360 test
python build_osatlas_items.py ... --out model/data/vision/osatlas_items
python build_schema_rows.py --items $OSA --out /workspace/v3_data/osatlas_schema.train.jsonl --per-item 2 --seed 5
# V5 state rows
python build_state_rows_m2w.py --out model/data/vision/state_rows/m2w.train.jsonl
python ../../harness/record_triplets.py --sites ../../harness/sites.txt --out model/data/vision/triplets/run1 --tasks-per-site 6 --seed 1
python ../../harness/synth_pages/gen.py --out ../../harness/synth_pages/site --serve &
python ../../harness/record_triplets.py --sites ../../harness/synth_pages/site/urls.txt --goals ../../harness/synth_pages/site/goals.jsonl --out model/data/vision/triplets/synth1 --tasks-per-site 6 --seed 2 --mix 0.5,0.2,0.15,0.15
python build_state_rows_record.py --steps model/data/vision/triplets/run1/steps.jsonl --out-dir model/data/vision/state_rows --tag rec1 --val-sites 30
```

### Stages

| run | init | data | flags (bsz 4 x accum 4 everywhere) | steps | headline |
|---|---|---|---|---|---|
| v1b-8b-m2w-jitter | scratch | m2w_train_j (hard) + soft.train.jsonl | lr 1e-4, head-lr 1e-3, warmup 50, eval-every 100 | ~900 | V0b 0.847, ECE 0.043 (T=0.5), AUROC 0.855, 153 ms |
| v2-8b-schema | v1b | m2w_schema (18,250 rows) + soft replay | lr 5e-5, head-lr 5e-4, warmup 50, eval-every 200 | 1,141 | ground 0.847; act 0.967, final 0.958, needs_text 0.934, progress 0.741, tag 0.829 |
| v3-8b-desktop | v2 | osatlas_schema (3,072) + 4,000 m2w_schema replay | lr 5e-5, head-lr 5e-4, warmup 30, eval-every 100 | 442 | desktop 0.914, ECE 0.034 (T=1.0), AUROC 0.958; web heads kept, progress -9 |
| v4-8b-multires | v3 | same as v3, `--multires 1288000,640000,320000` | lr 3e-5, head-lr 3e-4, warmup 20 | 442 | not adopted: web 0.840/0.820/0.810, desktop 0.847/0.739/0.486 at full/640k/320k |
| v5-8b-state | v3 | state_rows (rec1, synth1, m2w) + 4k web + 1.5k desktop replay | lr 4e-5, head-lr 4e-4, warmup 30, eval-every 150 | tbd | skip / effect / done heads |

Scripts: `runpod/run_v1b.sh`, `run_v2.sh`, `run_v3.sh`, `run_v4.sh`, `run_v5.sh`. Each script builds its data,
trains, and writes `runs/<name>/final/{backbone, heads.pt, s1_config.json, eval_*.json}`.

Notes that matter for reproduction:
- `--max-pixels` was a no-op before commit e1bd623 (transformers 5.x ignores a bare `max_pixels`); every run
  above used the full 1280x1000 crop = 1,240 visual tokens. After the fix the budget goes through `size`.
- Temperature is per format: web ground T=0.5, desktop ground T=1.25 (V3), noul/score heads T=1.0.
- Continue-training uses `--init-adapter <run>/final/backbone --init-heads <run>/final/heads.pt`.
- To change the base: set `--base`, keep everything else; heads read the hidden size from the config.
  For a MoE base (Qwen3-VL-30B-A3B) check that PEFT targets resolve inside the expert layers first.

### Evaluation
```
python eval_vl.py <run>/final --data $TRAIN/v0b.validation.jsonl --temps 0.5,0.75,1   # ground acc / ECE / AUROC / ms
python eval_schema.py <run>/final --data /workspace/m2w_schema/v0b_schema.validation.jsonl --temp 0.5   # per-qid
python eval_vl.py <run>/final --data /workspace/v3_data/desktop.validation.jsonl --temps 0.5,1
python ../probes/vision/escalation.py --student <eval_v0b.json> --teacher name=<teacher.json>:channel:ms:cost
```
Baselines and teachers: `teacher_qwen_vl.py` (logprob), `teacher_api.py` (Claude/Gemini verbal), `teacher_dgemma.py`,
`teacher_jev.py`, `tiny_specialist.py`; forms comparison `../probes/forms/eval_forms.py`; state-question judges
`judge_baselines.py`. Live harnesses: `../harness/` (see its README).

### Serving
`python serve.py <run>/final --port 8811 --temp 0.5` (TypeSafe-shaped request; per-question `temperature` override;
JSON-valued instructions/criteria accepted; one or more images).
