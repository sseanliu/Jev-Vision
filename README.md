# Jev-Vision: a step verifier for computer-use agents, with open weights

An 8B vision model that watches a computer-use agent work and answers, in one forward pass of about 160 ms, the
bounded questions the agent loop needs at every step:

| question | asked | answer |
|---|---|---|
| `ground` | before acting | which marked element to act on next |
| `skip` | before acting | is this element already in the state the task needs, so it should be left alone |
| `effect` | after acting, before + after screenshots | did the last action change the page as intended |
| `done` | after acting | has the goal been fully achieved |

It returns calibrated probabilities, never free text, in the request and response shape of TypeSafe's Jev
(`/v1/systemone`: a `state`, a dict of typed `questions`, one probability per question). The difference from Jev
and from the prompt-only open re-creations is that this one looks at the screenshot and was trained on screens, with
labels that come from the environment (URL and DOM change, field values, target page reached) rather than from a
model or a human.

**Weights: https://huggingface.co/SeanLiu/Jev-Vision** (V9, Apache-2.0). One-command server on any CUDA box with 24 GB:

```sh
curl -fsSL https://raw.githubusercontent.com/sseanliu/Jev-Vision/main/scripts/serve_jev_vision.sh | bash
```

This repository is updated as results land. Current checkpoint: V9. Status and plans are tracked in the commit log.

## Results

### General track: typed decisions on general images

1,500 items from public datasets (POPE, MME, NLVR2 two-image, A-OKVQA 4-way, Food-101 20-way), built by
`bench/build_general_items.py`. The frozen backbone with a prompt and next-token logit readout is what a prompt-only
open "vision Jev" gets from the same 8B model.

| system | POPE | MME | NLVR2 | A-OKVQA | Food-101 | mean | ECE (yes/no) | ms |
|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-8B frozen, prompt + logit readout | 0.900 | 0.903 | 0.897 | 0.903 | 0.950 | 0.911 | 0.072 to 0.085 | 185 per question |
| V5b (screens only) | 0.860 | 0.890 | 0.917 | 0.890 | 0.937 | 0.899 | 0.016 to 0.094 | 269 |
| V7 (general mix + screens) | 0.913 | 0.927 | 0.933 | 0.903 | 0.937 | 0.923 | 0.022 to 0.054 | 112 |
| **V9 (released)** | 0.907 | 0.927 | 0.930 | 0.893 | 0.930 | 0.917 | 0.034 to 0.047 | **74** |
| Qwen3.5-9B frozen, prompt + logit readout | 0.910 | 0.897 | 0.857 | 0.917 | 0.933 | 0.903 | 0.016 to 0.048 | 207 |

The 95% bootstrap intervals of the mean overlap (V9 0.902 to 0.932, frozen 0.897 to 0.925): accuracy is a tie. The
fine-tuned models state far fewer of their errors at p>=0.99 (13% of errors vs 37% for the frozen readout), and their
yes/no class AUROC (P(yes) against the label) is equal or better: V9 0.97 / 0.97 / 0.98 on POPE / MME / NLVR2 against
0.95 / 0.97 / 0.96 for the frozen readout. (An earlier revision of this README reported a ranking gap; it compared two
different AUROC definitions. Corrected 2026-09-21.) Frozen Qwen3.5-9B with the same readout scores 0.903 with the
best out-of-the-box calibration on the table. Full table with CIs and high-confidence errors in
`bench/general_leaderboard.md`.

### Screen tracks: per-step judgments for computer-use agents

Benchmark v1: fresh episodes on 30 web sites held out from training (site-level split), labels from the environment.
Accuracy per question with positive / negative class accuracy in brackets, expected calibration error, and latency.

Candidate track (screenshot with numbered marks plus a text table of the candidate elements):

| system | skip | effect | done | ECE skip/effect/done | ms |
|---|---|---|---|---|---|
| Jev 1.13 (text only, candidate table) | 0.870 (0.55/0.92) | 0.733 (0.66/0.98) | 0.887 (0.89/0.89) | 0.105/0.096/0.049 | 238 (API p50) |
| Claude Sonnet 5 (screenshots, verbal probability) | 0.850 (0.31/0.93) | 0.477 (0.33/0.94) | 0.780 (0.35/0.93) | 0.119/0.301/0.105 | n/a |
| rules (URL/DOM change for effect, constant no) | 0.865 | 0.892 | 0.758 | 0.035/0.008/0.142 | 0 |
| ours, V5b 8B | **0.936** (0.79/0.96) | 0.967 (0.97/0.96) | 0.896 (0.83/0.92) | 0.042/0.020/0.055 | **157** |
| ours, V7 8B | 0.928 | 0.962 | **0.898** | 0.048/0.019/0.066 | 185 |
| **ours, V9 8B (released)** | 0.922 | **0.977** | 0.887 | 0.054/0.010/0.077 | 154 |

Pixel track (raw screenshot with one marker on the candidate, no table; the same episodes):

| system | skip | effect | done | ECE | ms |
|---|---|---|---|---|---|
| ours, V5b 8B, zero-shot on this rendering | 0.931 (0.89/0.94) | 0.908 (0.97/0.69) | 0.902 (0.88/0.91) | 0.039/0.080/0.046 | 182 |
| ours, V7 8B, trained on this rendering | **0.933** | 0.915 | **0.902** | 0.042/0.052/0.061 | 187 |
| **ours, V9 8B (released)** | 0.931 | **0.931** | 0.891 | 0.040/0.035/0.067 | 152 |

Benchmark v0 (2,223 items, same 30 sites, earlier episodes): skip 0.955, effect 0.946, done 0.944, ground 0.939 for
V9 (V7: 0.952 / 0.941 / 0.935 / 0.946); full tables with AUROC and selective accuracy in `bench/v0_leaderboard.md` and `bench/v1_leaderboard.md`.
Latency is one H100, batch 1, 1280x1000 screenshots, all questions of a step in one forward.

Deciding, not only verifying: on the harness's own `operation` question (CLICK / TYPE_TEXT / DONE / BLOCKED with the
rules text jev-ultrafast sends, 615 held-out rows with task-derived gold), V5b answered zero-shot at 0.656; V9,
trained on 3,113 such rows, scores 0.911, and 0.969 on `click_target`. Closed loop (`harness/hang_behind.py`) on a
Google Flights one-way search: with Jev as the decider, the verifier passes every one of Jev's 10 actions and rates
the final state done at 0.97; with V9 as the decider, the model completes the task on its own (10 actions, 33 s, all
7 independent page checks pass, done 0.99). Earlier checkpoints failed this task (V5b declared DONE early, V7 looped
on a filled field).

Text-only decisions (JevBench public items, zero-shot, never trained on text decisions): easy 1.000, standard 0.958,
hard 0.523 with mean confidence 0.91 on the hard tier, i.e. badly over-confident out of domain; see Known limits on
the model card.

## Layout

| path | |
|---|---|
| `model/` | training (`train_vl.py`), evaluation (`eval_schema.py`, `eval_vl.py`), serving (`serve.py`), the packing scheme (`s1/`) |
| `model/runpod/` | the exact scripts each run was trained with (`run_v5.sh`, `run_v6.sh`, bootstrap for a fresh pod) |
| `bench/` | the step-verifier benchmark: items, baselines, scorer, leaderboard builder, pixel-track builder, audit page |
| `harness/` | recording real web episodes with environment labels (`record_triplets.py`), and the closed-loop verifier harness (`hang_behind.py`) |
| `probes/vision/` | data builders (recordings and Mind2Web to training rows), judge baselines (Jev, Claude, DJev) |
| `results/` | every evaluation quoted above, as JSON, plus the closed-loop step logs |
| `probes/`, `data/hume/` | the black-box probes of Jev this repository started as (see below) |

Training data and images are not in the repository (`model/data/vision/` is ignored except the row files); the
benchmark images and the weights are staged on Hugging Face and will be made public with the write-up.

## Model

Qwen3-VL-8B-Instruct with a LoRA (r=64 on attention and MLP projections, 175M trainable) and typed decision heads.
One request is packed as a shared state prefix (screenshot tokens, goal, history, candidate text) followed by one
isolated branch per question under a tree attention mask, so every question is answered in the same forward pass
without seeing the others. `choice` questions read out with a pointer over the option tokens, `noul` with a sigmoid,
`score` over ordered levels. Training is supervised on rows built from recorded episodes (labels from the
environment), Mind2Web-derived schema rows, and OS-Atlas desktop grounding, in stages; the recipe for each stage is
in `model/runpod/`.

Serve and query:

```sh
cd model && python serve.py runs/v5b-8b-state/final --port 8811 --temp 0.5
```

```sh
curl -s localhost:8811/v1/systemone -H 'content-type: application/json' -d '{
  "model": "s1",
  "state": "Task: Find one-way flights from Zurich to London on September 20.\nActions already taken:\n  - click on One way\n",
  "image": "<base64 PNG>",
  "questions": {
    "done": {"type": "noul", "instructions": "Has the goal been fully achieved, with nothing left to do?"}
  }
}'
```

`images` (a list) replaces `image` for two-screenshot questions such as `effect`. `state` may be a string or JSON;
JSON is rendered as indented text. Client code written for Jev can point at this server.

## Benchmark

`bench/README.md` describes the items, the label rules, the two tracks and the metrics. Items are tracked
(`bench/v0_items.jsonl`, `bench/v1_items.jsonl`, `bench/v1_items_pixel.jsonl`); images ship separately. Score a
submission with `bench/eval_submission.py`, rebuild the tables with `bench/leaderboard.py`, and audit the labels by
hand with `bench/audit_tool.py`.

What the benchmark tests is the per-step decision model, not the planning agent: the same items can be answered by a
text-only system from the candidate table, by a vision model from the screenshot, or by a rule.

## Where this started

The repository began as black-box probes of Jev (`probes/k_sweep.py`, `probes/tokenizer_*.py`,
`probes/capability.py`, `probes/counter_vs_model.py`) and an offline reading of the evidence bundle published by
Archer Hume at archerhume.com (`data/hume/`, mirrored with attribution). Those scripts and their results are kept
as they were; the findings are summarised in `results/`.

## Related open efforts

reflex (Qwen3.5-4B re-creation of Jev, same API), DiffusionGemma-as-Jev (vLLM PR 57250) and Jevenator 2,
JevBench (text decisions), DJev. All are prompt-only on general images or text; none is trained on screens or scored
on agent steps with environment labels, which is the gap this repository is about.

## License

Apache-2.0 (see `LICENSE`). Third-party data keeps its own terms: Mind2Web, OS-Atlas, and the Hume evidence files.
