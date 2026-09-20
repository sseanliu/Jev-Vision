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

This repository is updated as results land. Numbers below are from the current checkpoint (V5b); a V6 that also
trains the decision head in the harness's own question format is in progress. Status and plans are tracked in the
commit log.

## Results

Benchmark v1: fresh episodes on 30 web sites held out from training (site-level split), labels from the environment.
Accuracy per question with positive / negative class accuracy in brackets, expected calibration error, and latency.

Candidate track (screenshot with numbered marks plus a text table of the candidate elements):

| system | skip | effect | done | ECE skip/effect/done | ms |
|---|---|---|---|---|---|
| Jev 1.13 (text only, candidate table) | 0.870 (0.55/0.92) | 0.733 (0.66/0.98) | 0.887 (0.89/0.89) | 0.105/0.096/0.049 | 238 (API p50) |
| Claude Sonnet 5 (screenshots, verbal probability) | 0.850 (0.31/0.93) | 0.477 (0.33/0.94) | 0.780 (0.35/0.93) | 0.119/0.301/0.105 | n/a |
| rules (URL/DOM change for effect, constant no) | 0.865 | 0.892 | 0.758 | 0.035/0.008/0.142 | 0 |
| **ours, V5b 8B** | **0.936** (0.79/0.96) | **0.967** (0.97/0.96) | **0.896** (0.83/0.92) | 0.042/0.020/0.055 | **157** |

Pixel track (raw screenshot with one marker on the candidate, no table; the same episodes):

| system | skip | effect | done | ECE | ms |
|---|---|---|---|---|---|
| ours, V5b 8B, zero-shot on this rendering | 0.931 (0.89/0.94) | 0.908 (0.97/0.69) | 0.902 (0.88/0.91) | 0.039/0.080/0.046 | 182 |

Benchmark v0 (2,223 items, same 30 sites, earlier episodes): skip 0.953, effect 0.941, done 0.932, ground 0.952 for
V5b; full tables with AUROC and selective accuracy in `bench/v0_leaderboard.md` and `bench/v1_leaderboard.md`.
Latency is one H100, batch 1, 1280x1000 screenshots, all questions of a step in one forward.

Closed loop (`harness/hang_behind.py`): with Jev as the decider on a Google Flights task, the verifier passes every
one of Jev's 10 actions and rates the final state done at 0.97; the task's independent page checks all pass. With
our model as the decider it makes 6 correct moves and then stops early, which is what V6 addresses (see
`probes/vision/build_operation_rows.py`).

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
