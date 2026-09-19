# Harness

Everything here talks to one HTTP server, `model/serve.py`, which loads a trained decision checkpoint and
answers a TypeSafe-shaped request (state text, optional screenshot, a dict of typed questions) with one
forward pass.

```
# on the GPU box
cd model && python serve.py runs/v2-8b-schema/final --port 8811 --temp 0.5
# locally, tunnel it
ssh -N -L 8811:127.0.0.1:8811 -p <port> root@<host>
```

Request/response:

```
POST /v1/systemone
{"model": "s1", "state": "Task: ...", "image": "<base64 PNG | path | null>",
 "questions": {"ground": {"type": "choice", "instructions": "...", "criteria": {"1": "...", "none": "..."}},
               "final":  {"type": "noul",   "instructions": "After this action, will the task be complete?"}}}
-> {"model": "v2-8b-schema", "answers": {"ground": {"choice": "1", "confidence": 0.98, "probabilities": {...}},
                                          "final": {"noul": 0.72, "confidence": 0.44}}, "usage": {"latency_ms": 150}}
```

- `choose_action_s1.py`: bounded chooser with Cua's jev-use process contract. Reads `cua.jev_choice_request_v1`
  on stdin, writes `cua.jev_choice_v1` on stdout. Optional extensions: `screenshot_path`, extra `questions`.
  `--min-confidence` turns low-confidence picks into `abstain`/`reobserve` when those candidates exist.
- `browser_loop.py`: Playwright decide-act-verify loop. Visible interactive DOM elements become set-of-mark
  candidates; one forward answers ground / act / final. Stops after the action the model calls final, on
  `none`, or below the confidence floor (escalation point).
- `loop_bench.py`: 12 public-page tasks judged by the final URL; writes per-task traces and a summary JSON
  under `results/vision/harness/`.

Question phrasings that match the V2 training schema live in `probes/vision/build_schema_rows.py`
(ACT_VOCAB, FINAL_INSTR, NEEDS_TEXT_INSTR, PROGRESS_*, TAG_*); using them keeps the heads in-distribution.
