# Step-verifier benchmark (v0 candidate track; v1 candidate + pixel tracks)

Per-step judgments a computer-use agent loop needs, scored on real web pages with labels derived from the environment.
One item = a goal, the screenshot(s), the proposed or executed action, and one typed question. The thing under test is a
decision model (a fast verifier, a process reward model, or a prompted VLM used per step), not the planning agent.

## Questions

| question | inputs | asks | label source |
|---|---|---|---|
| `ground` | goal, screenshot with numbered marks, candidate table | which numbered element should be acted on next | recorded oracle action |
| `skip` | goal, screenshot, one candidate | is this element already in the state the task needs, so it should be left alone | field value already matches / goal already satisfied before the step |
| `effect` | goal, before + after screenshots, the executed action | did the last action change the page as intended | URL change, DOM diff, field value, pixel diff |
| `done` | goal, current screenshot | has the goal been fully achieved | target page reached (follow), strict search-result rule (search); current state only, not sticky |

Screenshots carry set-of-mark numbers; `elements_before` / `elements_after` are the candidate tables as text
(`element N: tag 'label'`), so text-only systems can be scored on the same items.

## Data

`bench/v0_items.jsonl`: 2,223 items from 30 sites held out from training (site-level split); skip 836, done 836,
effect 404, ground 147. Images (836 PNG, 1280x1000, 263 MB) are in the release bundle `release/v0/images/`
(HF dataset, private staging: `SeanLiu/step-verifier-bench-v0`). Rebuild with:

```
python bench/build_release.py --split rec1b.validation --name v0
```

Item fields: `id`, `question`, `type` (`noul` or `choice`), `instructions`, `criteria` (choice only), `state`
(goal + history + candidate text as the model sees it), `images` (relative paths), `label` (0/1 or criteria key),
`site`, `goal_kind`, `step_id`, `url_before`, `url_after`, `elements_before`, `elements_after`, `candidate`, `which`.

## Scoring

Submit one jsonl line per item: `{"id": ..., "p": <probability of yes>}` for noul questions,
`{"id": ..., "choice": <key>, "p": <probability of the choice>}` for ground, optionally `"ms"` per item.

```
python bench/eval_submission.py --items bench/v0_items.jsonl --pred bench/baselines/v5b.jsonl --name V5b --ms 190
```

Reported per question: accuracy, positive- and negative-class accuracy, ECE (10 bins), AUROC, selective accuracy at
80% and 90% coverage (most confident items kept), and mean ms per decision at a stated hardware budget. Positive
classes are rare for skip (11%) and done (23%), so `acc_pos` is the number to watch; a constant "no" scores 0.89 on
skip and 0.77 on done.

## Baselines shipped

`bench/baselines/*.jsonl` and `bench/v0_leaderboard.md`: Jev 1.13 (text only, 300 per question),
Claude Sonnet 5 with screenshots (verbal probability, 300 per question), and our V5 / V5b 8B decision models
(all items). Zero-shot V3 (no state training) is in `results/vision/v5-8b-state/eval_state_zeroshot_v3.json`.

## v1: two tracks from the same episodes

`bench/v1_items.jsonl` (2,141 items, candidate track) and `bench/v1_items_pixel.jsonl` (2,003 items, pixel track) come
from a fresh recording of the same 30 held-out sites (407 steps) made with `harness/record_triplets.py` after it started
saving unmarked screenshots and element boxes. The pixel track renders each step as the raw screenshot with one red
marker at the action point and no candidate table (`bench/build_pixel_track.py`); ids are `pixel:` + the candidate id.
Images: HF private dataset `SeanLiu/step-verifier-bench-v1` (`images/`, `pixel/`). Baselines: `bench/baselines/v1_*.jsonl`;
leaderboard: `bench/v1_leaderboard.md` (candidate table, then the pixel row). Score a checkpoint on either track with
`bench/items_to_rows.py` + `model/eval_schema.py` (`scripts/pod_eval_tracks.sh`).

## Not in v0

- pixel track: added in v1 (see above); ground on the pixel track still needs a coordinate output
- desktop items (OS-Atlas macOS rows exist for ground only)
- human audit sample of the rule labels
- a normalized latency harness across judges (ms in the table are self-reported: ours on one H100, batch 1, full-res)
