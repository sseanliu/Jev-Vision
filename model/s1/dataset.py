"""JSONL requests -> packed training examples, with augmentations that teach
the behaviours we want: option-order invariance of the *decision*, "none of
the above", distractor padding to large K, and multi-question requests."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from .packing import Packed, Packer, Question

DISTRACTOR_POOL = [
    "A volcanic eruption", "A chess tournament", "The ocean tide", "A ballet recital", "A passing comet",
    "A kite festival", "A street carnival", "A lighthouse", "A wheat harvest", "A penguin colony",
    "A jazz concert", "An origami workshop", "A lantern parade", "A snowman", "A trampoline",
    "A tango lesson", "A windmill", "A double rainbow", "A dinosaur fossil", "A bagpipe band",
]
NONE_KEYS = ["none", "none_of_the_above", "other", "unknown"]
NONE_DESCS = ["None of the above", "None of these options applies", "Something else", "Cannot tell"]


@dataclass
class Example:
    packed: Packed
    targets: list  # per question: int index (choice/score) or float (noul)


class RequestDataset(torch.utils.data.Dataset):
    def __init__(self, paths: list[Path], packer: Packer, max_tokens: int = 2048,
                 augment: bool = True, seed: int = 0, merge_prob: float = 0.3):
        self.rows = []
        for p in paths:
            with open(p) as f:
                self.rows += [json.loads(l) for l in f]
        self.packer = packer
        self.max_tokens = max_tokens
        self.augment = augment
        self.merge_prob = merge_prob
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.rows)

    def _augment_choice(self, q: dict, target: str, rng: random.Random):
        crit = dict(q["criteria"])
        r = rng.random()
        if r < 0.15 and len(crit) >= 3:
            # none-of-the-above: drop the gold option, add a none option as the target
            crit.pop(target)
            key, desc = rng.choice(list(zip(NONE_KEYS, NONE_DESCS)))
            crit[key] = desc
            target = key
        elif r < 0.30:
            # add a none option that is NOT the answer
            key, desc = rng.choice(list(zip(NONE_KEYS, NONE_DESCS)))
            if key not in crit:
                crit[key] = desc
        elif r < 0.45:
            # pad with irrelevant distractors up to a larger K
            k = rng.choice([2, 4, 8, 16])
            for d in rng.sample(DISTRACTOR_POOL, k):
                crit[d.lower().replace(" ", "_")] = d
        elif r < 0.55 and len(crit) > 3:
            # subsample options, keep gold
            keep = [k for k in crit if k != target]
            rng.shuffle(keep)
            keep = keep[: max(1, len(keep) // 2)] + [target]
            crit = {k: crit[k] for k in crit if k in keep}
        # always shuffle option order
        items = list(crit.items())
        rng.shuffle(items)
        return {**q, "criteria": dict(items)}, target

    def build(self, row: dict, rng: random.Random) -> Example:
        questions, targets = [], []
        for q in row["questions"]:
            t = row["targets"][q["qid"]]
            soft = isinstance(t, dict)  # distilled distribution over option keys
            if self.augment and q["qtype"] == "choice" and not soft:
                q, t = self._augment_choice(q, t, rng)
            elif self.augment and q["qtype"] in ("choice", "score") and soft:
                # soft rows: only shuffle option order (keeps the distribution valid)
                items = list(q["criteria"].items()) if q["qtype"] == "choice" else None
                if items is not None:
                    rng.shuffle(items)
                    q = {**q, "criteria": dict(items)}
            if q["qtype"] == "choice":
                qq = Question(q["qid"], "choice", q["instructions"], q["criteria"])
                targets.append([float(t.get(k, 0.0)) for k in q["criteria"]] if soft
                               else list(q["criteria"]).index(t))
            elif q["qtype"] == "noul":
                qq = Question(q["qid"], "noul", q["instructions"])
                targets.append(float(t) if isinstance(t, float) else (1.0 if t else 0.0))
            else:
                qq = Question(q["qid"], "score", q["instructions"], q["criteria"])
                targets.append([float(t.get(str(i), 0.0)) for i in range(len(q["criteria"]))] if soft
                               else int(t))
            questions.append(qq)
        packed = self.packer.pack(row["state"], questions)
        return Example(packed, targets)

    def __getitem__(self, i):
        rng = random.Random(self.rng.random() * 1e9 + i)
        row = self.rows[i]
        ex = self.build(row, rng)
        # Multi-question requests: merge another row's questions if it shares
        # no state (we synthesise by concatenating states with a separator).
        if self.augment and rng.random() < self.merge_prob:
            other = self.rows[rng.randrange(len(self.rows))]
            merged = {"state": row["state"] + "\n\n---\n\n" + other["state"],
                      "questions": [{**q, "qid": "a_" + q["qid"]} for q in row["questions"]]
                      + [{**q, "qid": "b_" + q["qid"]} for q in other["questions"]],
                      "targets": {**{"a_" + k: v for k, v in row["targets"].items()},
                                  **{"b_" + k: v for k, v in other["targets"].items()}}}
            cand = self.build(merged, rng)
            if len(cand.packed.input_ids) <= self.max_tokens:
                ex = cand
        if len(ex.packed.input_ids) > self.max_tokens:
            ex = self._truncate(row, rng)
        return ex

    def _truncate(self, row, rng):
        # crude: cut the state text to fit
        words = row["state"].split()
        for frac in (0.6, 0.4, 0.25, 0.15):
            r2 = {**row, "state": " ".join(words[: int(len(words) * frac)])}
            ex = self.build(r2, rng)
            if len(ex.packed.input_ids) <= self.max_tokens:
                return ex
        return ex


def collate(examples: list[Example], pad_id: int):
    L = max(len(e.packed.input_ids) for e in examples)
    B = len(examples)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    seg = torch.full((B, L), -1, dtype=torch.long)
    pos = torch.zeros((B, L), dtype=torch.long)
    for b, e in enumerate(examples):
        n = len(e.packed.input_ids)
        ids[b, :n] = torch.tensor(e.packed.input_ids)
        seg[b, :n] = torch.tensor(e.packed.segment_ids)
        pos[b, :n] = torch.tensor(e.packed.position_ids)
    return ids, seg, pos, examples
