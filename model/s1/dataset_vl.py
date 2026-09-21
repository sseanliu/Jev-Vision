"""Vision request rows -> packed examples.

Row format (one JSON per line):
  {"image": "<path relative to the jsonl's directory>", "state": "<text shown after the image>",
   "questions": [{"qid", "qtype", "instructions", "criteria"}],
   "targets": {qid: key | {key: p} | bool}}
Option order is NOT shuffled: keys are the numbers drawn on the screenshot.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from .packing import Question
from .vl import PackedVL, VLPacker


@dataclass
class ExampleVL:
    packed: PackedVL
    targets: list


class VLRequestDataset(torch.utils.data.Dataset):
    def __init__(self, paths: list[Path], packer: VLPacker, seed: int = 0, limit: int = 0,
                 budgets: list[int] | None = None):
        """budgets: if given, each example is packed at a pixel budget drawn uniformly from this list
        (multi-resolution training); None keeps the packer's fixed budget."""
        self.budgets = budgets
        self.rng = random.Random(seed + 17)
        self.rows = []
        for p in paths:
            root = Path(p).parent
            with open(p) as f:
                for l in f:
                    r = json.loads(l)
                    if "images" in r:  # V5 rows: list of absolute or relative paths (before, after)
                        r["images"] = [str(p) if str(p).startswith("/") else str(root / p) for p in r["images"]]
                    else:
                        r["image"] = str(root / r["image"])
                    self.rows.append(r)
        if limit:
            self.rows = random.Random(seed).sample(self.rows, min(limit, len(self.rows)))
        self.packer = packer

    def __len__(self):
        return len(self.rows)

    def build(self, row: dict) -> ExampleVL:
        questions, targets = [], []
        for q in row["questions"]:
            t = row["targets"][q["qid"]]
            soft = isinstance(t, dict)
            if q["qtype"] == "choice":
                questions.append(Question(q["qid"], "choice", q["instructions"], q["criteria"]))
                targets.append([float(t.get(k, 0.0)) for k in q["criteria"]] if soft else list(q["criteria"]).index(t))
            elif q["qtype"] == "noul":
                questions.append(Question(q["qid"], "noul", q["instructions"]))
                targets.append(float(t) if isinstance(t, float) else (1.0 if t else 0.0))
            else:
                questions.append(Question(q["qid"], "score", q["instructions"], q["criteria"]))
                targets.append([float(t.get(str(i), 0.0)) for i in range(len(q["criteria"]))] if soft else int(t))
        budget = self.rng.choice(self.budgets) if self.budgets else None
        imgs = row["images"] if "images" in row else row.get("image")
        if not imgs:  # text-only row: pack with the text packer; the model's text forward handles it
            return ExampleVL(self.packer.text.pack(row["state"], questions), targets)
        return ExampleVL(self.packer.pack(imgs, row["state"], questions, max_pixels=budget), targets)

    def __getitem__(self, i):
        return self.build(self.rows[i])
