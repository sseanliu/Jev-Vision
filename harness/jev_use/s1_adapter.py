"""Provider adapter for our open-weight decision model served by `model/serve.py` (typesafe repo).

Mirrors the TypeSafe SDK surface that `jev_adapter.choose_with_typesafe` uses (`client.system_one(state=..., questions={qid: Choice})`
-> `response.choices[qid].choice / .confidence / .probabilities`, `response.model`), so the example's request construction and
candidate validation stay untouched: only the provider changes.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping

from core import Candidate, VisualObservation
from jev_adapter import choose_with_typesafe


@dataclass(frozen=True)
class _Answer:
    choice: str
    confidence: float
    probabilities: dict[str, float]


@dataclass(frozen=True)
class _Response:
    choices: dict[str, _Answer]
    model: str | None


class S1Client:
    def __init__(self, server: str | None = None, screenshot_path: str | None = None):
        self.server = (server or os.environ.get("S1_SERVER", "http://127.0.0.1:8811")).rstrip("/")
        self.screenshot_path = screenshot_path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def system_one(self, *, state: Any, questions: Mapping[str, Any]) -> _Response:
        qs = {}
        for qid, q in questions.items():
            crit = dict(getattr(q, "criteria", {}) or {})
            qs[qid] = {"type": "choice", "instructions": getattr(q, "instructions", "") or "", "criteria": crit}
        payload = {"model": "s1", "state": state, "image": self.screenshot_path, "questions": qs}
        req = urllib.request.Request(self.server + "/v1/systemone", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        res = json.loads(urllib.request.urlopen(req, timeout=120).read())
        if "error" in res:
            raise RuntimeError(res["error"])
        choices = {qid: _Answer(a["choice"], float(a["confidence"]), {k: float(v) for k, v in a["probabilities"].items()})
                   for qid, a in res["answers"].items()}
        return _Response(choices, res.get("model"))


def choose_s1(
    candidates: list[Candidate],
    snapshot: dict[str, Any],
    visual: VisualObservation | None,
    history: list[dict[str, Any]],
) -> tuple[str, float, dict[str, float]]:
    with S1Client() as client:
        return choose_with_typesafe(client, candidates, snapshot, visual, history)
