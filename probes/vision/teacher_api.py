"""Teacher probe across API providers (verbal-probability channel, optional votes).

Same items and metrics as teacher_claude.py; adds Gemini so candidates can be compared on the
identical 300 Mind2Web ground items.

python teacher_api.py --provider gemini --model gemini-3.8-flash --limit 300 --votes 0
python teacher_api.py --provider anthropic --model claude-sonnet-5 --limit 300 --votes 0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from teacher_claude import SYSTEM, image_block, metrics, prompt

VERBAL_TAIL = ("Respond with a JSON object mapping each candidate key to your probability that it is the correct "
               "target; probabilities must sum to 1. JSON only.")
VOTE_TAIL = "Respond with the single best candidate key only (a number or none)."


def parse_verbal(text: str, keys: list[str]) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    p = {k: max(0.0, float(d.get(k, 0.0) or 0.0)) for k in keys}
    s = sum(p.values()) or 1.0
    return {k: v / s for k, v in p.items()}


def parse_vote(text: str) -> str:
    m = re.search(r"\b(none|\d+)\b", text.lower())
    return m.group(1) if m else "none"


class Anthropic:
    def __init__(self, model, thinking: bool = False, effort: str | None = None):
        import anthropic
        self.c = anthropic.Anthropic(timeout=300.0, max_retries=3); self.model = model
        self.extra = {}
        if thinking:
            self.extra["thinking"] = {"type": "adaptive"}
        if effort:
            self.extra["output_config"] = {"effort": effort}

    def call(self, png: Path, text: str):
        with self.c.messages.stream(model=self.model, max_tokens=8000 if self.extra.get("thinking") else 300, system=SYSTEM,
                                    messages=[{"role": "user", "content": [image_block(png), {"type": "text", "text": text}]}],
                                    **self.extra) as stream:
            msg = stream.get_final_message()
        return "".join(b.text for b in msg.content if b.type == "text"), msg.usage.input_tokens, msg.usage.output_tokens


class Gemini:
    def __init__(self, model, thinking_level: str | None):
        from google import genai
        from google.genai import types
        self.types = types
        self.c = genai.Client(api_key=os.environ["GEMINI_API_KEY"]); self.model = model
        self.thinking = types.ThinkingConfig(thinking_level=thinking_level) if thinking_level else None

    def call(self, png: Path, text: str):
        t = self.types
        cfg = t.GenerateContentConfig(system_instruction=SYSTEM, max_output_tokens=8000, thinking_config=self.thinking)
        r = self.c.models.generate_content(model=self.model, config=cfg,
                                           contents=[t.Part.from_bytes(data=png.read_bytes(), mime_type="image/png"), text])
        u = r.usage_metadata
        return (r.text or ""), (u.prompt_token_count or 0), ((u.candidates_token_count or 0) + (u.thoughts_token_count or 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="../../model/data/vision/m2w_items")
    ap.add_argument("--provider", choices=["anthropic", "gemini"], default="gemini")
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--thinking-level", default=None, help="gemini: low|medium|high (default: model default)")
    ap.add_argument("--thinking", action="store_true", help="anthropic: adaptive thinking")
    ap.add_argument("--effort", default=None, help="anthropic: low|medium|high|max output effort")
    ap.add_argument("--votes", type=int, default=0)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--price", default="0.75,3.75", help="$/1M in,out for the cost line")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.items)
    items = [json.loads(l) for l in open(root / "items.jsonl")][: a.limit]
    be = Anthropic(a.model, a.thinking, a.effort) if a.provider == "anthropic" else Gemini(a.model, a.thinking_level)
    usage = Counter(); lock = threading.Lock(); per_item = {}; errors = Counter()

    def work(item):
        keys = list(item["criteria"]); png = root / item["image"]
        text, i, o = be.call(png, prompt(item) + VERBAL_TAIL)
        with lock:
            usage["in"] += i; usage["out"] += o
        res = {"gold": item["gold"], "verbal": parse_verbal(text, keys), "raw": text[:300]}
        if a.votes:
            votes = Counter()
            for _ in range(a.votes):
                t, i, o = be.call(png, prompt(item) + VOTE_TAIL); votes[parse_vote(t)] += 1
                with lock:
                    usage["in"] += i; usage["out"] += o
            res["vote"] = {k: votes.get(k, 0) / a.votes for k in keys}
        return item["id"], res

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(work, it) for it in items]
        for k, f in enumerate(as_completed(futs), 1):
            try:
                iid, res = f.result(); per_item[iid] = res
            except Exception as e:
                errors[type(e).__name__] += 1
                print(f"  error {type(e).__name__}: {str(e)[:120]}", flush=True)
            if k % 50 == 0:
                print(f"  {k}/{len(items)} items, {time.time()-t0:.0f}s, in={usage['in']/1e6:.2f}M out={usage['out']/1e3:.0f}k", flush=True)
    pi, po = (float(x) for x in a.price.split(","))
    report = {"provider": a.provider, "model": a.model, "thinking_level": a.thinking_level, "votes": a.votes,
              "usage": dict(usage), "errors": dict(errors), "cost_usd": round((usage["in"] * pi + usage["out"] * po) / 1e6, 2)}
    empty = sum(1 for r in per_item.values() if max(r["verbal"].values()) == 0)
    for ch in (["verbal", "vote"] if a.votes else ["verbal"]):
        rows = []
        for r in per_item.values():
            p = r[ch]; top = max(p, key=p.get); rows.append({"conf": p[top], "hit": top == r["gold"]})
        report[ch] = metrics(rows); print(ch, report[ch])
    print(f"unparseable verbal answers: {empty}; errors: {dict(errors)}; cost ~${report['cost_usd']}")
    out = Path(a.out or f"../../results/vision/teacher_{a.model}_m2w{len(items)}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"report": report, "items": per_item}, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
