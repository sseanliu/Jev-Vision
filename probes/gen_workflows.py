"""Generate diverse synthetic decision workflows (state + typed questions) with
Claude, in our JSONL format, for Jev/open-model labelling.

Why: the 11 public datasets are templated and Jev is >=0.95 confident on 69% of
them, so students never see uncertainty. Here ~40% of questions are designed to
be genuinely ambiguous, option sets vary (2-64 options, none-of-the-above,
near-duplicates, absurd distractors), phrasings vary, and domains cover the
workflows a decision model is actually used for.

python gen_workflows.py --out ../model/data/jsonl_synth/synth.train.jsonl --rows 30000 \
    --model claude-opus-5 --workers 8

Rows carry no gold labels (the generator's own answer is kept under
"designed_answer" for auditing only); labels come from distill_jev.py.
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

DOMAINS = [
    "customer support tickets (SaaS, telecom, banking, e-commerce)", "content moderation of user posts and comments",
    "expense reports and invoices", "HR: job applications, performance notes, leave requests",
    "legal: contract clauses and compliance checks", "clinical intake notes and triage messages",
    "e-commerce product listings and reviews", "incident reports and system logs",
    "chat transcripts between users and agents", "internal emails and meeting notes",
    "insurance claims descriptions", "recruiting: candidate summaries and interview notes",
    "code review comments and bug reports", "real-estate listings and tenant messages",
    "travel bookings, flight changes and refund requests", "education: student submissions and feedback",
    "social media posts for brand safety", "financial transactions and fraud notes",
    "government forms and benefit applications", "logistics: shipment updates and delivery exceptions",
]
QUESTION_STYLES = [
    "route to a team or handler", "assign a priority or severity", "detect a policy violation",
    "decide whether escalation is needed", "extract which of several candidate values is the right one",
    "judge whether a claim is supported by the text", "classify intent", "rate a degree (sentiment, urgency, risk, quality)",
    "choose the next action", "decide whether two things refer to the same entity", "detect whether information is missing",
    "pick the best of several candidate replies",
]
OPTION_PATTERNS = [
    "a small set of 2-4 clearly distinct options", "6-12 options, several of which are near-duplicates",
    "16-40 options like an intent catalogue", "options that include a 'none of the above' or 'cannot tell' item that is sometimes the right answer",
    "options padded with 3-8 obviously irrelevant distractors", "an ordered Score with 3-7 levels described concretely",
    "a yes/no question (noul) whose answer is genuinely uncertain from the text",
]

SYSTEM = """You write training data for a decision model. The model reads a piece of STATE
(a document, message, log, record) and answers typed questions about it:
- choice: pick one option from a keyed set of options (criteria: {key: description}).
- noul: a yes/no judgement about the state.
- score: an ordered set of levels (criteria: [level description, ...]), lowest first.
Produce realistic, specific, varied text. Names, amounts, dates, products and jargon should look real but be fictional.
Return ONLY a JSON object with a single key "items": a list of objects, each:
{"state": str,
 "questions": [{"qid": str, "qtype": "choice"|"noul"|"score", "instructions": str, "criteria": {...}|[...]|null,
                "designed_answer": str|bool|int, "ambiguous": bool}]}
Rules:
- Each item has 1-4 questions about the SAME state, independent of each other.
- About 40% of questions must be genuinely ambiguous or underdetermined by the state (the honest answer is uncertain);
  mark those "ambiguous": true and still give your best designed_answer.
- Vary instruction phrasing; never reuse the same wording twice in one response.
- Option keys are short snake_case; descriptions are full sentences that define the option.
- Score levels must describe concrete situations, not just "low/medium/high".
- No markdown, no commentary, JSON only."""

lock = threading.Lock()


def make_prompt(rng: random.Random, n_items: int) -> str:
    dom = rng.choice(DOMAINS)
    styles = rng.sample(QUESTION_STYLES, 3)
    pattern = rng.choice(OPTION_PATTERNS)
    state_len = rng.choice(["1-2 sentences", "a short paragraph", "2-3 paragraphs", "a long message with several parts"])
    return (f"Domain: {dom}.\nState length: {state_len}.\nQuestion styles to use across the items: {', '.join(styles)}.\n"
            f"Option-set pattern to use for at least half of the choice/score questions: {pattern}.\n"
            f"Generate {n_items} items.")


def validate(item: dict) -> bool:
    if not isinstance(item.get("state"), str) or len(item["state"]) < 20:
        return False
    qs = item.get("questions")
    if not isinstance(qs, list) or not 1 <= len(qs) <= 4:
        return False
    seen = set()
    for q in qs:
        if q.get("qtype") not in ("choice", "noul", "score") or not isinstance(q.get("instructions"), str):
            return False
        if q.get("qid") in seen or not isinstance(q.get("qid"), str):
            return False
        seen.add(q["qid"])
        c = q.get("criteria")
        if q["qtype"] == "choice" and not (isinstance(c, dict) and 2 <= len(c) <= 64 and all(isinstance(v, str) for v in c.values())):
            return False
        if q["qtype"] == "score" and not (isinstance(c, list) and 2 <= len(c) <= 9):
            return False
    return True


def generate_batch(client: anthropic.Anthropic, model: str, rng: random.Random, n_items: int) -> list[dict]:
    prompt = make_prompt(rng, n_items)
    with client.messages.stream(
        model=model,
        max_tokens=16000,
        system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}],
        output_config={"effort": "medium"},
    ) as stream:
        msg = stream.get_final_message()
    if msg.stop_reason == "refusal":
        return []
    text = "".join(b.text for b in msg.content if b.type == "text")
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        return []
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    items = [it for it in data.get("items", []) if validate(it)]
    for it in items:
        it["source"] = "synth"
        it["generator"] = model
        it["prompt_seed"] = prompt
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../model/data/jsonl_synth/synth.train.jsonl")
    ap.add_argument("--rows", type=int, default=30000)
    ap.add_argument("--items-per-call", type=int, default=12)
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    have = sum(1 for _ in open(out)) if out.exists() else 0
    client = anthropic.Anthropic()
    rng = random.Random(a.seed + have)
    n_calls = max(0, (a.rows - have + a.items_per_call - 1) // a.items_per_call)
    print(f"have {have} rows; requesting ~{n_calls} calls x {a.items_per_call} items with {a.model}", flush=True)
    t0 = time.time()
    written, failed = 0, 0
    in_tok = out_tok = 0
    with out.open("a") as f, ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(generate_batch, client, a.model, random.Random(rng.random()), a.items_per_call) for _ in range(n_calls)]
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                items = fut.result()
            except anthropic.APIStatusError as e:
                failed += 1
                print(f"  api error {e.status_code}: {e.message[:80]}", flush=True)
                continue
            if not items:
                failed += 1
            with lock:
                for it in items:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
                written += len(items)
            if k % 20 == 0:
                print(f"  {k}/{n_calls} calls, {written} rows, {failed} empty, {time.time()-t0:.0f}s", flush=True)
    print(f"done: +{written} rows ({failed} empty calls) -> {out} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
