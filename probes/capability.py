"""Capability and calibration baseline for version tracking.

Families (all Choice questions with four numeric or short options):
- bob:       the two-games-in-a-row puzzle, original wording and a paraphrase,
             10 repeats each (multi-step reasoning without CoT)
- word_2step: 25 generated two-step word problems
- modexp:     25 generated a^b mod m problems (Hume's underconfident family)
- mult3x3:    25 generated three-digit products (control)

Records per item: correct?, top probability, full distribution. Reports
accuracy vs mean top-probability per family, as in Hume's fresh-math study.
Output is keyed by model version so later runs can be diffed.
"""

from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timezone

from common import ROOT, call, save

SEED = 20260917
BOB = [
    ("Bob wins a prize if he wins two games in a row. He must play three games, "
     "alternating opponents. Option A: Teacher, Student, Teacher. Option B: "
     "Student, Teacher, Student. Everybody knows the teacher is the better "
     "player. Which option gives Bob the better chance at the prize?",
     {"A": "Teacher, Student, Teacher", "B": "Student, Teacher, Student"}, "A"),
    ("To win, Mia needs two consecutive wins out of three matches against two "
     "opponents in alternation. Her sister is much stronger than her friend. "
     "Schedule X: sister, friend, sister. Schedule Y: friend, sister, friend. "
     "Which schedule maximises her chance of two consecutive wins?",
     {"X": "sister, friend, sister", "Y": "friend, sister, friend"}, "X"),
]


def gen_word_2step(rng):
    items = []
    for _ in range(25):
        a, b, c = rng.randint(12, 60), rng.randint(3, 9), rng.randint(15, 90)
        kind = rng.choice(["buy_then_sell", "boxes_then_give", "rate_then_left"])
        if kind == "buy_then_sell":
            ans = a * b - c
            q = (f"A shop buys {a} lamps at ${b} each, then sells them all for a "
                 f"total that is ${c} less than what it paid. How many dollars did "
                 f"the shop receive?")
        elif kind == "boxes_then_give":
            ans = a * b + c
            q = (f"Ana packs {a} boxes with {b} pears each and then finds {c} "
                 f"loose pears. How many pears does she have in total?")
        else:
            ans = c * b - a
            q = (f"A tap fills {c} litres per minute for {b} minutes; then {a} "
                 f"litres are drained. How many litres remain?")
        items.append((q, ans))
    return items


def gen_modexp(rng):
    items = []
    for _ in range(25):
        a, b, m = rng.randint(2, 9), rng.randint(3, 12), rng.choice([7, 11, 13, 17, 19, 23])
        items.append((f"What is {a}^{b} mod {m}?", pow(a, b, m)))
    return items


def gen_mult(rng):
    items = []
    for _ in range(25):
        a, b = rng.randint(100, 999), rng.randint(100, 999)
        items.append((f"What is {a} x {b}?", a * b))
    return items


def distractors(rng, ans: int, spread: int) -> list[int]:
    out = set()
    while len(out) < 3:
        d = ans + rng.choice([-1, 1]) * rng.randint(1, max(2, spread))
        if d != ans and d >= 0:
            out.add(d)
    return list(out)


def numeric_question(rng, text: str, ans: int, spread: int):
    opts = distractors(rng, ans, spread) + [ans]
    rng.shuffle(opts)
    criteria = {f"o{i}": str(v) for i, v in enumerate(opts)}
    correct = next(k for k, v in criteria.items() if v == str(ans))
    payload = {"model": "jev-latest", "state": text,
               "questions": {"a": {"type": "choice",
                                   "instructions": "Which option is the correct answer to the question in the state?",
                                   "criteria": criteria}}}
    return payload, correct


def run_item(payload, correct):
    res = call(payload)
    if res["status"] != 200:
        return {"status": res["status"]}
    ans = res["body"]["answers"]["a"]
    return {"status": 200, "choice": ans["choice"], "correct_key": correct,
            "is_correct": ans["choice"] == correct,
            "top_prob": ans["probabilities"][ans["choice"]],
            "p_correct": ans["probabilities"][correct],
            "probabilities": ans["probabilities"], "confidence": ans["confidence"],
            "model": res["body"]["model"], "server_ms": res.get("server_ms")}


def main() -> None:
    rng = random.Random(SEED)
    families = {}
    items = []
    for text, crit, correct in BOB:
        payload = {"model": "jev-latest", "state": text,
                   "questions": {"a": {"type": "choice",
                                       "instructions": "Which option should be chosen?",
                                       "criteria": crit}}}
        for rep in range(10):
            items.append(("bob", payload, correct, rep))
    for fam, gen, spread in (("word_2step", gen_word_2step, 40),
                             ("modexp", gen_modexp, 6),
                             ("mult3x3", gen_mult, 900)):
        for text, ans in gen(rng):
            payload, correct = numeric_question(rng, text, ans, spread)
            items.append((fam, payload, correct, 0))
    results = []
    model = None
    for i, (fam, payload, correct, rep) in enumerate(items, 1):
        r = run_item(payload, correct)
        r.update({"family": fam, "repeat": rep, "state": payload["state"],
                  "criteria": payload["questions"]["a"]["criteria"]})
        results.append(r)
        model = r.get("model", model)
        print(f"[{i:3d}/{len(items)}] {fam:10s} ok={r.get('is_correct')} "
              f"top={r.get('top_prob')} p_correct={r.get('p_correct')}", flush=True)
    out = {"schema": "jev-capability.v1", "model": model,
           "date": datetime.now(timezone.utc).isoformat(), "seed": SEED,
           "results": results}
    path = save(f"capability_{model}.json", out)
    print("saved", path)
    summarize(results, model)
    families.clear()


def summarize(results, model) -> None:
    print(f"\nCapability baseline for {model}")
    print(f"{'family':12s} {'n':>4s} {'accuracy':>9s} {'mean top-p':>11s} {'mean p(correct)':>16s}")
    for fam in ("bob", "word_2step", "modexp", "mult3x3"):
        rs = [r for r in results if r["family"] == fam and r["status"] == 200]
        if not rs:
            continue
        acc = sum(r["is_correct"] for r in rs) / len(rs)
        top = sum(r["top_prob"] for r in rs) / len(rs)
        pc = sum(r["p_correct"] for r in rs) / len(rs)
        print(f"{fam:12s} {len(rs):4d} {acc:9.2f} {top:11.2f} {pc:16.2f}")
    bob = [r for r in results if r["family"] == "bob" and r["status"] == 200]
    if bob:
        for variant in (0, 1):
            rs = bob[variant * 10:(variant + 1) * 10]
            if rs:
                print(f"  bob variant {variant}: chose correct {sum(r['is_correct'] for r in rs)}/10, "
                      f"mean p(correct)={sum(r['p_correct'] for r in rs)/len(rs):.2f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summarize":
        data = json.loads((ROOT / "results" / sys.argv[2]).read_text())
        summarize(data["results"], data["model"])
    else:
        main()
