"""Escalation curve: student answers when confident, defers to a teacher otherwise.

For thresholds tau on the student's top-p (after temperature T), report the fraction of items
escalated, the combined accuracy (student on kept items, teacher on escalated ones), and the
mean latency / cost per decision. This is the product mechanism the calibration work pays for.

python escalation.py --student ../../results/vision/v1b-8b-m2w-jitter/eval_v0b.json --temp 0.5 \
    --teacher qwen32b=../../results/vision/v1b-8b-m2w-jitter/teacher_qwen3vl32b_m2w300_j.json:logprob:250:0.001 \
    --teacher claude=...:verbal:2500:0.0034
teacher spec: name=path:channel:latency_ms:cost_usd
"""

from __future__ import annotations

import argparse
import json
import math


def softmax(z, T):
    m = max(z); e = [math.exp((v - m) / T) for v in z]; s = sum(e); return [v / s for v in e]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--student-ms", type=float, default=153)
    ap.add_argument("--student-cost", type=float, default=0.0002, help="USD per decision (H100 at $3.5/h, ~150 ms)")
    ap.add_argument("--teacher", action="append", default=[])
    ap.add_argument("--taus", default="0.0,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.01")
    a = ap.parse_args()
    st = json.load(open(a.student))["items"]
    student = {}
    for k, r in st.items():
        p = softmax(r["logits"], a.temp); top = max(range(len(p)), key=lambda i: p[i])
        student[k] = {"conf": p[top], "hit": r["keys"][top] == r["gold"]}
    teachers = {}
    for spec in a.teacher:
        name, rest = spec.split("=", 1); path, ch, ms, cost = rest.rsplit(":", 3)
        items = json.load(open(path))["items"]
        teachers[name] = {"ms": float(ms), "cost": float(cost),
                          "hit": {k: max(v[ch], key=v[ch].get) == v["gold"] for k, v in items.items() if k in student}}
    ids = list(student)
    print(f"student alone: acc {sum(student[k]['hit'] for k in ids)/len(ids):.3f}  ({a.student_ms:.0f} ms, ${a.student_cost:.4f})")
    for name, t in teachers.items():
        print(f"{name} alone: acc {sum(t['hit'][k] for k in ids)/len(ids):.3f}  ({t['ms']:.0f} ms, ${t['cost']:.4f})")
        print(f"  tau    escalated   acc   mean_ms   mean_$")
        for tau in (float(x) for x in a.taus.split(",")):
            esc = [k for k in ids if student[k]["conf"] < tau]
            acc = (sum(t["hit"][k] for k in esc) + sum(student[k]["hit"] for k in ids if k not in set(esc))) / len(ids)
            frac = len(esc) / len(ids)
            ms = a.student_ms + frac * t["ms"]; cost = a.student_cost + frac * t["cost"]
            print(f"  {tau:4.2f}   {frac:8.2f}   {acc:.3f}   {ms:7.0f}   {cost:.4f}")
        # oracle: escalate exactly the student's errors
        wrong = [k for k in ids if not student[k]["hit"]]
        acc_o = (sum(t["hit"][k] for k in wrong) + (len(ids) - len(wrong))) / len(ids)
        print(f"  oracle (escalate only errors, {len(wrong)/len(ids):.2f}): acc {acc_o:.3f}")


if __name__ == "__main__":
    main()
