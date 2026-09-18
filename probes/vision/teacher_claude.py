"""V0 teacher test: is Claude a usable *calibrated* vision teacher for `ground` items?

Two confidence channels per item (set-of-mark screenshot + task + numbered candidates):
  verbal : one call asking for a probability over the candidate numbers (SDK 1.x has no temperature knob)
  vote   : --votes calls at default sampling asking for the single best number; distribution = votes
Reports accuracy, ECE, and AUROC (top-p vs correctness) for each channel.

python teacher_claude.py --items ../../model/data/vision/m2w_items --model claude-sonnet-5 --votes 5 --limit 300
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

SYSTEM = ("You are grounding a web task to a UI element. The screenshot has numbered boxes drawn on candidate "
          "elements. Given the task and the actions already taken, decide which numbered element must be acted on "
          "next. If none of the boxed elements is right, answer none.")


def prompt(item: dict) -> str:
    hist = "\n".join(f"  - {h}" for h in item["history"]) or "  (none)"
    cands = "\n".join(f"  {k}: {v}" for k, v in item["criteria"].items())
    return (f"Task: {item['task']}\nActions already taken:\n{hist}\nCandidates:\n{cands}\n")


def image_block(path: Path) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                        "data": base64.b64encode(path.read_bytes()).decode()}}


def ask(client, model, item, root, mode):
    keys = list(item["criteria"])
    if mode == "verbal":
        tail = ("Respond with a JSON object mapping each candidate key to your probability that it is the correct "
                "target; probabilities must sum to 1. JSON only.")
    else:
        tail = "Respond with the single best candidate key only (a number or none)."
    msg = client.messages.create(
        model=model, max_tokens=300, system=SYSTEM,
        messages=[{"role": "user", "content": [image_block(root / item["image"]),
                                               {"type": "text", "text": prompt(item) + tail}]}])
    text = "".join(b.text for b in msg.content if b.type == "text").strip()
    if mode == "verbal":
        m = re.search(r"\{.*\}", text, re.S)
        try:
            d = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            d = {}
        p = {k: max(0.0, float(d.get(k, 0.0))) for k in keys}
        s = sum(p.values()) or 1.0
        return {k: v / s for k, v in p.items()}, msg.usage
    m = re.search(r"\b(none|\d+)\b", text.lower())
    return (m.group(1) if m else "none"), msg.usage


def metrics(rows):
    n = len(rows)
    acc = sum(r["hit"] for r in rows) / n
    bins = [[] for _ in range(10)]
    for r in rows:
        bins[min(9, int(r["conf"] * 10))].append(r)
    ece = sum(len(b) / n * abs(sum(x["hit"] for x in b) / len(b) - sum(x["conf"] for x in b) / len(b)) for b in bins if b)
    pos = [r["conf"] for r in rows if r["hit"]]; neg = [r["conf"] for r in rows if not r["hit"]]
    auroc = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg)) if pos and neg else float("nan")
    return {"n": n, "acc": round(acc, 3), "ece": round(ece, 3), "auroc": round(auroc, 3),
            "mean_top_p": round(sum(r["conf"] for r in rows) / n, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="../../model/data/vision/m2w_items")
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--votes", type=int, default=5)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="../../results/vision/teacher_claude.json")
    a = ap.parse_args()
    root = Path(a.items)
    items = [json.loads(l) for l in open(root / "items.jsonl")][: a.limit]
    client = anthropic.Anthropic(timeout=120.0, max_retries=3)
    usage = Counter(); lock = threading.Lock()
    per_item = {}

    def work(item):
        verbal, u = ask(client, a.model, item, root, "verbal")
        with lock:
            usage["in"] += u.input_tokens; usage["out"] += u.output_tokens
        votes = Counter()
        for _ in range(a.votes):
            v, u = ask(client, a.model, item, root, "vote")
            votes[v] += 1
            with lock:
                usage["in"] += u.input_tokens; usage["out"] += u.output_tokens
        vote_p = {k: votes.get(k, 0) / a.votes for k in item["criteria"]}
        return item["id"], {"gold": item["gold"], "verbal": verbal, "vote": vote_p}

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(work, it) for it in items]
        for k, f in enumerate(as_completed(futs), 1):
            try:
                iid, res = f.result()
                per_item[iid] = res
            except Exception as e:  # keep going; report at the end
                print(f"  error {type(e).__name__}: {str(e)[:100]}", flush=True)
            if k % 25 == 0:
                print(f"  {k}/{len(items)} items, {time.time()-t0:.0f}s, in={usage['in']/1e6:.2f}M out={usage['out']/1e3:.0f}k", flush=True)
    report = {"model": a.model, "votes": a.votes, "usage": dict(usage)}
    for ch in ("verbal", "vote"):
        rows = []
        for iid, r in per_item.items():
            p = r[ch]; top = max(p, key=p.get)
            rows.append({"conf": p[top], "hit": top == r["gold"]})
        report[ch] = metrics(rows)
        print(ch, report[ch])
    # agreement between the two channels' argmax
    agree = sum(max(r["verbal"], key=r["verbal"].get) == max(r["vote"], key=r["vote"].get) for r in per_item.values())
    report["argmax_agreement"] = round(agree / max(1, len(per_item)), 3)
    print("argmax agreement verbal vs vote:", report["argmax_agreement"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"report": report, "items": per_item}, indent=1))
    print(f"saved {a.out}; cost ~${(usage['in']*3 + usage['out']*15)/1e6:.2f} at Sonnet pricing")


if __name__ == "__main__":
    main()
