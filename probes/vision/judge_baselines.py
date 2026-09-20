"""Baselines for the state questions (skip / effect / done) on V5 rows.

  jev     TypeSafe Jev, text only: state text + the candidate table(s) as text, one Noul question
  claude  Anthropic model with the screenshot(s), asked for a probability (verbal)
  gemini  same with Gemini
  djev    DJev (djev.dev, DiffusionGemma-as-Jev): Jev-shaped request with the screenshot attached (before/after stacked
          into one image for effect rows); needs DJEV_API_KEY in .env
Reports per qid: n, acc (p>0.5), AUROC, ECE, plus acc on the positive and negative class (the "should not act"
half is the negative class for effect and the positive class for skip/done).

python judge_baselines.py --rows ../../model/data/vision/state_rows/rec1.validation.jsonl --judge jev --limit 300
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_env():
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())


def text_state(row):
    m = row.get("meta", {}); s = row["state"]
    if row["questions"][0]["qid"] == "effect":
        s += f"\nPage before the action:\n{m.get('text_before','')}\nURL before: {m.get('url_before','')}\nPage after the action:\n{m.get('text_after','')}\nURL after: {m.get('url_after','')}\n"
    else:
        which = m.get("which", "before"); s += f"\nCurrent page elements:\n{m.get('text_after' if which == 'after' else 'text_before','')}\nURL: {m.get('url_after' if which == 'after' else 'url_before','')}\n"
    return s


def judge_jev(row):
    from typesafe_sdk import Noul, TypeSafeClient
    q = row["questions"][0]
    with TypeSafeClient(timeout=60.0) as c:
        r = c.system_one(state=text_state(row), questions={q["qid"]: Noul(instructions=q["instructions"])})
    return float(r.answers[q["qid"]].noul)


def _img_block(path):
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(Path(path).read_bytes()).decode()}}


def judge_claude(row, model):
    import anthropic
    q = row["questions"][0]; labels = ["before the action", "after the action"] if len(row["images"]) == 2 else ["current screen"]
    content = []
    for lab, p in zip(labels, row["images"]):
        content += [{"type": "text", "text": f"Screenshot {lab}:"}, _img_block(p)]
    content.append({"type": "text", "text": row["state"] + f"\nQuestion: {q['instructions']}\nAnswer with a single number: the probability (0 to 1) that the answer is yes. Number only."})
    c = anthropic.Anthropic(timeout=120.0, max_retries=3)
    r = c.messages.create(model=model, max_tokens=20, messages=[{"role": "user", "content": content}])
    txt = "".join(b.text for b in r.content if b.type == "text"); m = re.search(r"[01](?:\.\d+)?|\.\d+", txt)
    return float(m.group(0)) if m else 0.5


def judge_gemini(row, model):
    from google import genai
    from google.genai import types
    q = row["questions"][0]; c = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    parts = []
    labels = ["before the action", "after the action"] if len(row["images"]) == 2 else ["current screen"]
    for lab, p in zip(labels, row["images"]):
        parts += [f"Screenshot {lab}:", types.Part.from_bytes(data=Path(p).read_bytes(), mime_type="image/png")]
    parts.append(row["state"] + f"\nQuestion: {q['instructions']}\nAnswer with a single number: the probability (0 to 1) that the answer is yes. Number only.")
    r = c.models.generate_content(model=model, contents=parts, config=types.GenerateContentConfig(max_output_tokens=20))
    m = re.search(r"[01](?:\.\d+)?|\.\d+", r.text or ""); return float(m.group(0)) if m else 0.5


def _data_url(png_bytes):
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode()


def _djev_image(row):
    """DJev takes one state image (<=2048x2048). For effect rows stack before/after vertically with labels."""
    from io import BytesIO
    from PIL import Image, ImageDraw
    ims = [Image.open(p).convert("RGB") for p in row["images"]]
    if len(ims) == 1:
        im = ims[0]
    else:
        w = max(i.width for i in ims); band = 28
        im = Image.new("RGB", (w, sum(i.height for i in ims) + band * len(ims)), "white"); y = 0
        for lab, i in zip(["BEFORE the action", "AFTER the action"], ims):
            ImageDraw.Draw(im).text((8, y + 6), lab, fill="red"); y += band; im.paste(i, (0, y)); y += i.height
    if max(im.size) > 2048:
        im.thumbnail((2048, 2048))
    buf = BytesIO(); im.save(buf, format="PNG"); return _data_url(buf.getvalue())


def djev_payload(row):
    q = row["questions"][0]
    state = row["state"]
    if len(row["images"]) == 2:
        state += "\nThe attached image shows the screen BEFORE the action (top) and AFTER the action (bottom)."
    return {"model": "djev", "state": state, "images": [_djev_image(row)],
            "questions": {q["qid"]: {"type": "noul", "instructions": q["instructions"]}}, "options": {"seed": 0}}


def judge_djev(row, base_url="https://api.djev.dev/v1/request"):
    """DJev (DiffusionGemma-as-Jev hosted API, djev.dev): Jev-shaped request with the screenshot attached."""
    import secrets
    import requests
    key = os.environ.get("DJEV_API_KEY")
    if not key:
        raise RuntimeError("DJEV_API_KEY missing (add it to .env)")
    r = requests.post(base_url, json=djev_payload(row), timeout=120,
                      headers={"Authorization": f"Bearer {key}", "X-Djev-Operation-Id": secrets.token_hex(24)})
    r.raise_for_status(); body = r.json()
    q = row["questions"][0]["qid"]; ans = body.get("answers", body).get(q, {})
    for k in ("noul", "yes", "probability", "value", "p"):
        if isinstance(ans.get(k), (int, float)):
            return float(ans[k])
    if isinstance(ans.get("probabilities"), dict):
        return float(ans["probabilities"].get("true", ans["probabilities"].get("yes", 0.5)))
    raise ValueError(f"unrecognised DJev answer: {json.dumps(ans)[:200]}")


def metrics(rows):
    n = len(rows); acc = sum((r["p"] > 0.5) == bool(r["y"]) for r in rows) / n
    bins = [[] for _ in range(10)]
    for r in rows:
        conf = r["p"] if r["p"] > 0.5 else 1 - r["p"]; hit = (r["p"] > 0.5) == bool(r["y"]); bins[min(9, int(conf * 10))].append((conf, hit))
    ece = sum(len(b) / n * abs(sum(h for _, h in b) / len(b) - sum(c for c, _ in b) / len(b)) for b in bins if b)
    pos = [r["p"] for r in rows if r["y"]]; neg = [r["p"] for r in rows if not r["y"]]
    auroc = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg)) if pos and neg else float("nan")
    acc_pos = sum(r["p"] > 0.5 for r in rows if r["y"]) / max(1, len(pos)); acc_neg = sum(r["p"] <= 0.5 for r in rows if not r["y"]) / max(1, len(neg))
    return {"n": n, "acc": round(acc, 3), "auroc": round(auroc, 3), "ece": round(ece, 3), "acc_pos": round(acc_pos, 3), "acc_neg": round(acc_neg, 3), "n_pos": len(pos)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True); ap.add_argument("--judge", choices=["jev", "claude", "gemini", "djev"], required=True)
    ap.add_argument("--model", default=None); ap.add_argument("--limit", type=int, default=300, help="rows per qid")
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--out", default=None)
    a = ap.parse_args(); load_env()
    model = a.model or {"claude": "claude-sonnet-5", "gemini": "gemini-3.8-flash", "jev": None, "djev": "djev"}[a.judge]
    rows = [json.loads(l) for l in open(a.rows)]
    by = defaultdict(list)
    for r in rows:
        if r["questions"][0]["qtype"] == "noul":
            by[r["questions"][0]["qid"]].append(r)
    sel = [r for q, rs in by.items() for r in rs[: a.limit]]
    fn = {"jev": lambda r: judge_jev(r), "claude": lambda r: judge_claude(r, model), "gemini": lambda r: judge_gemini(r, model),
          "djev": lambda r: judge_djev(r)}[a.judge]
    res = defaultdict(list); errors = 0; lock = threading.Lock(); t0 = time.time()

    def work(r):
        t = time.perf_counter(); p = fn(r)
        return r, p, round((time.perf_counter() - t) * 1000)

    with ThreadPoolExecutor(a.workers) as ex:
        for k, f in enumerate(as_completed([ex.submit(work, r) for r in sel]), 1):
            try:
                r, p, ms = f.result(); qid = r["questions"][0]["qid"]
                with lock:
                    res[qid].append({"p": p, "y": r["targets"][qid], "ms": ms, "meta": r.get("meta", {})})
            except Exception as e:
                errors += 1; print("  error", type(e).__name__, str(e)[:100], flush=True)
            if k % 100 == 0:
                print(f"  {k}/{len(sel)} {time.time()-t0:.0f}s", flush=True)
    report = {qid: metrics(rs) for qid, rs in res.items()}; report["_judge"] = a.judge; report["_model"] = model; report["_errors"] = errors
    allms = sorted(x["ms"] for rs in res.values() for x in rs)
    if allms:
        report["_ms_p50"] = allms[len(allms) // 2]; report["_ms_mean"] = round(sum(allms) / len(allms))
    print(json.dumps(report, indent=1))
    out = Path(a.out or ROOT / "results" / "state" / f"judge_{a.judge}_{Path(a.rows).stem}.json"); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"report": report, "rows": {q: rs for q, rs in res.items()}}, indent=1)); print("saved", out)


if __name__ == "__main__":
    main()
