"""Decide-act-verify browser loop with Playwright and the s1 decision server.

Each step: screenshot the viewport, collect visible interactive DOM elements with bounding boxes,
draw numbered set-of-mark boxes, ask the server one forward with three typed questions
(ground: which element; act: what to do; done: is the goal achieved), execute the chosen action,
repeat. Typed text comes from a tiny rule (value supplied per task) — the model only decides
*where* and *what kind*. Logs every decision with confidence and latency.

python harness/browser_loop.py --url https://example.com --goal "Open the 'More information' link" \
    --server http://127.0.0.1:8811 --max-steps 4 --log runs/loop.jsonl
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

COLORS = ["#e6194b", "#3cb44b", "#0082c8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#d2f53c",
          "#fabebe", "#008080", "#aa6e28", "#800000", "#808000", "#000080", "#808080", "#000000"]
JS_ELEMENTS = """
() => {
  const sel = 'a, button, input, select, textarea, [role=button], [role=link], [role=tab], [role=menuitem], [onclick], [contenteditable=true]';
  const out = []; const vw = window.innerWidth, vh = window.innerHeight;
  for (const el of document.querySelectorAll(sel)) {
    const r = el.getBoundingClientRect();
    if (r.width < 6 || r.height < 6 || r.bottom < 0 || r.right < 0 || r.top > vh || r.left > vw) continue;
    const st = getComputedStyle(el); if (st.visibility === 'hidden' || st.display === 'none') continue;
    const type = (el.getAttribute('type') || '').toLowerCase();
    if (type === 'hidden' || type === 'checkbox' || type === 'radio') continue;
    const txt = (el.innerText || el.getAttribute('aria-label') || el.getAttribute('placeholder') || el.getAttribute('title') || el.getAttribute('alt') || el.value || '').trim().replace(/\s+/g, ' ').slice(0, 60);
    const isField = ['input', 'textarea', 'select'].includes(el.tagName.toLowerCase()) || el.getAttribute('contenteditable') === 'true';
    if (!txt && !isField) continue;
    out.push({tag: el.tagName.toLowerCase(), type: type, text: txt, x: r.left, y: r.top, w: r.width, h: r.height});
    if (out.length >= 120) break;
  }
  return out;
}
"""
# same option phrasing as the V2 training schema (build_schema_rows.ACT_VOCAB); the trained head only ever saw
# click/type/select as gold, so `done` is signalled by the `final` noul question, not by the act head
ACTIONS = {"click": "click on an element", "type": "type text into a field", "select": "choose an option from a dropdown",
           "scroll": "scroll the page", "done": "stop, the task is complete", "ask_user": "ask the user for clarification"}
FINAL_INSTR = "After this action, will the task be complete?"


def render(png: bytes, els):
    im = Image.open(io.BytesIO(png)).convert("RGB"); d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except OSError:
        font = ImageFont.load_default()
    for i, e in enumerate(els, 1):
        col = COLORS[(i - 1) % len(COLORS)]
        d.rectangle([e["x"], e["y"], e["x"] + e["w"], e["y"] + e["h"]], outline=col, width=3)
        lab = str(i); tw = d.textlength(lab, font=font); lx, ly = max(0, e["x"] - tw - 6), max(0, e["y"] - 2)
        d.rectangle([lx, ly, lx + tw + 6, ly + 24], fill=col); d.text((lx + 3, ly + 1), lab, fill="white", font=font)
    buf = io.BytesIO(); im.save(buf, format="PNG"); return buf.getvalue()


def ask_jev(state, questions):
    """TypeSafe Jev with the same state and questions; its contract carries no screenshot."""
    import os
    from typesafe_sdk import Choice, Noul, TypeSafeClient
    if not os.environ.get("TYPESAFE_API_KEY"):
        for line in (Path(__file__).resolve().parents[1] / ".env").read_text().splitlines():
            if line.startswith("TYPESAFE_API_KEY="):
                os.environ["TYPESAFE_API_KEY"] = line.split("=", 1)[1].strip()
    qs = {}
    for qid, q in questions.items():
        qs[qid] = Choice(instructions=q["instructions"], criteria=q["criteria"]) if q["type"] == "choice" else Noul(instructions=q["instructions"])
    t0 = time.time()
    with TypeSafeClient(timeout=60.0) as c:
        r = c.system_one(state=state, questions=qs)
    ms = (time.time() - t0) * 1000
    out = {}
    for qid, q in questions.items():
        a = r.answers[qid]
        if q["type"] == "choice":
            out[qid] = {"choice": a.choice, "confidence": float(a.confidence), "probabilities": dict(a.probabilities)}
        else:
            out[qid] = {"noul": float(a.noul), "confidence": round(abs(float(a.noul) - 0.5) * 2, 4)}
    return {"model": r.model, "answers": out, "usage": {"latency_ms": round(ms, 1)}}


def ask(server, state, image_png, questions):
    if server == "jev":
        return ask_jev(state, questions)
    payload = {"model": "s1", "state": state, "image": base64.b64encode(image_png).decode(), "questions": questions}
    r = urllib.request.Request(server.rstrip("/") + "/v1/systemone", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=300).read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True); ap.add_argument("--goal", required=True)
    ap.add_argument("--server", default="http://127.0.0.1:8811", help="decision server URL, or 'jev' for TypeSafe Jev (no screenshot)")
    ap.add_argument("--text", default="", help="text to type when the model decides to type")
    ap.add_argument("--max-steps", type=int, default=6); ap.add_argument("--min-confidence", type=float, default=0.3)
    ap.add_argument("--log", default="runs/browser_loop.jsonl"); ap.add_argument("--headless", action="store_true")
    ap.add_argument("--k", type=int, default=30, help="max candidates sent to the model")
    ap.add_argument("--done-threshold", type=float, default=0.5, help="stop after acting when the `final` noul exceeds this")
    a = ap.parse_args()
    print(json.dumps(run_task(a)))


def run_task(a):
    """Run one task; returns {"final_url", "steps", "stop"}. `a` needs the CLI attributes."""
    Path(a.log).parent.mkdir(parents=True, exist_ok=True); log = open(a.log, "a")
    history, steps, stop = [], [], "max_steps"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=a.headless); page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(a.url, wait_until="domcontentloaded"); page.wait_for_timeout(800)
        for step in range(a.max_steps):
            els = page.evaluate(JS_ELEMENTS)[: a.k]
            shot = page.screenshot(); marked = render(shot, els)
            crit = {str(i + 1): f"element {i+1}: {e['tag']}{(' ' + e['type']) if e['type'] else ''} '{e['text']}'" for i, e in enumerate(els)}
            crit["none"] = "none of the marked elements is the right target"
            hist = "\n".join(f"  - {h}" for h in history) or "  (none)"
            state = f"Task: {a.goal}\nActions already taken:\n{hist}\n"
            qs = {"ground": {"type": "choice", "instructions": "Which numbered element should be acted on next to make progress on the task?", "criteria": crit},
                  "act": {"type": "choice", "instructions": "What kind of action should be taken next?", "criteria": ACTIONS},
                  "done": {"type": "noul", "instructions": FINAL_INSTR}}
            t0 = time.time(); res = ask(a.server, state, marked, qs); wall = (time.time() - t0) * 1000
            g, act, done = res["answers"]["ground"], res["answers"]["act"], res["answers"]["done"]
            rec = {"step": step, "url": page.url, "n_candidates": len(els), "ground": g["choice"], "ground_conf": g["confidence"],
                   "act": act["choice"], "act_conf": act["confidence"], "done_p": done["noul"], "model_ms": res["usage"]["latency_ms"], "wall_ms": round(wall)}
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush(); steps.append(rec)
            if act["choice"] == "done":
                stop = "done"; break
            if g["confidence"] < a.min_confidence or g["choice"] == "none" or act["choice"] == "ask_user":
                stop = "escalate"; break
            final = done["noul"] > a.done_threshold  # trained semantics: "after this action, will the task be complete?"
            if act["choice"] == "scroll":
                page.mouse.wheel(0, 600); history.append("scrolled down"); page.wait_for_timeout(500); continue
            e = els[int(g["choice"]) - 1]; cx, cy = e["x"] + e["w"] / 2, e["y"] + e["h"] / 2
            if act["choice"] == "type":
                page.mouse.click(cx, cy); page.keyboard.type(a.text or ""); page.keyboard.press("Enter")
                history.append(f"typed '{a.text}' into {e['tag']} '{e['text']}'")
            else:
                page.mouse.click(cx, cy); history.append(f"clicked {e['tag']} '{e['text']}'")
            page.wait_for_timeout(1200)
            if final:
                stop = "done"; break
        Path(a.log).with_suffix(".final.png").write_bytes(page.screenshot())
        final = page.url; browser.close()
    return {"final_url": final, "steps": steps, "stop": stop}


if __name__ == "__main__":
    main()
