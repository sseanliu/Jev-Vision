"""Record (before, action, after) triplets on public web pages with automatic labels for the three
state questions: skip (candidate already satisfied), effect (did the action change the page), done
(is the goal reached). No model in the loop: goals are generated from the page itself so success is
checkable, and the policy deliberately mixes correct actions with wrong ones so every label has
positives and negatives.

Goal kinds
  follow   "Open '<link text>'"                     done = URL equals/contains the link href target
  search   "Search for '<q>'"                       done = URL or page text contains q after submit
  click    "Click '<button text>'"                  done = page changed after clicking that element

Policy per step: oracle (goal element), random candidate, repeat last action, or an unrelated
element, with the given mix. Each step records screenshot + candidate table before and after, the
action, and labels. Output: <out>/steps.jsonl + <out>/img/*.png.

python harness/record_triplets.py --sites harness/sites.txt --out model/data/vision/triplets/run1 --tasks-per-site 4
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import time
from pathlib import Path

from PIL import Image, ImageChops
from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright

from browser_loop import JS_ELEMENTS, render

JS_HREF = "(i) => { const els = document.querySelectorAll('a, button, input, select, textarea, [role=button], [role=link], [role=tab], [role=menuitem], [onclick], [contenteditable=true]'); return null; }"
JS_ELEMENTS_FULL = JS_ELEMENTS.replace("out.push({tag:", "out.push({href: el.getAttribute('href') || '', disabled: !!el.disabled, tag:")
JS_VALUE = "(idx) => { const sel='a, button, input, select, textarea, [role=button], [role=link], [role=tab], [role=menuitem], [onclick], [contenteditable=true]'; const vis=[]; const vw=innerWidth,vh=innerHeight; for (const el of document.querySelectorAll(sel)) { const r=el.getBoundingClientRect(); if (r.width<6||r.height<6||r.bottom<0||r.right<0||r.top>vh||r.left>vw) continue; const st=getComputedStyle(el); if (st.visibility==='hidden'||st.display==='none') continue; vis.push(el);} return vis.map(e => e.value ?? '') }"


def pix_diff(a: bytes, b: bytes) -> float:
    ia = Image.open(io.BytesIO(a)).convert("L").resize((256, 160)); ib = Image.open(io.BytesIO(b)).convert("L").resize((256, 160))
    d = ImageChops.difference(ia, ib); h = d.histogram(); tot = sum(h)
    return sum(h[16:]) / tot  # fraction of pixels differing by >= 16 levels


def sig(els) -> str:
    return hashlib.md5(json.dumps([(e["tag"], e["text"], round(e["x"]), round(e["y"])) for e in els]).encode()).hexdigest()


def candidates(page, k):
    els = page.evaluate(JS_ELEMENTS_FULL)[:k]
    return els


def make_goal(rng, els, url):
    links = [e for e in els if e["tag"] == "a" and e.get("href") and len(e["text"]) >= 3 and not e["href"].startswith(("#", "javascript", "mailto"))]
    fields = [e for e in els if e["tag"] in ("input", "textarea") and e["type"] in ("", "text", "search")]
    buttons = [e for e in els if e["tag"] == "button" and len(e["text"]) >= 2]
    kinds = []
    if links: kinds.append("follow")
    if fields: kinds.append("search")
    if buttons: kinds.append("click")
    if not kinds:
        return None
    kind = rng.choice(kinds)
    if kind == "follow":
        e = rng.choice(links); return {"kind": kind, "goal": f"Open '{e['text'][:50]}'", "target": e, "value": ""}
    if kind == "search":
        e = rng.choice(fields); q = rng.choice(["python", "weather", "history", "music", "recipe", "map", "news", "help", "login", "price"])
        return {"kind": kind, "goal": f"Search for '{q}'", "target": e, "value": q}
    e = rng.choice(buttons); return {"kind": kind, "goal": f"Click '{e['text'][:50]}'", "target": e, "value": ""}


def same_el(a, b) -> bool:
    return a["tag"] == b["tag"] and a["text"] == b["text"] and abs(a["x"] - b["x"]) < 4 and abs(a["y"] - b["y"]) < 4


def find_target(els, target):
    for i, e in enumerate(els):
        if same_el(e, target):
            return i
    for i, e in enumerate(els):  # after navigation positions move; fall back to tag+text
        if e["tag"] == target["tag"] and e["text"] == target["text"] and e["text"]:
            return i
    return None


def done_check(goal, page, start_url) -> bool:
    url = page.url
    if goal.get("done_contains"):
        return goal["done_contains"] in url
    if goal["kind"] == "follow":
        href = goal["target"]["href"]
        return url != start_url and (href.split("#")[0].rstrip("/") in url or url.rstrip("/").endswith(href.rstrip("/")) or (href.startswith("http") and url.startswith(href.split("?")[0][:60])))
    if goal["kind"] == "search":
        return search_done(url, goal["value"])
    return False  # click goals: done is decided from effect at the oracle step


SEARCH_PARAMS = r"(?:[?&#](?:q|query|search|search_query|s|k|keyword|keywords|term|text|p|wd|find|searchTerm)=)"


def search_done(url: str, q: str) -> bool:
    """A search counts as submitted only when the query appears as a search parameter or a /search path
    segment of the URL. Page text and incidental URL matches (a page titled '...Recipe') do not count."""
    from urllib.parse import unquote_plus
    u = unquote_plus(url).lower(); q = q.lower()
    if re.search(SEARCH_PARAMS + r"[^&#]*" + re.escape(q), u):
        return True
    if re.search(r"/search[^?]*/" + re.escape(q) + r"(?:[/?&#]|$)", u):
        return True
    # exact-title landing (Wikipedia-style "search jumps to the article"): last path segment equals the query
    last = u.split("?")[0].split("#")[0].rstrip("/").rsplit("/", 1)[-1].replace("_", " ")
    return last == q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tasks-per-site", type=int, default=4); ap.add_argument("--max-steps", type=int, default=3)
    ap.add_argument("--k", type=int, default=30); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mix", default="0.5,0.25,0.15,0.10", help="oracle, random, repeat, unrelated")
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--limit-sites", type=int, default=0)
    ap.add_argument("--goals", default=None, help="jsonl of exact goals {url, kind, goal, target_text, value, done_contains}; sites = its urls")
    ap.add_argument("--start-site", type=int, default=0, help="skip the first N sites (resume); ids keep the global site index")
    a = ap.parse_args()
    rng = random.Random(a.seed); out = Path(a.out); (out / "img").mkdir(parents=True, exist_ok=True)
    mix = [float(x) for x in a.mix.split(",")]
    goals_by_url = {}
    if a.goals:
        for l in open(a.goals):
            g = json.loads(l); goals_by_url.setdefault(g["url"], []).append(g)
        sites = list(goals_by_url)
    else:
        sites = [l.strip() for l in open(a.sites) if l.strip() and not l.startswith("#")]
    if a.limit_sites:
        sites = sites[: a.limit_sites]
    log = open(out / "steps.jsonl", "a"); n_steps = 0; t_start = time.time()
    UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0 Safari/537.36"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=a.headless)
        for si, site in enumerate(sites):
            if si < a.start_site:
                continue
            # one context per site: popups, service workers and renderer processes die with it
            ctx = browser.new_context(viewport={"width": 1280, "height": 900}, user_agent=UA)
            for ti in range(a.tasks_per_site):
                page = ctx.new_page()
                try:
                    page.goto(site, wait_until="domcontentloaded", timeout=25000); page.wait_for_timeout(1200)
                    els = candidates(page, a.k)
                    if site in goals_by_url:
                        g = goals_by_url[site][ti % len(goals_by_url[site])]
                        tgt = next((e for e in els if e["text"] == g["target_text"] or (e["tag"] in ("input", "textarea") and g["target_text"] in (e["text"], ""))), None)
                        goal = {"kind": g["kind"], "goal": g["goal"], "target": tgt, "value": g.get("value", ""), "done_contains": g.get("done_contains", "")} if tgt else None
                    else:
                        goal = make_goal(rng, els, page.url)
                    if not goal:
                        page.close(); continue
                    start_url = page.url; history = []; last_action = None; goal_reached = False
                    for step in range(a.max_steps):
                        before_shot = page.screenshot(); before_els = candidates(page, a.k); before_sig = sig(before_els); before_url = page.url
                        before_vals = page.evaluate(JS_VALUE)[: len(before_els)]
                        if goal_reached:
                            done_before = True
                        else:
                            done_before = done_check(goal, page, start_url)
                        tidx = find_target(before_els, goal["target"])
                        # candidate-level skip labels: satisfied if field already holds the value, or goal reached
                        skip = {}
                        for i, e in enumerate(before_els):
                            satisfied = False
                            if goal["kind"] == "search" and tidx == i and before_vals[i] and before_vals[i].strip().lower() == goal["value"].lower():
                                satisfied = True
                            if done_before:
                                satisfied = True
                            skip[str(i + 1)] = int(satisfied)
                        # choose policy
                        r = rng.random(); kind = "oracle" if r < mix[0] else "random" if r < mix[0] + mix[1] else "repeat" if r < mix[0] + mix[1] + mix[2] else "unrelated"
                        if kind == "oracle" and tidx is None:
                            kind = "random"
                        if kind == "repeat" and last_action is None:
                            kind = "random"
                        if kind == "oracle":
                            idx = tidx
                        elif kind == "repeat":
                            idx = find_target(before_els, last_action["el"]); idx = idx if idx is not None else rng.randrange(len(before_els))
                        elif kind == "unrelated":
                            pool = [i for i, e in enumerate(before_els) if e["tag"] not in ("a", "button") or e.get("disabled")] or list(range(len(before_els)))
                            idx = rng.choice(pool)
                        else:
                            idx = rng.randrange(len(before_els))
                        e = before_els[idx]; cx, cy = e["x"] + e["w"] / 2, e["y"] + e["h"] / 2
                        act = "type" if (e["tag"] in ("input", "textarea") and (kind == "oracle" and goal["kind"] == "search" or (kind != "oracle" and rng.random() < 0.5))) else "click"
                        typed = goal["value"] if act == "type" and kind == "oracle" else (rng.choice(["test", "hello", "abc"]) if act == "type" else "")
                        try:
                            if act == "type":
                                page.mouse.click(cx, cy); page.keyboard.press("Meta+A"); page.keyboard.type(typed); page.keyboard.press("Enter")
                            else:
                                page.mouse.click(cx, cy)
                        except Exception as ex:
                            act_err = type(ex).__name__
                        else:
                            act_err = None
                        try:
                            page.wait_for_load_state("domcontentloaded", timeout=5000)
                        except PWTimeout:
                            pass
                        page.wait_for_timeout(800); after1 = page.screenshot(); page.wait_for_timeout(500); after2 = page.screenshot()
                        after_els = candidates(page, a.k); after_sig = sig(after_els); after_url = page.url
                        after_vals = page.evaluate(JS_VALUE)[: len(after_els)]
                        d_change = pix_diff(before_shot, after1); d_noise = pix_diff(after1, after2)
                        url_changed = after_url != before_url; dom_changed = after_sig != before_sig
                        value_applied = act == "type" and any(v.strip().lower() == typed.lower() for v in after_vals) and typed != ""
                        effect = int(url_changed or dom_changed or value_applied or d_change > 0.02)
                        noisy = d_noise > 0.02 and not url_changed and not dom_changed
                        goal_reached = goal_reached or done_check(goal, page, start_url) or (goal["kind"] == "click" and kind == "oracle" and effect == 1)
                        done_after = int(goal_reached)
                        sid = f"{si:03d}_{ti}_{step}"
                        (out / "img" / f"{sid}_before.png").write_bytes(render(before_shot, before_els)); (out / "img" / f"{sid}_after.png").write_bytes(render(after1, after_els))
                        rec = {"id": sid, "site": site, "goal": goal["goal"], "goal_kind": goal["kind"], "goal_value": goal["value"], "target_text": goal["target"]["text"],
                               "step": step, "history": list(history), "policy": kind, "target_href": goal["target"].get("href", ""), "start_url": start_url, "action": act, "typed": typed, "chosen": str(idx + 1), "target_idx": (str(tidx + 1) if tidx is not None else None),
                               "before_img": f"img/{sid}_before.png", "after_img": f"img/{sid}_after.png", "before_url": before_url, "after_url": after_url,
                               "candidates": {str(i + 1): f"element {i+1}: {c['tag']}{(' ' + c['type']) if c['type'] else ''} '{c['text']}'" + (f" value='{before_vals[i][:40]}'" if i < len(before_vals) and before_vals[i] else "") for i, c in enumerate(before_els)},
                               "candidates_after": {str(i + 1): f"element {i+1}: {c['tag']}{(' ' + c['type']) if c['type'] else ''} '{c['text']}'" + (f" value='{after_vals[i][:40]}'" if i < len(after_vals) and after_vals[i] else "") for i, c in enumerate(after_els)},
                               "labels": {"skip": skip, "effect": effect, "done_before": int(done_before), "done_after": done_after, "noisy": int(noisy)},
                               "signals": {"url_changed": url_changed, "dom_changed": dom_changed, "value_applied": value_applied, "pix_change": round(d_change, 4), "pix_noise": round(d_noise, 4), "act_err": act_err}}
                        log.write(json.dumps(rec, ensure_ascii=False) + "\n"); log.flush(); n_steps += 1
                        history.append(f"{act} on {e['tag']} '{e['text'][:40]}'" + (f" typed '{typed}'" if typed else "")); last_action = {"el": e}
                        if done_after and rng.random() < 0.5:
                            break  # sometimes keep going after done, to record "already done, should skip" steps
                except Exception as ex:
                    log.write(json.dumps({"site": site, "task": ti, "error": f"{type(ex).__name__}: {str(ex)[:120]}"}) + "\n"); log.flush()
                finally:
                    try:
                        page.close()
                    except Exception:
                        pass
            try:
                ctx.close()
            except Exception:
                pass
            print(f"[{si+1}/{len(sites)}] {site} steps so far {n_steps} ({time.time()-t_start:.0f}s)", flush=True)
        browser.close()
    print("done", n_steps, "steps")


if __name__ == "__main__":
    main()
