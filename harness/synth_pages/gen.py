"""Generate small synthetic pages with controlled state variants for the triplet recorder.

Each template renders under several states via query params the recorder never sees (it only sees the DOM):
  ?filled=1     the target field already holds the goal value (skip positives)
  ?wrong=1      the target field holds a wrong default value (must be overwritten)
  ?checked=1    the target checkbox already checked
  ?disabled=1   the submit/target button is disabled (clicking has no effect)
  ?overlay=1    a modal overlay covers the page (clicks land on the overlay, no effect)
  ?done=1       the page is already in the goal state (results shown)
Templates: search, signup form, settings toggles, date picker, list with filters, tabs.
Serves them with a tiny HTTP server; goals are embedded as data attributes so the recorder's generic goal
generator still works, and a JSON side file lists (url, goal, target, value) for exact-goal runs.

python harness/synth_pages/gen.py --out harness/synth_pages/site --port 8765 --serve
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import threading
from pathlib import Path

BASE = """<!doctype html><html><head><meta charset=utf-8><title>{title}</title>
<style>body{{font-family:system-ui;margin:24px;max-width:900px}} .row{{margin:10px 0}} input,select,button{{font-size:16px;padding:6px 10px}}
button{{cursor:pointer}} button:disabled{{opacity:.5}} .results{{margin-top:16px;padding:12px;background:#f3f3f3}} .hidden{{display:none}}
#overlay{{position:fixed;inset:0;background:rgba(0,0,0,.45);display:flex;align-items:center;justify-content:center}} #overlay.hidden{{display:none}}
#overlay div{{background:#fff;padding:24px;border-radius:8px}} nav a{{margin-right:14px}} .tab{{display:inline-block;padding:8px 14px;border:1px solid #ccc;margin-right:4px;cursor:pointer}} .tab.active{{background:#ddd}}
</style></head><body>
<nav><a href="/index__base.html">Home</a><a href="/search__base.html">Search</a><a href="/signup__base.html">Sign up</a><a href="/settings__base.html">Settings</a><a href="/booking__base.html">Booking</a><a href="/catalog__base.html">Catalog</a><a href="/help__base.html">Help</a></nav>
<h2>{title}</h2>
{body}
<div id="overlay" class="{overlay_cls}"><div><p>We use cookies.</p><button id="ovl-accept" onclick="document.getElementById('overlay').remove()">Accept</button> <button onclick="document.getElementById('overlay').remove()">Decline</button></div></div>
<script>
const q = new URLSearchParams(location.search);
{script}
</script></body></html>"""

TEMPLATES = {
    "search.html": ("Product search", """
<div class=row><label>Search <input id=q name=q placeholder="Search products" value="{qval}"></label> <button id=go {dis}>Search</button></div>
<div id=results class="results {res_cls}">Results for <b id=rq>{rq}</b>: <ul><li>Alpha widget</li><li>Beta gadget</li><li>Gamma tool</li></ul></div>
""", """
document.getElementById('go').onclick = () => { const v=document.getElementById('q').value; if(!v) return; history.pushState({}, '', '/search.html?q='+encodeURIComponent(v)); document.getElementById('rq').textContent=v; document.getElementById('results').classList.remove('hidden'); };
document.getElementById('q').addEventListener('keydown', e => { if(e.key==='Enter') document.getElementById('go').click(); });
"""),
    "signup.html": ("Create account", """
<form onsubmit="return false">
<div class=row><label>Name <input id=name name=name value="{name}"></label></div>
<div class=row><label>Email <input id=email name=email type=email value="{email}"></label></div>
<div class=row><label>City <input id=city name=city value="{city}"></label></div>
<div class=row><label><input type=checkbox id=tos {tos}> I agree to the terms</label></div>
<div class=row><button id=submit {dis}>Create account</button></div>
</form><div id=ok class="results {res_cls}">Account created for <span id=okname>{name}</span></div>
""", """
document.getElementById('submit').onclick = () => { if(!document.getElementById('tos').checked) return; document.getElementById('okname').textContent=document.getElementById('name').value; document.getElementById('ok').classList.remove('hidden'); history.pushState({}, '', '/signup.html?created=1'); };
"""),
    "settings.html": ("Settings", """
<div class=row><label><input type=checkbox id=dark {dark}> Dark mode</label></div>
<div class=row><label><input type=checkbox id=notif {notif}> Email notifications</label></div>
<div class=row><label>Language <select id=lang><option>English</option><option {es}>Spanish</option><option {de}>German</option></select></label></div>
<div class=row><button id=save {dis}>Save settings</button></div>
<div id=saved class="results {res_cls}">Settings saved</div>
""", """
document.getElementById('save').onclick = () => { document.getElementById('saved').classList.remove('hidden'); history.pushState({}, '', '/settings.html?saved=1'); };
"""),
    "booking.html": ("Book a table", """
<div class=row><label>Date <input id=date type=text placeholder="YYYY-MM-DD" value="{date}"></label></div>
<div class=row><label>Guests <select id=guests><option>1</option><option>2</option><option {g4}>4</option></select></label></div>
<div class=row><button id=book {dis}>Book</button></div>
<div id=booked class="results {res_cls}">Booked for <span id=bdate>{date}</span></div>
""", """
document.getElementById('book').onclick = () => { const d=document.getElementById('date').value; if(!d) return; document.getElementById('bdate').textContent=d; document.getElementById('booked').classList.remove('hidden'); history.pushState({}, '', '/booking.html?booked='+d); };
"""),
    "catalog.html": ("Catalog", """
<div class=row><span class="tab {t_all}" onclick="show('all')">All</span><span class="tab {t_sale}" onclick="show('sale')">On sale</span><span class="tab {t_new}" onclick="show('new')">New</span></div>
<div class=row><label><input type=checkbox id=instock {instock} onchange="history.pushState({{}}, '', '/catalog.html?instock='+this.checked)"> In stock only</label></div>
<ul id=items><li>Item A <button onclick="cart(this)">Add to cart</button></li><li>Item B <button onclick="cart(this)">Add to cart</button></li><li>Item C <button onclick="cart(this)">Add to cart</button></li></ul>
<div id=cart class="results {res_cls}">Cart: <span id=cartn>{cartn}</span> item(s)</div>
""", """
function show(t){ document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active')); event.target.classList.add('active'); history.pushState({}, '', '/catalog.html?tab='+t); }
function cart(b){ const n=document.getElementById('cartn'); n.textContent=(parseInt(n.textContent)||0)+1; document.getElementById('cart').classList.remove('hidden'); b.textContent='Added'; b.disabled=true; }
"""),
    "help.html": ("Help center", """
<div class=row><a href="/help__base.html#faq" id=faq>Frequently asked questions</a></div>
<div class=row><a href="/help__base.html?page=contact" id=contact>Contact support</a></div>
<div class=row><a href="/help__base.html?page=status">Service status</a></div>
<div class=row><button id=chat {dis}>Start chat</button></div>
<div id=page class="results {res_cls}">Page: <span>{page}</span></div>
""", ""),
    "index.html": ("Welcome", """
<p>Demo site for state-judgement data.</p><div class=row><a href="/search__base.html">Go to search</a> · <a href="/signup__base.html">Sign up</a> · <a href="/booking__base.html">Book</a></div>
""", ""),
}

VARIANTS = [{}, {"filled": 1}, {"wrong": 1}, {"checked": 1}, {"disabled": 1}, {"overlay": 1}, {"done": 1}]
GOALS = {  # exact goals whose values match the "filled" variants, so skip positives and overwrite cases both occur
    "search.html": [{"kind": "search", "goal": "Search for 'widget'", "target_text": "Search products", "value": "widget", "done_contains": "q=widget"}],
    "signup.html": [{"kind": "search", "goal": "Enter the name 'Ada Lovelace'", "target_text": "Name", "value": "Ada Lovelace", "done_contains": ""},
                    {"kind": "click", "goal": "Click 'Create account'", "target_text": "Create account", "value": "", "done_contains": "created=1"}],
    "settings.html": [{"kind": "click", "goal": "Turn on 'Dark mode'", "target_text": "Dark mode", "value": "", "done_contains": ""},
                      {"kind": "click", "goal": "Click 'Save settings'", "target_text": "Save settings", "value": "", "done_contains": "saved=1"}],
    "booking.html": [{"kind": "search", "goal": "Enter the date '2026-10-01'", "target_text": "YYYY-MM-DD", "value": "2026-10-01", "done_contains": ""},
                     {"kind": "click", "goal": "Click 'Book'", "target_text": "Book", "value": "", "done_contains": "booked="}],
    "catalog.html": [{"kind": "click", "goal": "Open the 'On sale' tab", "target_text": "On sale", "value": "", "done_contains": "tab=sale"},
                     {"kind": "click", "goal": "Click 'Add to cart' for Item A", "target_text": "Add to cart", "value": "", "done_contains": ""}],
    "help.html": [{"kind": "follow", "goal": "Open 'Contact support'", "target_text": "Contact support", "value": "", "done_contains": "page=contact"},
                  {"kind": "click", "goal": "Click 'Start chat'", "target_text": "Start chat", "value": "", "done_contains": ""}],
    "index.html": [{"kind": "follow", "goal": "Open 'Go to search'", "target_text": "Go to search", "value": "", "done_contains": "search__base"}],
}


def render(name, v):
    title, body, script = TEMPLATES[name]
    filled = v.get("filled"); wrong = v.get("wrong"); done = v.get("done")
    ctx = dict(title=title, dis="disabled" if v.get("disabled") else "", overlay_cls="" if v.get("overlay") else "hidden",
               res_cls="" if done else "hidden", qval="widget" if filled else ("gadget" if wrong else ""), rq="widget" if done else "",
               name="Ada Lovelace" if filled else ("test" if wrong else ""), email="ada@example.com" if filled else "", city="London" if filled else ("Paris" if wrong else ""),
               tos="checked" if v.get("checked") or done else "", dark="checked" if v.get("checked") or done else "", notif="", es="", de="selected" if filled else "",
               date="2026-10-01" if filled else ("2026-01-01" if wrong else ""), g4="selected" if filled else "", t_all="active" if not done else "", t_sale="active" if done else "", t_new="",
               instock="checked" if v.get("checked") or done else "", cartn="1" if done else "0", page="contact" if done else "")
    return BASE.format(body=body.format(**ctx), script=script, **{k: ctx[k] for k in ("title", "overlay_cls")})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="harness/synth_pages/site"); ap.add_argument("--port", type=int, default=8765); ap.add_argument("--serve", action="store_true")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); urls = []
    for name in TEMPLATES:
        for i, v in enumerate(VARIANTS):
            tag = "_".join(f"{k}{val}" for k, val in v.items()) or "base"
            fn = f"{name[:-5]}__{tag}.html"; (out / fn).write_text(render(name, v)); urls.append(f"http://127.0.0.1:{a.port}/{fn}")
    (out / "urls.txt").write_text("\n".join(urls) + "\n"); print("wrote", len(urls), "pages")
    with (out / "goals.jsonl").open("w") as f:
        for name in TEMPLATES:
            for v in VARIANTS:
                tag = "_".join(f"{k}{val}" for k, val in v.items()) or "base"
                for g in GOALS.get(name, []):
                    f.write(json.dumps({"url": f"http://127.0.0.1:{a.port}/{name[:-5]}__{tag}.html", "variant": tag, **g}) + "\n")
    if a.serve:
        os.chdir(out)
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), http.server.SimpleHTTPRequestHandler)
        print("serving on", a.port, flush=True); srv.serve_forever()


if __name__ == "__main__":
    main()
