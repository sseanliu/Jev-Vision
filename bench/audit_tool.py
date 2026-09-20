"""Build a self-contained HTML page for a human audit of the rule labels.

Samples N items per question from bench/<name>_items.jsonl, writes release/<name>/audit.html next to the images.
Open it in a browser; answer each item with the keyboard (y / n / u for unsure, arrows to move); the page keeps
answers in localStorage and the "Download" button writes audit_<name>.jsonl ({id, question, rule_label, human, note}).
Score agreement with:  python bench/audit_tool.py --score release/v0/audit_v0.jsonl

python bench/audit_tool.py --name v0 --per-question 120 --seed 7
"""

import argparse
import html
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Label audit __NAME__</title>
<style>
body{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:0;background:#f4f4f4;color:#222}
#top{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:8px 16px;display:flex;gap:16px;align-items:center;font-size:14px}
#top b{font-size:16px}
.item{display:none;padding:12px 16px}
.item.on{display:block}
.q{background:#fff;border:1px solid #ddd;padding:10px 12px;margin-bottom:8px;white-space:pre-wrap;font-size:14px;line-height:1.35}
.imgs{display:flex;gap:8px;flex-wrap:wrap}
.imgs img{max-width:48%;border:1px solid #ccc;background:#fff}
.imgs img.single{max-width:70%}
.ans{margin-top:8px;font-size:15px}
.ans span{display:inline-block;padding:3px 10px;border-radius:4px;margin-right:6px;border:1px solid #bbb}
.ans .y{background:#dff5dd}.ans .n{background:#f8dcdc}.ans .u{background:#eee}
button{font-size:14px;padding:4px 10px}
.rule{color:#888;font-size:12px}
</style></head><body>
<div id="top"><b>Label audit __NAME__</b><span id="pos"></span><span>keys: <kbd>y</kbd> yes <kbd>n</kbd> no <kbd>u</kbd> unsure, <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> move, <kbd>r</kbd> reveal rule label</span>
<span id="done"></span><button onclick="download()">Download jsonl</button><input id="note" placeholder="note (optional)" style="flex:1"></div>
<div id="items">__ITEMS__</div>
<script>
const items=__META__; let i=+(localStorage.getItem('audit_i___NAME__')||0);
const ans=JSON.parse(localStorage.getItem('audit_a___NAME__')||'{}');
function show(k){i=Math.max(0,Math.min(items.length-1,k));document.querySelectorAll('.item').forEach((e,j)=>e.classList.toggle('on',j===i));
 const it=items[i];const a=ans[it.id]||{};document.getElementById('pos').textContent=(i+1)+' / '+items.length+'  ['+it.question+']';
 document.getElementById('done').textContent=Object.keys(ans).length+' answered';document.getElementById('note').value=a.note||'';
 document.querySelectorAll('.item.on .ans span').forEach(s=>s.style.outline=(s.dataset.v===a.human)?'3px solid #333':'');
 localStorage.setItem('audit_i___NAME__',i);window.scrollTo(0,0);}
function mark(v){const it=items[i];ans[it.id]={id:it.id,question:it.question,rule_label:it.rule_label,human:v,note:document.getElementById('note').value};
 localStorage.setItem('audit_a___NAME__',JSON.stringify(ans));show(i+1);}
document.addEventListener('keydown',e=>{if(e.target.id==='note'&&e.key!=='Escape')return;
 if(e.key==='y')mark('yes');else if(e.key==='n')mark('no');else if(e.key==='u')mark('unsure');
 else if(e.key==='ArrowRight')show(i+1);else if(e.key==='ArrowLeft')show(i-1);
 else if(e.key==='r'){const r=document.querySelector('.item.on .rule');r.style.display=r.style.display==='block'?'none':'block';}});
function download(){const rows=Object.values(ans).map(a=>JSON.stringify(a)).join('\\n')+'\\n';
 const b=new Blob([rows],{type:'application/json'});const u=URL.createObjectURL(b);const a=document.createElement('a');a.href=u;a.download='audit___NAME__.jsonl';a.click();}
show(i);
</script></body></html>"""


def build(name, per_question, seed, img_prefix=None):
    items = [json.loads(l) for l in open(ROOT / "bench" / f"{name}_items.jsonl")]
    by = defaultdict(list)
    for it in items:
        if it["type"] == "noul":
            by[it["question"]].append(it)
    rng = random.Random(seed); sample = []
    for q, its in sorted(by.items()):
        # balance: half positives if available
        pos = [x for x in its if x["label"]]; neg = [x for x in its if not x["label"]]
        rng.shuffle(pos); rng.shuffle(neg)
        k = min(per_question, len(its)); kp = min(len(pos), k // 2); kn = min(len(neg), k - kp)
        sample += pos[:kp] + neg[:kn]
    rng.shuffle(sample)
    blocks, meta = [], []
    for it in sample:
        srcs = [(img_prefix.rstrip("/") + "/" + p.split("/", 1)[1]) if img_prefix else p for p in it["images"]]
        imgs = "".join(f'<img class="{"single" if len(it["images"]) == 1 else ""}" src="{p}" loading="lazy">' for p in srcs)
        cap = " (left: before, right: after)" if len(it["images"]) == 2 else ""
        blocks.append(
            f'<div class="item"><div class="q"><b>{html.escape(it["question"])}</b>{cap}\n{html.escape(it["state"])}\n<b>Q:</b> {html.escape(it["instructions"])}'
            f'\n<span class="rule" style="display:none">rule label: {it["label"]}   url_before: {html.escape(str(it["url_before"]))}   url_after: {html.escape(str(it["url_after"]))}</span></div>'
            f'<div class="imgs">{imgs}</div><div class="ans"><span class="y" data-v="yes">y = yes</span><span class="n" data-v="no">n = no</span><span class="u" data-v="unsure">u = unsure</span></div></div>')
        meta.append({"id": it["id"], "question": it["question"], "rule_label": it["label"]})
    page = PAGE.replace("__NAME__", name).replace("__ITEMS__", "\n".join(blocks)).replace("__META__", json.dumps(meta))
    out = ROOT / "release" / name / "audit.html"; out.write_text(page)
    print(f"{len(sample)} items -> {out}")


def score(path):
    rows = [json.loads(l) for l in open(path)]
    by = defaultdict(list)
    for r in rows:
        if r["human"] in ("yes", "no"):
            by[r["question"]].append((int(r["rule_label"]), 1 if r["human"] == "yes" else 0))
    for q, xs in sorted(by.items()):
        agree = sum(a == b for a, b in xs) / len(xs)
        unsure = sum(1 for r in rows if r["question"] == q and r["human"] == "unsure")
        print(f"{q:7} n={len(xs):4} agreement={agree:.3f} unsure={unsure}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v0"); ap.add_argument("--per-question", type=int, default=120); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--score", default=None)
    ap.add_argument("--img-prefix", default=None, help="replace the leading images/ of item paths with this (e.g. an absolute recording dir)")
    a = ap.parse_args()
    if a.score:
        score(a.score)
    else:
        build(a.name, a.per_question, a.seed, a.img_prefix)


if __name__ == "__main__":
    main()
