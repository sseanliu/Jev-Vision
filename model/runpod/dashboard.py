"""Tiny live training dashboard served from the pod.

python runpod/dashboard.py --runs runs --port 8888

Serves / (the page) and /api/runs (every runs/*/log.jsonl parsed). The page
polls every 15 s and draws train loss, eval accuracy and eval ECE per run as
single-series line charts with a crosshair tooltip. No external assets.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

RUNS = Path("runs")

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Training runs</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#fcfcfb;--ink:#1d1d1b;--muted:#6b6b66;--grid:#e6e6e2;--s1:#2a78d6;--s2:#1baf7a;--s3:#eb6834;--card:#ffffff}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#1a1a19;--ink:#ececea;--muted:#9a9a94;--grid:#2e2e2c;--s1:#3987e5;--s2:#199e70;--s3:#d95926;--card:#222220}}
body{margin:0;padding:16px;background:var(--bg);color:var(--ink);font:14px/1.4 -apple-system,system-ui,sans-serif}
h1{font-size:18px;margin:0 0 4px}.sub{color:var(--muted);margin-bottom:16px}
.run{margin-bottom:28px}.run h2{font-size:15px;margin:0 0 8px}
.tiles{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:10px}
.tile{background:var(--card);border:1px solid var(--grid);border-radius:8px;padding:8px 12px;min-width:120px}
.tile .k{color:var(--muted);font-size:12px}.tile .v{font-size:20px;font-variant-numeric:tabular-nums}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}
.chart{background:var(--card);border:1px solid var(--grid);border-radius:8px;padding:8px}
.chart .t{font-size:13px;color:var(--muted);margin:0 0 4px 4px}
svg{width:100%;height:180px;display:block}
.tip{position:fixed;pointer-events:none;background:var(--card);border:1px solid var(--grid);border-radius:6px;padding:4px 8px;font-size:12px;display:none;color:var(--ink)}
</style></head><body>
<h1>Training runs</h1><div class="sub" id="sub">loading…</div>
<div id="root"></div><div class="tip" id="tip"></div>
<script>
const C={loss:'--s1',acc:'--s2',ece:'--s3'};
function line(el,pts,color,fmt){
  const W=el.clientWidth||320,H=180,P={l:44,r:10,t:8,b:22};
  if(pts.length<2){el.innerHTML='<text x="10" y="20" fill="var(--muted)" font-size="12">waiting for data</text>';return;}
  const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]);
  const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
  const pad=(y1-y0)*0.08||0.01,ya=y0-pad,yb=y1+pad;
  const X=x=>P.l+(x-x0)/(x1-x0||1)*(W-P.l-P.r),Y=y=>P.t+(1-(y-ya)/(yb-ya))*(H-P.t-P.b);
  let g='';for(let i=0;i<4;i++){const y=ya+(yb-ya)*i/3;g+=`<line x1="${P.l}" x2="${W-P.r}" y1="${Y(y)}" y2="${Y(y)}" stroke="var(--grid)"/><text x="${P.l-6}" y="${Y(y)+4}" text-anchor="end" font-size="11" fill="var(--muted)">${fmt(y)}</text>`;}
  for(let i=0;i<=4;i++){const x=x0+(x1-x0)*i/4;g+=`<text x="${X(x)}" y="${H-6}" text-anchor="middle" font-size="11" fill="var(--muted)">${Math.round(x)}</text>`;}
  const d=pts.map((p,i)=>(i?'L':'M')+X(p[0]).toFixed(1)+' '+Y(p[1]).toFixed(1)).join(' ');
  const last=pts[pts.length-1];
  el.innerHTML=g+`<path d="${d}" fill="none" stroke="var(${color})" stroke-width="2" stroke-linejoin="round"/>`
   +`<circle cx="${X(last[0])}" cy="${Y(last[1])}" r="4" fill="var(${color})" stroke="var(--card)" stroke-width="2"/>`
   +`<line id="ch" x1="0" x2="0" y1="${P.t}" y2="${H-P.b}" stroke="var(--muted)" stroke-dasharray="3 3" style="display:none"/>`;
  const tip=document.getElementById('tip'),ch=el.querySelector('#ch');
  el.onmousemove=e=>{const r=el.getBoundingClientRect(),mx=e.clientX-r.left;let bi=0,bd=1e9;pts.forEach((p,i)=>{const dd=Math.abs(X(p[0])-mx);if(dd<bd){bd=dd;bi=i;}});
    const p=pts[bi];ch.setAttribute('x1',X(p[0]));ch.setAttribute('x2',X(p[0]));ch.style.display='';
    tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY-10)+'px';tip.textContent=`step ${p[0]}: ${fmt(p[1])}`;};
  el.onmouseleave=()=>{ch.style.display='none';tip.style.display='none';};
}
async function refresh(){
  const r=await fetch('/api/runs',{cache:'no-store'});const runs=await r.json();
  const root=document.getElementById('root');root.innerHTML='';
  for(const run of runs){
    const loss=run.train.map(x=>[x.step,x.loss]),acc=run.eval.map(x=>[x.step,x.acc]),ece=run.eval.map(x=>[x.step,x.ece]);
    const lt=run.train[run.train.length-1]||{},le=run.eval[run.eval.length-1]||{};
    const el=document.createElement('div');el.className='run';
    el.innerHTML=`<h2>${run.name}</h2><div class="tiles">
      <div class="tile"><div class="k">step</div><div class="v">${lt.step??'–'}</div></div>
      <div class="tile"><div class="k">train loss</div><div class="v">${lt.loss?.toFixed(3)??'–'}</div></div>
      <div class="tile"><div class="k">eval acc</div><div class="v">${le.acc?.toFixed(3)??'–'}</div></div>
      <div class="tile"><div class="k">eval ECE</div><div class="v">${le.ece?.toFixed(3)??'–'}</div></div>
      <div class="tile"><div class="k">elapsed</div><div class="v">${lt.elapsed?(lt.elapsed/60).toFixed(0)+' min':'–'}</div></div></div>
      <div class="charts"><div class="chart"><div class="t">train loss</div><svg id="l"></svg></div>
      <div class="chart"><div class="t">eval accuracy</div><svg id="a"></svg></div>
      <div class="chart"><div class="t">eval ECE (lower is better)</div><svg id="e"></svg></div></div>`;
    root.appendChild(el);
    line(el.querySelector('#l'),loss,C.loss,v=>v.toFixed(2));line(el.querySelector('#a'),acc,C.acc,v=>v.toFixed(3));line(el.querySelector('#e'),ece,C.ece,v=>v.toFixed(3));
  }
  document.getElementById('sub').textContent=`${runs.length} run(s) · updated ${new Date().toLocaleTimeString()} · polls every 15 s`;
}
refresh();setInterval(refresh,15000);window.addEventListener('resize',refresh);
</script></body></html>"""


def read_runs():
    out = []
    for log in sorted(RUNS.glob("*/log.jsonl")):
        train, ev = [], []
        for line in log.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "eval" in rec:
                ev.append({"step": rec["step"], **rec["eval"]})
            elif "loss" in rec:
                train.append(rec)
        out.append({"name": log.parent.name, "train": train, "eval": ev})
    return out


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/runs"):
            body = json.dumps(read_runs()).encode()
            ctype = "application/json"
        else:
            body = PAGE.encode()
            ctype = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--port", type=int, default=8888)
    a = ap.parse_args()
    RUNS = Path(a.runs)
    HTTPServer(("0.0.0.0", a.port), H).serve_forever()
