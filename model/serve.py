"""HTTP server exposing the vision decision model with a TypeSafe-shaped contract, so harnesses
written for Jev (e.g. Cua's jev-use adapter) can point at it instead.

POST /v1/system-one
  {"model": "...", "state": <str | object>, "image": <base64 PNG | file path | null>,
   "questions": {qid: {"type": "choice"|"noul"|"score", "instructions": str, "criteria": {...}|[...],
                       "temperature": float (optional, overrides --temp; formats calibrate differently)}}}
  -> {"model": "<checkpoint>", "answers": {qid: {"type", "choice"|"noul"|"score", "confidence", "probabilities"}},
      "usage": {"visual_tokens": int, "latency_ms": float}}

A JSON `state` is rendered as indented text. `image` is optional: without it the model runs
text-only (state prefix has no vision tokens). Single forward per request, all questions at once.

python serve.py runs/v2-8b-schema/final --port 8811 --temp 0.5
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from PIL import Image

from eval_vl import load
from s1.packing import Question

MODEL = {}


def render_state(state) -> str:
    return state if isinstance(state, str) else json.dumps(state, indent=1, ensure_ascii=False)


def decode_image(img):
    if img is None:
        return None
    if isinstance(img, str) and len(img) < 4096 and Path(img).exists():
        return Image.open(img).convert("RGB")
    raw = base64.b64decode(img.split(",", 1)[-1] if isinstance(img, str) else img)
    return Image.open(io.BytesIO(raw)).convert("RGB")


@torch.no_grad()
def answer(req: dict) -> dict:
    model, packer, name = MODEL["model"], MODEL["packer"], MODEL["name"]
    qs, order = [], []
    for qid, q in req["questions"].items():
        qs.append(Question(qid, q["type"], q["instructions"], q.get("criteria"))); order.append(qid)
    img = decode_image(req.get("image"))
    state = render_state(req["state"])
    t0 = time.perf_counter()
    if img is not None:
        packed = packer.pack(img, state, qs)
        logits = model(packed, MODEL["device"])
    else:
        packed = packer.text.pack(state, qs)
        logits = model.__class__.__mro__[1].forward(model, packed, MODEL["device"])  # DecisionModel.forward (text-only)
    if MODEL["device"].type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    out = {}
    for qid, z, qtype, keys in zip(order, logits, packed.qtypes, packed.option_keys):
        z = z.float()
        temp = float(req["questions"][qid].get("temperature", MODEL["temp"]))  # per-question / per-platform override
        if qtype == "noul":
            p = float(torch.sigmoid(z)); out[qid] = {"type": "noul", "noul": p, "confidence": round(abs(p - 0.5) * 2, 4)}
        else:
            pr = torch.softmax(z / temp, -1); probs = {k: float(v) for k, v in zip(keys, pr)}
            top = max(probs, key=probs.get); K = len(keys)
            conf = (probs[top] - 1 / K) / (1 - 1 / K) if K > 1 else 1.0
            entry = {"type": qtype, "probabilities": probs, "confidence": round(conf, 4)}
            if qtype == "choice":
                entry["choice"] = top
            else:
                entry["score"] = float(sum(p_i * i for i, p_i in enumerate(pr)))
            out[qid] = entry
    return {"model": name, "answers": out, "usage": {"tokens": len(packed.input_ids), "latency_ms": round(ms, 1)}}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n))
            res = answer(req); code = 200
        except Exception as e:  # report, don't crash the server
            res = {"error": f"{type(e).__name__}: {e}"}; code = 400
        body = json.dumps(res).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--port", type=int, default=8811)
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--max-pixels", type=int, default=1288 * 1000)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model, packer = load(Path(a.checkpoint), device, a.max_pixels)
    MODEL.update(model=model, packer=packer, temp=a.temp, device=device, name=Path(a.checkpoint).parent.name)
    print(f"serving {MODEL['name']} on :{a.port} ({device})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
