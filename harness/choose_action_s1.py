"""Bounded chooser with the jev-use process contract, backed by our vision decision server.

Reads one `cua.jev_choice_request_v1` JSON document on stdin and writes one `cua.jev_choice_v1`
on stdout, exactly like Cua's `python/choose_action.py`, so an application that already owns
capture, candidate construction, execution and verification can swap providers by changing
the chooser path. Two extensions over the Cua request, both optional:
  - "screenshot_path": a PNG the model may look at (Cua's contract carries only typed regions)
  - "questions": extra typed questions to answer on the same forward (e.g. {"done": {"type":"noul",...}})
The response keeps Cua's fields (schema, selected_id, model, confidence, probabilities) and adds
"answers" for the extra questions.

python harness/choose_action_s1.py --server http://127.0.0.1:8811 < request.json
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8811")
    ap.add_argument("--min-confidence", type=float, default=0.0, help="below this, answer 'abstain' if offered, else 'reobserve'")
    a = ap.parse_args()
    req = json.load(sys.stdin)
    if req.get("schema") != "cua.jev_choice_request_v1":
        sys.exit("expected cua.jev_choice_request_v1")
    cands = req["candidates"]
    ids = [c["id"] for c in cands]
    if len(set(ids)) != len(ids) or len(ids) > 32:
        sys.exit("candidate ids must be unique and at most 32")
    for k in ("tool", "arguments", "screenshot", "environment"):
        if any(k in c for c in cands):
            sys.exit(f"candidate field '{k}' is not allowed")
    hist = "\n".join(f"  - {json.dumps(h, ensure_ascii=False)[:200]}" for h in req.get("history", [])) or "  (none)"
    regions = req.get("regions") or []
    reg_txt = "\n".join(f"  - {r.get('id')}: {r.get('kind')} '{(r.get('text') or r.get('label') or '')[:60]}' at "
                        f"({r['bounds']['x']},{r['bounds']['y']},{r['bounds']['width']}x{r['bounds']['height']})"
                        for r in regions) or "  (none)"
    state = f"Goal: {req['goal']}\nCapture: {req.get('capture_id')}\nVisual regions:\n{reg_txt}\nHistory:\n{hist}\n"
    questions = {"driver_action": {"type": "choice", "instructions": "Which complete executable action should Cua Driver run next?",
                                   "criteria": {c["id"]: c["description"] for c in cands}}}
    questions.update(req.get("questions") or {})
    payload = {"model": "s1", "state": state, "image": req.get("screenshot_path"), "questions": questions}
    r = urllib.request.Request(a.server.rstrip("/") + "/v1/systemone", data=json.dumps(payload).encode(),
                               headers={"Content-Type": "application/json"})
    res = json.loads(urllib.request.urlopen(r, timeout=120).read())
    if "error" in res:
        sys.exit(res["error"])
    ans = res["answers"]["driver_action"]
    selected, conf = ans["choice"], ans["confidence"]
    if conf < a.min_confidence:
        selected = "abstain" if "abstain" in ids else ("reobserve" if "reobserve" in ids else selected)
    if selected not in ids:
        sys.exit(f"model selected unknown candidate {selected}")
    out = {"schema": "cua.jev_choice_v1", "selected_id": selected, "model": res.get("model"), "confidence": conf,
           "probabilities": ans["probabilities"]}
    extra = {k: v for k, v in res["answers"].items() if k != "driver_action"}
    if extra:
        out["answers"] = extra
    out["latency_ms"] = res.get("usage", {}).get("latency_ms")
    json.dump(out, sys.stdout); sys.stdout.write("\n")


if __name__ == "__main__":
    main()
