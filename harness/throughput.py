"""Concurrent-request throughput of the decision server (one screenshot + 3 typed questions per request).

python harness/throughput.py --server http://127.0.0.1:8811 --image model/data/vision/m2w_items_j/img/<any>.png
"""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

QS = {"ground": {"type": "choice", "instructions": "Which numbered element should be acted on next to make progress on the task?",
                 "criteria": {str(i): f"element {i}" for i in range(1, 10)} | {"none": "none of the marked elements is the right target"}},
      "act": {"type": "choice", "instructions": "What kind of action should be taken next?",
              "criteria": {"click": "click on an element", "type": "type text into a field", "select": "choose an option from a dropdown"}},
      "final": {"type": "noul", "instructions": "After this action, will the task be complete?"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8811"); ap.add_argument("--image", required=True)
    ap.add_argument("--n", type=int, default=24); ap.add_argument("--concurrency", default="1,4,8")
    a = ap.parse_args()
    img = base64.b64encode(Path(a.image).read_bytes()).decode()
    payload = json.dumps({"model": "s1", "state": "Task: open the settings page\nActions already taken:\n  (none)\n", "image": img, "questions": QS}).encode()

    def one(_):
        t0 = time.perf_counter()
        r = urllib.request.Request(a.server.rstrip("/") + "/v1/systemone", data=payload, headers={"Content-Type": "application/json"})
        res = json.loads(urllib.request.urlopen(r, timeout=300).read())
        return (time.perf_counter() - t0) * 1000, res["usage"]["latency_ms"]

    one(0)  # warm
    for c in (int(x) for x in a.concurrency.split(",")):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(c) as ex:
            rows = list(ex.map(one, range(a.n)))
        wall = time.perf_counter() - t0
        e2e = [r[0] for r in rows]; model = [r[1] for r in rows]
        print(json.dumps({"concurrency": c, "n": a.n, "req_per_s": round(a.n / wall, 2), "e2e_p50_ms": round(statistics.median(e2e), 1),
                          "e2e_p90_ms": round(sorted(e2e)[int(0.9 * len(e2e)) - 1], 1), "model_p50_ms": round(statistics.median(model), 1)}))


if __name__ == "__main__":
    main()
