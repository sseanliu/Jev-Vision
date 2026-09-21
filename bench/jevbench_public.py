"""Run the JevBench public items (datasets/public/{easy,original,hard}.jsonl from github.com/fstandhartinger/jevbench, fetched next to this script) (easy / original=standard / hard) through a /v1/systemone endpoint, text-only.
Reports per-tier accuracy (the harness's Intelligence inputs) and latency; not the official score."""
import json, sys, time, urllib.request, statistics
URL = sys.argv[1]; name = sys.argv[2]
out = {}
for tier in ["easy", "original", "hard"]:
    rows = [json.loads(l) for l in open(f"{tier}.jsonl") if l.strip()]
    hits = []; ms = []; per_type = {}
    for r in rows:
        q = dict(r["question"])
        body = {"model": "s1", "state": r["state"], "questions": {"q": q}}
        t0 = time.perf_counter()
        try:
            resp = json.load(urllib.request.urlopen(urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=180))
            a = resp["answers"]["q"]
        except Exception as e:
            a = {"error": str(e)[:80]}
        ms.append((time.perf_counter() - t0) * 1000)
        exp = str(r["expected"])
        if q["type"] == "noul":
            p = a.get("noul"); pred = None if p is None else ("yes" if p >= 0.5 else "no")
        elif q["type"] == "choice":
            pred = a.get("choice")
        else:
            pred = None if a.get("score") is None else str(int(round(a["score"])))
        hit = int(pred == exp); hits.append(hit); per_type.setdefault(q["type"], []).append(hit)
    out[tier] = {"n": len(rows), "acc": round(sum(hits) / len(hits), 3), "p50_ms": round(statistics.median(ms)), "by_type": {k: round(sum(v) / len(v), 3) for k, v in per_type.items()}}
    print(name, tier, out[tier], flush=True)
json.dump(out, open(f"result_{name}.json", "w"), indent=1)
