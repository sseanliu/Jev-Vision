"""Small live-browser benchmark for the decide-act-verify loop: stable public pages, one goal each,
success judged by the final URL. Anecdotal by design (a dozen tasks), but paired across checkpoints.

python harness/loop_bench.py --server http://127.0.0.1:8811 --tag v1b --out results/vision/harness
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

from browser_loop import run_task

TASKS = [
    {"id": "example_more_info", "url": "https://example.com", "goal": "Open the 'More information' link", "ok": "iana.org"},
    {"id": "wiki_search", "url": "https://en.wikipedia.org/wiki/Main_Page", "goal": "Search Wikipedia for 'Transformer (machine learning)'",
     "text": "Transformer (machine learning)", "ok": "Transformer"},
    {"id": "wiki_talk_tab", "url": "https://en.wikipedia.org/wiki/Python_(programming_language)", "goal": "Open the Talk page of this article", "ok": "Talk:Python"},
    {"id": "wiki_history", "url": "https://en.wikipedia.org/wiki/Python_(programming_language)", "goal": "Open the edit history of this article", "ok": "action=history"},
    {"id": "wiki_random", "url": "https://en.wikipedia.org/wiki/Main_Page", "goal": "Go to a random article", "ok": "!Main_Page"},
    {"id": "python_downloads", "url": "https://www.python.org", "goal": "Go to the Downloads page", "ok": "/downloads"},
    {"id": "pydocs_tutorial", "url": "https://docs.python.org/3/", "goal": "Open the Python tutorial", "ok": "tutorial"},
    {"id": "gh_issues", "url": "https://github.com/huggingface/transformers", "goal": "Open the Issues tab of this repository", "ok": "/issues"},
    {"id": "hn_newest", "url": "https://news.ycombinator.com", "goal": "Show the newest submissions", "ok": "newest"},
    {"id": "ddg_search", "url": "https://duckduckgo.com", "goal": "Search for 'playwright python'", "text": "playwright python", "ok": "q=playwright"},
    {"id": "pypi_search", "url": "https://pypi.org", "goal": "Search PyPI for the package 'requests'", "text": "requests", "ok": "q=requests"},
    {"id": "rust_learn", "url": "https://www.rust-lang.org", "goal": "Open the Learn page", "ok": "/learn"},
]


def judge(ok: str, url: str) -> bool:
    return (ok[1:] not in url) if ok.startswith("!") else (ok in url)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8811"); ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="results/vision/harness"); ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--k", type=int, default=30); ap.add_argument("--only", default=None)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rows = []
    for t in TASKS:
        if a.only and t["id"] not in a.only.split(","):
            continue
        args = SimpleNamespace(url=t["url"], goal=t["goal"], server=a.server, text=t.get("text", ""), max_steps=a.max_steps,
                               min_confidence=0.3, log=str(out / f"{a.tag}_bench_{t['id']}.jsonl"), headless=True, k=a.k, done_threshold=0.9)
        Path(args.log).unlink(missing_ok=True)
        t0 = time.time()
        try:
            r = run_task(args); err = None
        except Exception as e:  # keep the bench going
            r = {"final_url": "", "steps": [], "stop": "error"}; err = f"{type(e).__name__}: {str(e)[:100]}"
        succ = judge(t["ok"], r["final_url"])
        # first-action success: judged after step 0 alone (url after the first action = start of step 1, else final)
        first_url = r["steps"][1]["url"] if len(r["steps"]) > 1 else r["final_url"]
        row = {"id": t["id"], "success": succ, "first_action_success": judge(t["ok"], first_url), "n_steps": len(r["steps"]), "stop": r["stop"],
               "final_url": r["final_url"][:100], "model_ms": [s["model_ms"] for s in r["steps"]], "secs": round(time.time() - t0, 1), "error": err}
        rows.append(row); print(json.dumps(row), flush=True)
    n = len(rows); s = sum(r["success"] for r in rows); f = sum(r["first_action_success"] for r in rows)
    ms = [m for r in rows for m in r["model_ms"]]
    summary = {"tag": a.tag, "n": n, "success": s, "first_action_success": f, "mean_model_ms": round(sum(ms) / max(1, len(ms)), 1), "rows": rows}
    (out / f"{a.tag}_bench.json").write_text(json.dumps(summary, indent=1))
    print(f"{a.tag}: end-state success {s}/{n}, first-action success {f}/{n}, mean model {summary['mean_model_ms']} ms")


if __name__ == "__main__":
    main()
