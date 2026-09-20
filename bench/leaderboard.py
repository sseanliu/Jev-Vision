"""Score every baseline in bench/baselines/ (or a given list) on an items file and write the leaderboard.

Also builds the `rules` baseline on the fly: effect = 1 if the URL changed or the element table changed, skip and done
= constant "no" (p=0.1). It shows what the environment signals alone give a text-only system and what the label adds
(pixel change, applied values, target-page reachability).

python bench/leaderboard.py --items bench/v0_items.jsonl --name v0 --ms v5=190 --ms v5b=190
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_submission import score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"jev": "Jev 1.13 (text only, candidate table)", "claude": "Claude Sonnet 5 (screenshots, verbal probability)",
          "djev": "DJev (DiffusionGemma-as-Jev API, screenshot)", "v5": "V5 (ours, 8B, screenshots)",
          "v5b": "V5b (ours, 8B, screenshots)", "rules": "rules: URL/DOM change for effect, constant no for skip/done",
          "v5b_pixel": "V5b on the pixel track (zero-shot, no candidate table)"}


def rules_preds(items):
    out = {}
    for it in items:
        if it["type"] != "noul":
            continue
        if it["question"] == "effect":
            changed = (it.get("url_before") != it.get("url_after")) or (it.get("elements_before") != it.get("elements_after"))
            out[it["id"]] = {"id": it["id"], "p": 0.9 if changed else 0.1}
        else:
            out[it["id"]] = {"id": it["id"], "p": 0.1}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--baselines", nargs="*", default=None, help="names under bench/baselines (default: all)")
    ap.add_argument("--ms", action="append", default=[], help="name=ms self-reported latency")
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.items)]
    ms = dict(kv.split("=") for kv in a.ms)
    names = a.baselines or sorted(p.stem for p in (ROOT / "bench/baselines").glob("*.jsonl"))
    reports = {}
    for n in names:
        preds = {}
        for l in open(ROOT / "bench/baselines" / f"{n}.jsonl"):
            d = json.loads(l); preds[d["id"]] = d
        r = score(items, preds, LABELS.get(n, n), float(ms[n]) if n in ms else None)
        if all(q not in r for q in ("skip", "effect", "done")):
            continue
        reports[n] = r
    reports["rules"] = score(items, rules_preds(items), LABELS["rules"], 0.0)
    json.dump(reports, open(ROOT / "bench" / f"{a.name}_leaderboard.json", "w"), indent=1)
    qs = ["skip", "effect", "done"]
    lines = ["| system | " + " | ".join(f"{q} acc (pos/neg)" for q in qs) + " | ECE (skip/effect/done) | AUROC | sel@90 | ms |",
             "|---|" + "---|" * (len(qs) + 4)]
    for r in reports.values():
        cells = [f"{r[q]['acc']:.3f} ({r[q]['acc_pos']:.2f}/{r[q]['acc_neg']:.2f}) n={r[q]['n']}" if q in r else "-" for q in qs]
        e = "/".join(f"{r[q]['ece']:.3f}" if q in r else "-" for q in qs)
        au = "/".join(f"{r[q]['auroc']:.3f}" if q in r else "-" for q in qs)
        s = "/".join(f"{r[q]['sel90']:.3f}" if q in r else "-" for q in qs)
        lines.append(f"| {r['_name']} | " + " | ".join(cells) + f" | {e} | {au} | {s} | {r['_ms'] if r['_ms'] is not None else 'n/a'} |")
    (ROOT / "bench" / f"{a.name}_leaderboard.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
