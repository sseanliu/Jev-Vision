"""Hang our decision model behind Jev Ultrafast as a per-step verifier and count what it would have caught.

Jev (or whatever TYPESAFE_BASE_URL points at) makes every decision exactly as in jev-ultrafast. This script drives the
Agent one command at a time and, without intervening, asks our server three questions per step:
  skip    before acting: is the chosen element already in the state the task needs (action unnecessary)?
  effect  after acting: did the last action change the page as intended? (before + after screenshots)
  done    after acting: has the goal been fully achieved?
It logs per-step verdicts next to Jev's own signals (page_changed, DONE) and the task's independent verifier.

Run from the jev-ultrafast checkout so its .env (Jev key, text helper) applies:
  cd ~/Github/jev-ultrafast && uv run --env-file .env python ~/Github/typesafe/harness/hang_behind.py --task flights \
      --verifier http://127.0.0.1:8811 --out ~/Github/typesafe/results/vision/harness/hang_behind/flights_jev_v5b
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

from jev_ultrafast import Agent

Q_SKIP = "Is this element already in the state the task needs, so it should be left alone?"
Q_EFFECT = "Did the last action change the page as intended?"
Q_DONE = "Has the goal been fully achieved, with nothing left to do?"

TASKS = {
    "flights": {
        "url": "https://www.google.com/travel/flights?hl=en",
        "goal": "Find one-way flights from Zurich to London on September 20, 2026, for one adult in economy. "
                "Stop when matching flight options are visible. Do not select or book a flight.",
    },
    "wiki": {
        "url": "https://en.wikipedia.org/wiki/Main_Page",
        "goal": "Open the Wikipedia article about the Eiffel Tower.",
    },
}


def ask(server, state, images, qid, instructions):
    body = {"model": "s1", "state": state, "questions": {qid: {"type": "noul", "instructions": instructions}}}
    if len(images) == 1:
        body["image"] = images[0]
    else:
        body["images"] = images
    req = urllib.request.Request(server.rstrip("/") + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    r = json.load(urllib.request.urlopen(req, timeout=120))
    if "error" in r:
        raise RuntimeError(r["error"])
    return float(r["answers"][qid]["noul"]), round((time.perf_counter() - t) * 1000)


def history_text(history):
    lines = []
    for h in history:
        s = f"  - {h['kind']} on '{h['action']}'"
        if h.get("text"):
            s += f" typed '{h['text']}'"
        lines.append(s)
    return "Actions already taken:\n" + ("\n".join(lines) + "\n" if lines else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), default=None)
    ap.add_argument("--url"); ap.add_argument("--goal")
    ap.add_argument("--verifier", default="http://127.0.0.1:8811")
    ap.add_argument("--out", required=True)
    ap.add_argument("--show", action="store_true", help="bring the agent's tab to the front so a person can watch")
    ap.add_argument("--keep-open", action="store_true", help="leave the tab open at the end")
    a = ap.parse_args()
    url, goal = (TASKS[a.task]["url"], TASKS[a.task]["goal"]) if a.task else (a.url, a.goal)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    agent = Agent(url, goal, screenshots=True)
    if a.show:
        from browser_harness.helpers import cdp
        cdp("Target.activateTarget", targetId=agent.browser.target)
    steps = []
    try:
        while agent.state["status"] not in {"done", "blocked"}:
            st = agent.state
            try:
                agent.command("predict", {})
            except ValueError as e:
                print("stop:", e); break
            dec = st["decision"]
            before = st["page"]
            base = f"Task: {goal}\n" + history_text(st["history"])
            rec = {"step": len(st["history"]) + 1, "jev_choice": dec["choice"], "operation": dec["operation"],
                   "jev_confidence": dec["confidence"], "jev_ms": dec["latency_ms"], "url_before": before["url"]}
            if dec["choice"] in {"DONE", "BLOCKED"}:
                p, ms = ask(a.verifier, base, [before["screenshot"]], "done", Q_DONE)
                rec.update(jev_terminal=dec["choice"], done_p=p, done_ms=ms)
                steps.append(rec)
                print(f"step {rec['step']:>2}  Jev says {dec['choice']}   verifier done={p:.2f}", flush=True)
                try:
                    agent.command("act", {"fingerprint": before["fingerprint"]})
                except Exception as e:  # StalePage on a terminal decision: the run still ends here
                    rec.update(act_error=f"{type(e).__name__}: {e}")
                break
            action = next(x for x in before["actions"] if x["id"] == dec["choice"])
            label = action["label"].strip()
            p_skip, ms = ask(a.verifier, base + f"Candidate: {action['kind']} '{label}'\n", [before["screenshot"]], "skip", Q_SKIP)
            rec.update(action=label, kind=action["kind"], skip_p=p_skip, skip_ms=ms)
            try:
                agent.command("act", {"fingerprint": before["fingerprint"]})
            except Exception as e:  # StalePage or budget: record and continue the loop
                rec.update(act_error=f"{type(e).__name__}: {e}"); steps.append(rec)
                print(f"step {rec['step']:>2}  act error {e}", flush=True)
                if "budget" in str(e).lower():
                    break
                continue
            after = st["page"]; h = st["history"][-1]
            base2 = f"Task: {goal}\n" + history_text(st["history"])
            p_eff, ms_e = ask(a.verifier, base2 + f"Last action: {h['kind']} on '{label}'\n", [before["screenshot"], after["screenshot"]], "effect", Q_EFFECT)
            p_done, ms_d = ask(a.verifier, base2, [after["screenshot"]], "done", Q_DONE)
            rec.update(page_changed=h.get("page_changed"), url_after=after["url"], effect_p=p_eff, effect_ms=ms_e,
                       done_p=p_done, done_ms=ms_d, elapsed_ms=st["elapsed_ms"])
            steps.append(rec)
            flags = []
            if p_skip > 0.5: flags.append("SKIP")
            if p_eff < 0.5: flags.append("NO-EFFECT")
            if p_done > 0.5: flags.append("DONE?")
            print(f"step {rec['step']:>2}  {h['kind']:5} {label[:40]:40} changed={h.get('page_changed')}  "
                  f"skip={p_skip:.2f} effect={p_eff:.2f} done={p_done:.2f}  {' '.join(flags)}", flush=True)
    finally:
        snap = agent.snapshot()
        verification = None
        if a.task == "flights":
            sys.path.insert(0, str(Path.home() / "Github/jev-ultrafast/examples"))
            from flights_s1 import verify  # independent page checks
            verification = verify(snap["page"])
        summary = {
            "task": a.task or url, "goal": goal, "status": snap["status"], "steps": len(snap["history"]),
            "elapsed_ms": snap["elapsed_ms"],
            "jev_no_change_steps": sum(1 for h in snap["history"] if h.get("page_changed") is False),
            "verifier_skip_flags": sum(1 for s in steps if s.get("skip_p", 0) > 0.5 and "effect_p" in s),
            "verifier_no_effect_flags": sum(1 for s in steps if "effect_p" in s and s["effect_p"] < 0.5),
            "verifier_done_first_step": next((s["step"] for s in steps if s.get("done_p", 0) > 0.5), None),
            "jev_terminal": next((s.get("jev_terminal") for s in steps if s.get("jev_terminal")), None),
            "verifier_done_at_terminal": next((s.get("done_p") for s in steps if s.get("jev_terminal")), None),
            "verification": verification,
            "effect_vs_page_changed_agreement": (
                sum(1 for s in steps if "effect_p" in s and (s["effect_p"] > 0.5) == bool(s.get("page_changed")))
                / max(1, sum(1 for s in steps if "effect_p" in s))),
        }
        (out / "steps.jsonl").write_text("\n".join(json.dumps(s) for s in steps) + "\n")
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary, indent=1))
        if not a.keep_open:
            agent.close()


if __name__ == "__main__":
    main()
