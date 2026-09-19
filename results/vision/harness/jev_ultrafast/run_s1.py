"""Same as run.py but observes with screenshots so an image-capable decision server can use them.

S1_SCREENSHOT=1 TYPESAFE_BASE_URL=http://127.0.0.1:8811/v1/systemone uv run --env-file .env python examples/run_s1.py --url URL --goal '...'
"""

import argparse

from jev_ultrafast import Agent

parser = argparse.ArgumentParser()
parser.add_argument("--url", required=True)
parser.add_argument("--goal", action="append", required=True)
args = parser.parse_args()

with Agent(args.url, args.goal, screenshots=True) as agent:
    for state in agent.run():
        d = state.get("decision") or {}
        print(f"{state['elapsed_ms']:>5} ms  {len(state['history'])} actions  {state['status']}  {d.get('operation','')} {d.get('target','')} conf={d.get('confidence','')} model_ms={d.get('latency_ms','')}")
    print(state["page"]["url"])
