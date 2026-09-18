#!/usr/bin/env bash
# After gen_workflows.py finishes: label synthetic rows with Jev, report Jev's confidence
# distribution, and sync the labelled set to the pod for run 3b.
set -euo pipefail
cd "$(dirname "$0")/.."; . scripts/pod.env; set -a; . ./.env; set +a
mkdir -p model/data/jsonl_synth_jev
(cd probes && ../.venv/bin/python distill_jev.py --in ../model/data/jsonl_synth --out ../model/data/jsonl_synth_jev --per-source 100000 --workers 6)
python3 - <<'PY'
import json, collections
tot=0; b=collections.Counter(); amb=collections.Counter()
for l in open('model/data/jsonl_synth_jev/synth.train.jsonl'):
    r=json.loads(l)
    for q in r['questions']:
        t=r['targets'][q['qid']]; top=max(t,1-t) if q['qtype']=='noul' else max(t.values()); tot+=1
        k='<0.5' if top<0.5 else '<0.8' if top<0.8 else '<0.95' if top<0.95 else '>=0.95'; b[k]+=1
        amb[(bool(q.get('ambiguous')), top<0.8)]+=1
print("synthetic questions:", tot, "| Jev top-p buckets:", dict(b))
print("designed-ambiguous vs Jev-uncertain:", {f"amb={a},unc={u}": n for (a,u),n in sorted(amb.items())})
PY
rsync -rlt --no-o --no-g --no-p -z -e "ssh -i $HOME/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -p $POD_PORT" model/data/jsonl_synth_jev/ "root@$POD_HOST:/workspace/jev-probes/model/data/jsonl_synth_jev/"
echo "synced jsonl_synth_jev to pod"
