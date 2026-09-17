"""Second tokenizer pass: (1) signed direction of misses (does Jev use more or
fewer tokens than the candidate?), (2) a wider sweep including small-vocabulary
and sentencepiece tokenizers, restricted to non-digit probes.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokenizer_match import hf_counter, load_probes, tiktoken_counter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

REPOS = {
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
    "gpt2": "openai-community/gpt2",
    "Llama-2-7b": "NousResearch/Llama-2-7b-hf",
    "Mistral-7B-v0.1": "mistralai/Mistral-7B-v0.1",
    "Llama-3.1-8B": "NousResearch/Meta-Llama-3.1-8B",
    "gemma-2-9b": "google/gemma-2-9b",
    "phi-4": "microsoft/phi-4",
    "SmolLM2-1.7B": "HuggingFaceTB/SmolLM2-1.7B",
    "OLMo-2-7B": "allenai/OLMo-2-1124-7B",
    "gpt-neox-20b": "EleutherAI/gpt-neox-20b",
    "bloom": "bigscience/bloom-560m",
    "xlm-roberta": "FacebookAI/xlm-roberta-base",
    "t5": "google-t5/t5-base",
    "bert-uncased": "google-bert/bert-base-uncased",
    "Yi-1.5-9B": "01-ai/Yi-1.5-9B",
    "InternLM2.5": "internlm/internlm2_5-7b",
    "Baichuan2": "baichuan-inc/Baichuan2-7B-Base",
    "Falcon-H1": "tiiuae/Falcon-H1-7B-Base",
    "Nemotron-H": "nvidia/Nemotron-H-8B-Base-8K",
    "MiniMax-M1": "MiniMaxAI/MiniMax-M1-80k",
    "Hunyuan-A13B": "tencent/Hunyuan-A13B-Instruct",
    "Seed-OSS-36B": "ByteDance-Seed/Seed-OSS-36B-Base",
    "Apertus-8B": "swiss-ai/Apertus-8B-2509",
    "Granite-3.3": "ibm-granite/granite-3.3-8b-base",
    "Cohere-Command-A": "CohereLabs/c4ai-command-a-03-2025",
    "Jamba-1.5": "ai21labs/AI21-Jamba-Mini-1.5",
    "Arcee-AFM": "arcee-ai/AFM-4.5B-Base",
    "Trillion-7B": "trillionlabs/Trillion-7B-preview",
    "EXAONE-4": "LGAI-EXAONE/EXAONE-4.0-32B",
    "Ling/Ring": "inclusionAI/Ring-mini-2.0",
    "Step3": "stepfun-ai/step3",
    "Mixtral-8x22B": "mistralai/Mixtral-8x22B-v0.1",
    "Mistral-Nemo": "mistralai/Mistral-Nemo-Base-2407",
    "Pixtral": "mistralai/Pixtral-12B-2409",
    "gpt-oss-20b": "openai/gpt-oss-20b",
}


def counts_for(count, s: str) -> set[int]:
    cands = {count(s)}
    for pre in ("\n", " "):
        try:
            cands.add(count(pre + s) - count(pre))
        except Exception:
            pass
    return cands


def main() -> None:
    base, probes = load_probes()
    nodigit = [p for p in probes if not p["has_digit"]]
    print(f"non-digit probes: {len(nodigit)}")

    cands = {
        "o200k+per-digit": tiktoken_counter("o200k_base", per_digit=True),
        "cl100k": tiktoken_counter("cl100k_base"),
    }
    for name, repo in REPOS.items():
        try:
            cands[name] = hf_counter(repo)
        except Exception as e:
            print(f"  skip {name}: {type(e).__name__}: {str(e)[:70]}")

    rows = []
    signed = {}
    for name, count in cands.items():
        agree = 0
        fam = defaultdict(lambda: [0, 0])
        diffs = Counter()
        for p in nodigit:
            cs = counts_for(count, p["state"])
            ok = p["jev"] in cs
            agree += ok
            fam[p["family"]][1] += 1
            fam[p["family"]][0] += ok
            if not ok:
                nearest = min(cs, key=lambda c: abs(c - p["jev"]))
                d = p["jev"] - nearest
                diffs["jev>cand" if d > 0 else "jev<cand"] += 1
        rows.append((agree, name, dict(fam), dict(diffs)))
        signed[name] = dict(diffs)
    rows.sort(reverse=True)
    print("\n%-20s %9s   %-44s %s" % ("tokenizer", "no-digit", "by family", "miss direction"))
    for agree, name, fam, diffs in rows:
        fs = " ".join(f"{k}={v[0]}/{v[1]}" for k, v in sorted(fam.items()))
        print("%-20s %4d/%-4d   %-44s %s" % (name, agree, len(nodigit), fs, diffs))

    # Word-family misses for the top candidate: which words, and how does Jev
    # split them relative to Qwen?
    top = rows[0][1]
    count = cands[top]
    print(f"\nWord/pretok-family misses for {top} (label, jev, {top}):")
    shown = 0
    for p in nodigit:
        if p["family"] == "run":
            continue
        cs = counts_for(count, p["state"])
        if p["jev"] not in cs:
            print(f"   {p['label'][:48]:<50} jev={p['jev']:<3} cand={sorted(cs)}")
            shown += 1
            if shown >= 40:
                break
    out = ROOT / "results" / "tokenizer_misses.json"
    out.write_text(json.dumps({"rows": rows, "signed": signed}, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
