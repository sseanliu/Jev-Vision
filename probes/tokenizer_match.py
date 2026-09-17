"""Offline tokenizer comparison on Hume's 421 fingerprint probes.

tokens_jev(S) = input_tokens(S) - input_tokens(state="") per the file's note.
For each candidate tokenizer we count tokens of S in isolation and with a
leading newline/space (unknown template), and score a probe as agreeing if
either count matches. Agreement is reported per probe family, with and without
digit probes, to test the hypothesis: "o200k-family vocabulary with per-digit
pre-tokenisation".
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "hume" / "tokenizer-fingerprint.json"


def load_probes():
    d = json.loads(DATA.read_text())
    trials = d["trials"]
    controls = [t for t in trials if t["family"] == "control"]
    empties = [t["in_tok"] for t in controls if t["state"] == ""]
    base = empties[0] if empties else d["meta"]["empty_state_input_tokens_pilot"]
    probes = []
    for t in trials:
        if t["status"] != 200 or t["family"] == "control":
            continue
        probes.append(
            {
                "family": t["family"],
                "label": t["label"],
                "state": t["state"],
                "jev": t["in_tok"] - base,
                "has_digit": any(ch.isdigit() for ch in t["state"]),
            }
        )
    return base, probes


# --- candidate tokenizers -------------------------------------------------

def tiktoken_counter(name: str, per_digit: bool = False):
    enc = tiktoken.get_encoding(name)
    if not per_digit:
        return lambda s: len(enc.encode(s, disallowed_special=()))
    # Re-run with the numeric rule changed from \p{N}{1,3} to \p{N}: split the
    # string on digits first, tokenise the non-digit chunks normally, count each
    # digit as one token.
    def count(s: str) -> int:
        n = 0
        for chunk in re.split(r"(\d)", s):
            if chunk == "":
                continue
            if chunk.isdigit() and len(chunk) == 1:
                n += 1
            else:
                n += len(enc.encode(chunk, disallowed_special=()))
        return n
    return count


def hf_counter(repo: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=False)
    return lambda s: len(tok.encode(s, add_special_tokens=False))


def main() -> None:
    base, probes = load_probes()
    print(f"empty-state baseline input_tokens = {base}; probes = {len(probes)}")
    fams = defaultdict(int)
    for p in probes:
        fams[p["family"]] += 1
    print("families:", dict(fams), "| with digits:", sum(p["has_digit"] for p in probes))

    candidates = {
        "o200k_base": tiktoken_counter("o200k_base"),
        "o200k_harmony": tiktoken_counter("o200k_harmony"),
        "o200k_base+per-digit": tiktoken_counter("o200k_base", per_digit=True),
        "o200k_harmony+per-digit": tiktoken_counter("o200k_harmony", per_digit=True),
        "cl100k_base": tiktoken_counter("cl100k_base"),
    }
    hf_repos = {
        "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
        "Qwen3-8B": "Qwen/Qwen3-8B",
        "gpt-oss-120b": "openai/gpt-oss-120b",
        "DeepSeek-V3": "deepseek-ai/DeepSeek-V3",
        "Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
        "gemma-3-12b": "google/gemma-3-12b-it",
        "Mistral-Small-3.1": "mistralai/Mistral-Small-3.1-24B-Base-2503",
        "GLM-4.5": "zai-org/GLM-4.5",
        "Kimi-K2": "moonshotai/Kimi-K2-Instruct",
    }
    for name, repo in hf_repos.items():
        try:
            candidates[name] = hf_counter(repo)
        except Exception as e:  # gated or missing
            print(f"  skip {name}: {type(e).__name__}: {str(e)[:80]}")

    results = {}
    for name, count in candidates.items():
        agree = defaultdict(lambda: [0, 0])  # family -> [agree, total]
        agree_nodigit = [0, 0]
        agree_digit = [0, 0]
        misses = []
        for p in probes:
            s = p["state"]
            cands = {count(s)}
            for pre in ("\n", " "):
                try:
                    cands.add(count(pre + s) - count(pre))
                except Exception:
                    pass
            ok = p["jev"] in cands
            agree[p["family"]][1] += 1
            agree[p["family"]][0] += ok
            tgt = agree_digit if p["has_digit"] else agree_nodigit
            tgt[1] += 1
            tgt[0] += ok
            if not ok and len(misses) < 6:
                misses.append((p["label"][:40], p["jev"], sorted(cands)))
        total = sum(v[0] for v in agree.values()), sum(v[1] for v in agree.values())
        results[name] = {
            "total": total,
            "by_family": {k: tuple(v) for k, v in agree.items()},
            "no_digit": tuple(agree_nodigit),
            "digit": tuple(agree_digit),
            "misses": misses,
        }

    print("\n%-26s %9s %9s %9s   %s" % ("tokenizer", "all", "no-digit", "digit", "by family"))
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["total"][0]):
        fam = " ".join(f"{k}={v[0]}/{v[1]}" for k, v in sorted(r["by_family"].items()))
        print(
            "%-26s %4d/%-4d %4d/%-4d %4d/%-4d   %s"
            % (name, *r["total"], *r["no_digit"], *r["digit"], fam)
        )
    print("\nSample misses (label, jev_count, candidate_counts):")
    for name, r in results.items():
        print(f"  {name}:")
        for m in r["misses"][:4]:
            print(f"     {m}")
    out = ROOT / "results" / "tokenizer_match.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=1, default=list))
    print("saved", out)


if __name__ == "__main__":
    main()
