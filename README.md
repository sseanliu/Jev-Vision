# jev-probes

Black-box probes of TypeSafe AI's Jev (`jev-1.13.0`) and an offline analysis of
Archer Hume's evidence bundle. The goal is to pin down how Jev is built and
trained by behavioural fingerprinting, so that open replications can be scored
against the same fingerprints.

## Layout

- `probes/common.py` — API client. Keeps the request id and the
  `x-envoy-upstream-service-time` header for every call; retries on 429/5xx.
- `probes/k_sweep.py` — option-interaction vs list-dependent temperature.
  Appends K−4 absurd options to a four-option Choice question and tracks the
  pairwise log-odds among the originals. `--variant concrete` replaces the
  "Cannot tell" option with a concrete cause as a control.
- `probes/tokenizer_match.py` — replays Hume's 421 tokenizer probes against
  public tokenizers (tiktoken + Hugging Face), with and without digit probes.
- `probes/tokenizer_misses.py` — wider tokenizer sweep with the *direction* of
  every disagreement (does Jev count more or fewer tokens than the candidate?).
- `probes/counter_vs_model.py` — is the billing token counter the model's
  tokenizer? Compares server-time slope per billed token across text families
  the counter over-counts by different factors.
- `probes/capability.py` — capability/calibration baseline for version
  tracking: two-games puzzle, two-step word problems, modular exponentiation,
  three-digit products.
- `data/hume/` — mirror of the 11 evidence JSON files from
  [archerhume.com](https://archerhume.com/posts/jevs-architecture-unmasked/).
- `results/` — trial records (`*_trials.json`), logs, and summary tables.

## Setup

```sh
uv venv && source .venv/bin/activate
uv pip install tiktoken transformers requests numpy tokenizers
echo 'TYPESAFE_API_KEY=...' > .env
cd probes && python k_sweep.py            # ~180 requests
python k_sweep.py --variant concrete
python tokenizer_match.py                 # offline
python tokenizer_misses.py                # offline, downloads ~40 tokenizers
python capability.py
python counter_vs_model.py                # ~130 requests up to 28k tokens
```

Every script prints a summary at the end and can re-summarise a saved run
with `--summarize`.

## Findings so far (2026-09-17, jev-1.13.0)

1. **Option interaction is semantic, not a temperature.** Adding absurd
   options moves only the pairs that involve a "Cannot tell" option (about 2
   nats, monotone in K, saturating by K≈12–16); pairs among concrete options
   are stable, and replacing "Cannot tell" with a concrete cause removes the
   effect. A list-dependent softmax temperature cannot move one option
   selectively.
2. **The token counter is finer-grained than any frontier tokenizer.** On
   Hume's non-digit probes every disagreement with Qwen/o200k/gpt-oss has Jev
   counting *more* tokens (65/65, 68/68, 69/69). English words split into 2–5
   pieces, Cyrillic/Arabic/CJK per character. Total tokens over the 170 word
   probes place it with 32k–49k vocabularies (Llama-2, SmolLM2, Mistral-7B),
   not the 100k–200k vocabularies of frontier models. Either TypeSafe trained
   its own tokenizer, or the billing counter is not the model's tokenizer.
   `counter_vs_model.py` is designed to tell these apart.
3. The o200k/gpt-oss subset hypothesis is retracted: once digits are split,
   o200k and Qwen tie at 347/415, and both are missed in the same direction.

Detailed write-up: `~/Obsidian/Projects/TypeSafe/Jev-Training-Deep-Research.md`.
