"""Convert public datasets into the unified typed-question format.

One JSON object per line:
{
  "source": "arc_challenge", "split": "train",
  "state": "...",
  "questions": [{"qid": "q", "qtype": "choice", "instructions": "...", "criteria": {...}}],
  "targets": {"q": "key"}            # choice: key; noul: true/false; score: level index
}

Only train/validation splits of datasets we are allowed to train on. MMLU,
MMLU-Pro, ANLI, GPQA, CLINC150 are held out and handled in eval.py.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).resolve().parent / "jsonl"


def emit(rows, name, split):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.{split}.jsonl"
    with path.open("w") as f:
        for r in rows:
            r["source"], r["split"] = name, split
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name:16s} {split:11s} {len(rows):7d} -> {path.name}")


def choice(state, instructions, options: dict, answer_key, qid="q"):
    return {"state": state,
            "questions": [{"qid": qid, "qtype": "choice", "instructions": instructions, "criteria": options}],
            "targets": {qid: answer_key}}


def noul(state, instructions, answer: bool, qid="q"):
    return {"state": state,
            "questions": [{"qid": qid, "qtype": "noul", "instructions": instructions}],
            "targets": {qid: bool(answer)}}


def score(state, instructions, levels: list, answer_idx: int, qid="q"):
    return {"state": state,
            "questions": [{"qid": qid, "qtype": "score", "instructions": instructions, "criteria": levels}],
            "targets": {qid: int(answer_idx)}}


# ---------------------------------------------------------------- converters

def conv_arc(split, limit):
    rows = []
    for cfg in ("ARC-Easy", "ARC-Challenge"):
        ds = load_dataset("allenai/ai2_arc", cfg, split=split)
        for ex in ds.select(range(min(limit, len(ds)))):
            opts = {lab.lower(): txt for lab, txt in zip(ex["choices"]["label"], ex["choices"]["text"])}
            rows.append(choice(ex["question"], "Which option correctly answers the question in the state?",
                               opts, ex["answerKey"].lower()))
    return rows


def conv_openbookqa(split, limit):
    ds = load_dataset("allenai/openbookqa", "main", split=split)
    rows = []
    for ex in ds.select(range(min(limit, len(ds)))):
        opts = {lab.lower(): txt for lab, txt in zip(ex["choices"]["label"], ex["choices"]["text"])}
        rows.append(choice(ex["question_stem"], "Which option correctly completes or answers the state?",
                           opts, ex["answerKey"].lower()))
    return rows


def conv_commonsenseqa(split, limit):
    ds = load_dataset("tau/commonsense_qa", split=split)
    rows = []
    for ex in ds.select(range(min(limit, len(ds)))):
        if not ex["answerKey"]:
            continue
        opts = {lab.lower(): txt for lab, txt in zip(ex["choices"]["label"], ex["choices"]["text"])}
        rows.append(choice(ex["question"], "Which option is the most sensible answer?", opts, ex["answerKey"].lower()))
    return rows


def conv_sciq(split, limit):
    ds = load_dataset("allenai/sciq", split=split)
    rows = []
    rng = random.Random(0)
    for ex in ds.select(range(min(limit, len(ds)))):
        opts = [ex["correct_answer"], ex["distractor1"], ex["distractor2"], ex["distractor3"]]
        rng.shuffle(opts)
        crit = {f"o{i}": o for i, o in enumerate(opts)}
        key = next(k for k, v in crit.items() if v == ex["correct_answer"])
        state = (ex["support"] + "\n\n" if ex["support"] else "") + ex["question"]
        rows.append(choice(state, "Which option correctly answers the question in the state?", crit, key))
    return rows


def conv_boolq(split, limit):
    ds = load_dataset("google/boolq", split=split)
    rows = []
    for ex in ds.select(range(min(limit, len(ds)))):
        rows.append(noul(ex["passage"], ex["question"].rstrip("?") + "?", ex["answer"]))
    return rows


def conv_agnews(split, limit):
    ds = load_dataset("fancyzhx/ag_news", split=split)
    labels = {"world": "World news", "sports": "Sports", "business": "Business",
              "scitech": "Science and technology"}
    keys = list(labels)
    rows = []
    for ex in ds.select(range(min(limit, len(ds)))):
        rows.append(choice(ex["text"], "Which section does this news article belong to?", labels, keys[ex["label"]]))
    return rows


def conv_dbpedia(split, limit):
    ds = load_dataset("fancyzhx/dbpedia_14", split=split)
    names = ["company", "educational_institution", "artist", "athlete", "office_holder",
             "mean_of_transportation", "building", "natural_place", "village", "animal",
             "plant", "album", "film", "written_work"]
    labels = {n: n.replace("_", " ") for n in names}
    rows = []
    for ex in ds.select(range(min(limit, len(ds)))):
        rows.append(choice(ex["content"], "What kind of entity does this text describe?", labels, names[ex["label"]]))
    return rows


def conv_trec(split, limit):
    ds = load_dataset("CogComp/trec", split=split, revision="refs/convert/parquet")
    coarse = ["abbreviation", "entity", "description", "human", "location", "numeric"]
    labels = {"abbreviation": "Asks about an abbreviation", "entity": "Asks about an entity or thing",
              "description": "Asks for a description or definition", "human": "Asks about a person or group",
              "location": "Asks about a location", "numeric": "Asks for a number, date or quantity"}
    rows = []
    for ex in ds.select(range(min(limit, len(ds)))):
        rows.append(choice(ex["text"], "What type of answer is this question asking for?", labels,
                           coarse[ex["coarse_label"]]))
    return rows


def conv_banking77(split, limit):
    ds = load_dataset("mteb/banking77", split=split)
    names = sorted({ex["label_text"] for ex in ds})
    labels = {n: n.replace("_", " ") for n in names}
    rows = []
    for ex in ds.select(range(min(limit, len(ds)))):
        rows.append(choice(ex["text"], "Which banking intent does this customer message express?", labels,
                           ex["label_text"]))
    return rows


def conv_sst5(split, limit):
    ds = load_dataset("SetFit/sst5", split=split)
    levels = ["very negative", "negative", "neutral", "positive", "very positive"]
    rows = []
    for ex in ds.select(range(min(limit, len(ds)))):
        rows.append(score(ex["text"], "How positive is the sentiment of this review sentence?", levels, ex["label"]))
    return rows


def conv_mnli(split, limit):
    ds = load_dataset("nyu-mll/glue", "mnli", split=split)
    labels = {"entailment": "The hypothesis follows from the premise",
              "neutral": "The hypothesis may or may not be true given the premise",
              "contradiction": "The hypothesis contradicts the premise"}
    keys = list(labels)
    rows = []
    for ex in ds.select(range(min(limit, len(ds)))):
        if ex["label"] < 0:
            continue
        state = f"Premise: {ex['premise']}\nHypothesis: {ex['hypothesis']}"
        rows.append(choice(state, "What is the relationship between the hypothesis and the premise?", labels,
                           keys[ex["label"]]))
    return rows


CONVERTERS = {
    "arc": (conv_arc, {"train": "train", "validation": "validation"}),
    "openbookqa": (conv_openbookqa, {"train": "train", "validation": "validation"}),
    "commonsenseqa": (conv_commonsenseqa, {"train": "train", "validation": "validation"}),
    "sciq": (conv_sciq, {"train": "train", "validation": "validation"}),
    "boolq": (conv_boolq, {"train": "train", "validation": "validation"}),
    "agnews": (conv_agnews, {"train": "train", "validation": "test"}),
    "dbpedia": (conv_dbpedia, {"train": "train", "validation": "test"}),
    "trec": (conv_trec, {"train": "train", "validation": "test"}),
    "banking77": (conv_banking77, {"train": "train", "validation": "test"}),
    "sst5": (conv_sst5, {"train": "train", "validation": "validation"}),
    "mnli": (conv_mnli, {"train": "train", "validation": "validation_matched"}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--limit-train", type=int, default=20000)
    ap.add_argument("--limit-val", type=int, default=1000)
    a = ap.parse_args()
    for name, (fn, splits) in CONVERTERS.items():
        if a.only and name not in a.only:
            continue
        for our_split, hf_split in splits.items():
            limit = a.limit_train if our_split == "train" else a.limit_val
            try:
                emit(fn(hf_split, limit), name, our_split)
            except Exception as e:
                print(f"{name:16s} {our_split:11s} FAILED: {type(e).__name__}: {str(e)[:120]}")


if __name__ == "__main__":
    main()
