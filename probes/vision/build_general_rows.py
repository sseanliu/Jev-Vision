"""General-vision training rows (typed questions from public TRAIN splits) for the V7 general mix.

Sources (train splits only; the general-track eval uses val/test splits of A-OKVQA, Food-101, NLVR2 and the
eval-only POPE and MME, so nothing here overlaps an eval item):
  aokvqa   HuggingFaceM4/A-OKVQA train      4-way choice
  food101  ethz/food101 train               20-way choice (gold + 19 sampled)
  nlvr2    lmms-lab/NLVR2 balanced/unbalanced dev? no: train is not on that repo; we use HuggingFaceM4/NLVR2 if loadable, else skip
  vqa_yn   HuggingFaceM4/VQAv2 train        yes/no questions only -> noul
  gqa_yn   lmms-lab/GQA train_balanced      yes/no questions -> noul (if loadable)
Row format is model/train_vl.py's (images, state, questions[{qid,qtype,instructions,criteria}], targets).

python build_general_rows.py --out /workspace/vision/general_rows --n-aokvqa 6000 --n-food 4000 --n-vqa 8000 --n-gqa 4000 --n-nlvr2 6000
"""

import argparse
import json
import random
from pathlib import Path

from PIL import Image

MAX_SIDE = 1024
YN_INSTR = ["Answer the question about the image: {q}", "{q}", "Looking at the image, {q}"]


def save(img, path: Path):
    img = img.convert("RGB"); w, h = img.size; s = MAX_SIDE / max(w, h)
    if s < 1:
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BICUBIC)
    path.parent.mkdir(parents=True, exist_ok=True); img.save(path, "JPEG", quality=88)


def stream(name, split, n, seed, **kw):
    from datasets import load_dataset
    ds = load_dataset(name, split=split, streaming=True, **kw).shuffle(seed=seed, buffer_size=2000)
    out = []
    for ex in ds:
        out.append(ex)
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--n-aokvqa", type=int, default=6000); ap.add_argument("--n-food", type=int, default=4000)
    ap.add_argument("--n-vqa", type=int, default=8000); ap.add_argument("--n-gqa", type=int, default=4000); ap.add_argument("--n-nlvr2", type=int, default=6000)
    a = ap.parse_args(); out = Path(a.out); img = out / "img"; out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed); rows = []; n = {}

    def row(src, images, state, q, target):
        rows.append({"images": [str(p) for p in images], "state": state, "questions": [q], "targets": {q["qid"]: target}, "meta": {"source": src}})
        n[src] = n.get(src, 0) + 1

    # A-OKVQA train, 4-way choice (qid "choice" so it shares the head with other choice questions)
    try:
        for i, ex in enumerate(stream("HuggingFaceM4/A-OKVQA", "train", a.n_aokvqa, a.seed)):
            p = img / f"aokvqa_{i}.jpg"; save(ex["image"], p); keys = "ABCD"
            row("aokvqa", [p], "A photograph.", {"qid": "vqa_choice", "qtype": "choice", "instructions": ex["question"].strip(), "criteria": {keys[k]: c for k, c in enumerate(ex["choices"])}}, keys[int(ex["correct_choice_idx"])])
    except Exception as e:
        print("aokvqa failed:", e)
    # Food-101 train, 20-way
    try:
        from datasets import load_dataset
        names = load_dataset("ethz/food101", split="train", streaming=True).features["label"].names
        for i, ex in enumerate(stream("ethz/food101", "train", a.n_food, a.seed)):
            p = img / f"food_{i}.jpg"; save(ex["image"], p); gold = names[int(ex["label"])]
            opts = rng.sample([x for x in names if x != gold], 19) + [gold]; rng.shuffle(opts)
            row("food101", [p], "A photograph.", {"qid": "vqa_choice", "qtype": "choice", "instructions": "Which dish is shown in the photo?", "criteria": {o: o.replace("_", " ") for o in opts}}, gold)
    except Exception as e:
        print("food101 failed:", e)
    # VQAv2 train, yes/no only
    try:
        got = 0
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceM4/VQAv2", split="train", streaming=True).shuffle(seed=a.seed, buffer_size=2000)
        for ex in ds:
            ans = str(ex.get("multiple_choice_answer", "")).strip().lower()
            if ans not in {"yes", "no"}:
                continue
            p = img / f"vqa_{got}.jpg"; save(ex["image"], p)
            row("vqa_yn", [p], "A photograph.", {"qid": "vqa_yn", "qtype": "noul", "instructions": rng.choice(YN_INSTR).format(q=ex["question"].strip())}, int(ans == "yes"))
            got += 1
            if got >= a.n_vqa:
                break
    except Exception as e:
        print("vqa failed:", e)
    # GQA balanced train, yes/no only
    try:
        got = 0
        from datasets import load_dataset
        ds = load_dataset("lmms-lab/GQA", "train_balanced_instructions", split="train", streaming=True).shuffle(seed=a.seed, buffer_size=2000)
        imgs = None
        for ex in ds:
            ans = str(ex.get("answer", "")).strip().lower()
            if ans not in {"yes", "no"}:
                continue
            if "image" not in ex:
                break  # instructions config has no images; skip GQA
            p = img / f"gqa_{got}.jpg"; save(ex["image"], p)
            row("gqa_yn", [p], "A photograph.", {"qid": "vqa_yn", "qtype": "noul", "instructions": rng.choice(YN_INSTR).format(q=ex["question"].strip())}, int(ans == "yes"))
            got += 1
            if got >= a.n_gqa:
                break
        if got == 0:
            print("gqa: no image field, skipped")
    except Exception as e:
        print("gqa failed:", e)
    # NLVR2 train (two images)
    try:
        for i, ex in enumerate(stream("lmms-lab/NLVR2", "unbalanced_dev", a.n_nlvr2, a.seed)):
            pl = img / f"nlvr2_{i}_l.jpg"; pr = img / f"nlvr2_{i}_r.jpg"; save(ex["left_image"], pl); save(ex["right_image"], pr)
            row("nlvr2", [pl, pr], f"Statement: {ex['question'].strip()}", {"qid": "pair_true", "qtype": "noul", "instructions": "Is the statement true of the pair of images (left image first, right image second)?"}, int(str(ex["answer"]).strip().lower() == "true"))
    except Exception as e:
        print("nlvr2 failed:", e)

    rng.shuffle(rows)
    with open(out / "general.train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(rows), "rows", n, "->", out / "general.train.jsonl")


if __name__ == "__main__":
    main()
