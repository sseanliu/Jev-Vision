"""General-track items: typed vision decisions from public datasets, in the step-verifier item format.

Sources (sampled with a fixed seed, images resized to <= 1024 px on the long side):
  pope     lmms-lab/POPE            object presence, yes/no                      -> noul
  mme      lmms-lab/MME             perception/cognition yes/no                  -> noul
  aokvqa   HuggingFaceM4/A-OKVQA    4-way multiple choice                        -> choice
  food101  ethz/food101             dish classification, 20 options incl. gold   -> choice
  nlvr2    HuggingFaceM4/NLVR2      two images + statement, true/false           -> noul (multi-image)

Writes <out>/items.jsonl (benchmark item schema, track "general") and <out>/general.rows.jsonl (training-row
schema for model/eval_schema.py), images under <out>/images/.

python bench/build_general_items.py --out /workspace/bench/general --per-source 300 --seed 11
"""

import argparse
import json
import random
from pathlib import Path

from PIL import Image

MAX_SIDE = 1024


def save(img, path: Path):
    img = img.convert("RGB")
    w, h = img.size
    s = MAX_SIDE / max(w, h)
    if s < 1:
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BICUBIC)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", quality=90)


def load(name, split, n, seed, **kw):
    from datasets import load_dataset
    ds = load_dataset(name, split=split, streaming=True, **kw)
    # take a fixed prefix then sample: streaming datasets are not shuffleable across shards cheaply
    pool = []
    for i, ex in enumerate(ds):
        pool.append(ex)
        if len(pool) >= n * 4:
            break
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--per-source", type=int, default=300); ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--sources", default="pope,mme,aokvqa,food101,nlvr2")
    a = ap.parse_args(); out = Path(a.out); img_dir = out / "images"; out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed); items = []; counts = {}

    def add(src, i, qtype, instructions, images, label, state="", criteria=None, meta=None):
        it = {"id": f"{src}_{i}", "track": "general", "source": src, "question": src, "type": qtype, "instructions": instructions,
              "state": state, "images": [str(p.relative_to(out)) for p in images], "label": label}
        if criteria:
            it["criteria"] = criteria
        if meta:
            it["meta"] = meta
        items.append(it); counts[src] = counts.get(src, 0) + 1

    srcs = a.sources.split(",")
    if "pope" in srcs:
        try:
            for i, ex in enumerate(load("lmms-lab/POPE", "test", a.per_source, a.seed)):
                p = img_dir / f"pope_{i}.jpg"; save(ex["image"], p)
                add("pope", i, "noul", ex["question"].strip(), [p], int(str(ex["answer"]).strip().lower().startswith("y")), meta={"category": ex.get("category")})
        except Exception as e:
            print("pope failed:", e)
    if "mme" in srcs:
        try:
            for i, ex in enumerate(load("lmms-lab/MME", "test", a.per_source, a.seed)):
                p = img_dir / f"mme_{i}.jpg"; save(ex["image"], p)
                q = ex["question"].replace("Please answer yes or no.", "").strip()
                add("mme", i, "noul", q, [p], int(str(ex["answer"]).strip().lower().startswith("y")), meta={"category": ex.get("category")})
        except Exception as e:
            print("mme failed:", e)
    if "aokvqa" in srcs:
        try:
            for i, ex in enumerate(load("HuggingFaceM4/A-OKVQA", "validation", a.per_source, a.seed)):
                p = img_dir / f"aokvqa_{i}.jpg"; save(ex["image"], p)
                keys = "ABCD"; crit = {keys[k]: c for k, c in enumerate(ex["choices"])}
                add("aokvqa", i, "choice", ex["question"].strip(), [p], keys[int(ex["correct_choice_idx"])], criteria=crit)
        except Exception as e:
            print("aokvqa failed:", e)
    if "food101" in srcs:
        try:
            from datasets import load_dataset
            names = load_dataset("ethz/food101", split="validation", streaming=True).features["label"].names
            for i, ex in enumerate(load("ethz/food101", "validation", a.per_source, a.seed)):
                p = img_dir / f"food101_{i}.jpg"; save(ex["image"], p)
                gold = names[int(ex["label"])]
                opts = rng.sample([n for n in names if n != gold], 19) + [gold]; rng.shuffle(opts)
                crit = {o: o.replace("_", " ") for o in opts}
                add("food101", i, "choice", "Which dish is shown in the photo?", [p], gold, criteria=crit)
        except Exception as e:
            print("food101 failed:", e)
    if "nlvr2" in srcs:
        loaded = False
        for name, split in [("HuggingFaceM4/NLVR2", "validation"), ("lmms-lab/NLVR2", "test")]:
            try:
                for i, ex in enumerate(load(name, split, a.per_source, a.seed)):
                    left = ex.get("left_image") or ex.get("image_0") or ex.get("images", [None])[0]
                    right = ex.get("right_image") or ex.get("image_1") or ex.get("images", [None, None])[1]
                    pl = img_dir / f"nlvr2_{i}_left.jpg"; pr = img_dir / f"nlvr2_{i}_right.jpg"; save(left, pl); save(right, pr)
                    sent = ex.get("sentence") or ex.get("statement") or ex.get("question")
                    lab = ex.get("label"); lab = int(lab) if isinstance(lab, (int, bool)) else int(str(lab).strip().lower() in {"true", "1", "yes"})
                    add("nlvr2", i, "noul", "Is the statement true of the pair of images (left image first, right image second)?", [pl, pr], lab, state=f"Statement: {sent}")
                loaded = True; break
            except Exception as e:
                print(name, "failed:", e)
        if not loaded:
            print("nlvr2 skipped")

    with open(out / "items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    # rows for model/eval_schema.py
    with open(out / "general.rows.jsonl", "w") as f:
        for it in items:
            q = {"qid": it["question"], "qtype": it["type"], "instructions": it["instructions"]}
            if it["type"] == "choice":
                q["criteria"] = it["criteria"]
            f.write(json.dumps({"images": [str(out / p) for p in it["images"]], "state": it["state"] or "A photograph.",
                                "questions": [q], "targets": {it["question"]: it["label"]}, "meta": {"id": it["id"], "source": it["source"]}}, ensure_ascii=False) + "\n")
    print(len(items), "items", counts)


if __name__ == "__main__":
    main()
