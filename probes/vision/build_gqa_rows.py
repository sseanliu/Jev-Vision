"""GQA yes/no training rows (single-image noul) for the V7 general mix; images are Visual Genome, so no overlap with
the COCO-based POPE eval. Joins lmms-lab/GQA train_balanced_instructions with train_balanced_images by imageId.

python build_gqa_rows.py --out /workspace/vision/general_rows --n 8000 --per-image 2
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=8000); ap.add_argument("--per-image", type=int, default=2)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args(); rng = random.Random(a.seed); out = Path(a.out); img_dir = out / "img"
    from datasets import load_dataset
    ins = load_dataset("lmms-lab/GQA", "train_balanced_instructions", split="train", streaming=True)
    want = {}; n_yes = n_no = 0
    for ex in ins:
        ans = ex["answer"].strip().lower()
        if ans not in {"yes", "no"}:
            continue
        # keep yes/no balanced
        if (ans == "yes" and n_yes > n_no + 50) or (ans == "no" and n_no > n_yes + 50):
            continue
        lst = want.setdefault(ex["imageId"], [])
        if len(lst) >= a.per_image:
            continue
        lst.append((ex["question"].strip(), int(ans == "yes")))
        n_yes += ans == "yes"; n_no += ans == "no"
        if n_yes + n_no >= a.n:
            break
    print("questions", n_yes + n_no, "yes", n_yes, "no", n_no, "images", len(want))
    imgs = load_dataset("lmms-lab/GQA", "train_balanced_images", split="train", streaming=True)
    rows = []; seen = 0
    for ex in imgs:
        qs = want.get(ex["id"])
        if not qs:
            continue
        p = img_dir / f"gqa_{ex['id']}.jpg"; save(ex["image"], p); seen += 1
        for k, (q, y) in enumerate(qs):
            rows.append({"images": [str(p)], "state": "A photograph.", "questions": [{"qid": "vqa_yn", "qtype": "noul", "instructions": rng.choice(YN_INSTR).format(q=q)}], "targets": {"vqa_yn": y}, "meta": {"source": "gqa_yn", "image_id": ex["id"]}})
        if seen >= len(want):
            break
    rng.shuffle(rows)
    with open(out / "gqa_yn.train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(rows), "rows, images saved", seen, "->", out / "gqa_yn.train.jsonl")


if __name__ == "__main__":
    main()
