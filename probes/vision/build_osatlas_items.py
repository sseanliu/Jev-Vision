"""Desktop `ground` items from OS-Atlas macOS screenshots (V3: a second format next to Mind2Web).

OS-Atlas gives, per screenshot, accessibility elements (name + normalised bbox + AX role), split
across several JSON rows per image. We pool the elements per image, pick a target, draw K-1
distractors from the same screen (preferring same-role and similar-text elements so the item is
not solved by role alone), render numbered set-of-mark boxes, and phrase the task with varied
templates ("Open Screen Time settings" style is not available, so tasks are of the form
"Click 'Screen Time'" with paraphrases). --none-frac items drop the target box.
Split is by image so test screens never appear in training.

python build_osatlas_items.py --root ../../model/data/vision/os_atlas/desktop_domain --out ../../model/data/vision/osatlas_items --k 9 --none-frac 0.1
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

COLORS = ["#e6194b", "#3cb44b", "#0082c8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#d2f53c",
          "#fabebe", "#008080", "#aa6e28", "#800000", "#808000", "#000080", "#808080", "#000000"]
TASKS = ["Click '{n}'.", "Select the '{n}' item.", "Open '{n}'.", "Activate the control labelled '{n}'.",
         "Choose '{n}' on this screen.", "Press '{n}'."]
INSTR = "Which numbered element should be acted on next to make progress on the task?"
ROLE = {"AXStaticText": "text", "AXTextField": "text field", "AXRadioButton": "radio button", "AXButton": "button",
        "AXGroup": "group", "AXCheckBox": "checkbox", "AXDockItem": "dock item", "AXPopUpButton": "popup button",
        "AXMenuItem": "menu item", "AXLink": "link", "AXImage": "image", "AXTab": "tab", "AXRow": "row", "AXCell": "cell"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../../model/data/vision/os_atlas/desktop_domain")
    ap.add_argument("--out", default="../../model/data/vision/osatlas_items")
    ap.add_argument("--k", type=int, default=9)
    ap.add_argument("--none-frac", type=float, default=0.1)
    ap.add_argument("--per-image", type=int, default=3, help="items per screenshot")
    ap.add_argument("--test-images", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    root = Path(a.root); out = Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    rows = json.load(open(root / "macos_splited.json"))
    by_img = defaultdict(dict)
    for r in rows:
        for e in r["elements"]:
            name = re.sub(r"\s+", " ", e["instruction"]).strip()
            if not name or len(name) > 60:
                continue
            key = (name, tuple(round(v, 3) for v in e["bbox"]))
            by_img[r["img_filename"]][key] = e
    imgs = [f for f, els in by_img.items() if len(els) >= a.k and (root / "macos_images" / f).exists()]
    rng.shuffle(imgs)
    test_imgs = set(imgs[: a.test_images])
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    except OSError:
        font = ImageFont.load_default()
    items, requests = {"train": [], "test": []}, {"train": [], "test": []}
    for f in imgs:
        els = list(by_img[f].values())
        im = Image.open(root / "macos_images" / f).convert("RGB"); W, H = im.size
        split = "test" if f in test_imgs else "train"
        for k in range(a.per_image):
            tgt = rng.choice(els)
            others = [e for e in els if e is not tgt]
            # prefer confusable distractors: same role, then similar text length
            others.sort(key=lambda e: (e["data_type"] != tgt["data_type"], abs(len(e["instruction"]) - len(tgt["instruction"])) + rng.random() * 3))
            none_item = rng.random() < a.none_frac
            need = a.k if none_item else a.k - 1
            chosen = others[:need]
            cands = [(e, False) for e in chosen] + ([] if none_item else [(tgt, True)])
            rng.shuffle(cands)
            crop = im.copy(); draw = ImageDraw.Draw(crop)
            criteria, gold = {}, "none"
            for i, (e, is_t) in enumerate(cands, 1):
                x0, y0, x1, y1 = e["bbox"][0] * W, e["bbox"][1] * H, e["bbox"][2] * W, e["bbox"][3] * H
                col = COLORS[(i - 1) % len(COLORS)]
                draw.rectangle([x0, y0, x1, y1], outline=col, width=3)
                label = str(i); tw = draw.textlength(label, font=font)
                lx, ly = max(0, x0 - tw - 6), max(0, y0 - 2)
                draw.rectangle([lx, ly, lx + tw + 6, ly + 26], fill=col); draw.text((lx + 3, ly + 1), label, fill="white", font=font)
                criteria[str(i)] = f"element {i}: {ROLE.get(e['data_type'], e['data_type'])}"
                if is_t:
                    gold = str(i)
            criteria["none"] = "none of the marked elements is the right target"
            name = f"{Path(f).stem}_{k}.png"; crop.save(out / "img" / name)
            task = rng.choice(TASKS).format(n=tgt["instruction"])
            iid = f"osatlas_{Path(f).stem}_{k}"
            items[split].append({"id": iid, "image": f"img/{name}", "task": task, "history": [], "target_repr": f"[{tgt['data_type']}] {tgt['instruction']}",
                                 "op": "CLICK", "target_tag": ROLE.get(tgt["data_type"], tgt["data_type"]), "step_index": 0, "n_steps": 1,
                                 "criteria": criteria, "gold": gold, "source": "osatlas_macos"})
            requests[split].append({"id": iid, "image": f"img/{name}", "state": f"Task: {task}\nActions already taken:\n  (none)\n",
                                    "questions": [{"qid": "ground", "qtype": "choice", "instructions": INSTR, "criteria": criteria}],
                                    "targets": {"ground": gold}})
    for split in ("train", "test"):
        with (out / f"items.{split}.jsonl").open("w") as fh:
            for it in items[split]:
                fh.write(json.dumps(it, ensure_ascii=False) + "\n")
        with (out / f"osatlas.{'train' if split == 'train' else 'validation'}.jsonl").open("w") as fh:
            for r in requests[split]:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split}: {len(items[split])} items from {len([f for f in imgs if (f in test_imgs) == (split == 'test')])} images; none gold: {sum(i['gold'] == 'none' for i in items[split])}")


if __name__ == "__main__":
    main()
