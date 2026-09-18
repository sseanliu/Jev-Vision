"""Build `ground` choice items from Multimodal-Mind2Web for the teacher calibration test (V0).

Each Mind2Web step has a full-page screenshot, a natural task, the previous actions, one or two
positive candidate elements and ~1000 negative candidates with bounding boxes. We crop a viewport
around the target, keep the positive plus K hard negatives inside the crop (clickable, similar
size, nearest first), draw numbered set-of-mark boxes, and emit one choice question:
"which numbered element should be acted on next" with an extra `none` option.

python build_m2w_items.py --out ../../model/data/vision/m2w_items --n 300 --k 9
"""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

import duckdb
from PIL import Image, ImageDraw, ImageFont

PARQUETS = [
    "https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web/resolve/refs%2Fconvert%2Fparquet/default/test_domain/0002.parquet",
    "https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web/resolve/refs%2Fconvert%2Fparquet/default/test_domain/0001.parquet",
]
VIEW_H = 1000
COLORS = ["#e6194b", "#3cb44b", "#0082c8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#d2f53c",
          "#fabebe", "#008080", "#aa6e28", "#800000", "#808000", "#000080", "#808080", "#000000"]


def parse_box(attrs: str):
    a = json.loads(attrs)
    if "bounding_box_rect" not in a:
        return None, a
    x, y, w, h = (float(v) for v in a["bounding_box_rect"].split(","))
    return (x, y, w, h), a


def describe(cand: dict, attrs: dict) -> str:
    bits = [cand.get("tag", "")]
    for k in ("name", "aria_label", "title", "placeholder", "alt", "type", "value"):
        if attrs.get(k):
            bits.append(f'{k}="{str(attrs[k])[:40]}"')
    return " ".join(b for b in bits if b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../../model/data/vision/m2w_items")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--k", type=int, default=9, help="candidates per item (incl. the positive)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    items = []
    for url in PARQUETS:
        if len(items) >= a.n:
            break
        rows = con.execute(
            f"SELECT annotation_id, action_uid, confirmed_task, action_reprs, target_action_index, "
            f"target_action_reprs, operation, pos_candidates, neg_candidates, screenshot FROM '{url}'").fetchall()
        rng.shuffle(rows)
        for ann, uid, task, reprs, tidx, tgt, op, pos, neg, shot in rows:
            if len(items) >= a.n:
                break
            if not pos or not shot or not shot.get("bytes"):
                continue
            pb, pa = parse_box(json.loads(pos[0])["attributes"])
            if pb is None or pb[2] <= 0 or pb[3] <= 0 or pb[2] > 1000 or pb[3] > 600:
                continue
            im = Image.open(io.BytesIO(shot["bytes"])).convert("RGB")
            W, H = im.size
            cy = pb[1] + pb[3] / 2
            top = int(max(0, min(H - VIEW_H, cy - VIEW_H / 2)))
            view = (0, top, W, min(H, top + VIEW_H))

            def inside(b):
                return b[1] >= view[1] and b[1] + b[3] <= view[3] and b[2] * b[3] < 0.25 * W * VIEW_H and b[2] > 8 and b[3] > 8

            negs = []
            for c in neg:
                cd = json.loads(c)
                b, at = parse_box(cd["attributes"])
                if b is None or not inside(b):
                    continue
                if at.get("is_clickable") != "true" and cd.get("tag") not in ("a", "button", "input", "select", "textarea"):
                    continue
                d = abs((b[1] + b[3] / 2) - cy) + abs((b[0] + b[2] / 2) - (pb[0] + pb[2] / 2)) * 0.5
                negs.append((d, b, cd, at))
            if len(negs) < a.k - 1:
                continue
            negs.sort(key=lambda t: t[0])
            chosen = negs[: a.k - 1]
            cands = [(pb, json.loads(pos[0]), pa, True)] + [(b, cd, at, False) for _, b, cd, at in chosen]
            rng.shuffle(cands)
            crop = im.crop(view)
            draw = ImageDraw.Draw(crop)
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
            except OSError:
                font = ImageFont.load_default()
            criteria, gold = {}, None
            for i, (b, cd, at, is_pos) in enumerate(cands, 1):
                x0, y0 = b[0], b[1] - top
                x1, y1 = x0 + b[2], y0 + b[3]
                col = COLORS[(i - 1) % len(COLORS)]
                draw.rectangle([x0, y0, x1, y1], outline=col, width=3)
                label = str(i)
                tw = draw.textlength(label, font=font)
                lx, ly = max(0, x0 - tw - 6), max(0, y0 - 2)
                draw.rectangle([lx, ly, lx + tw + 6, ly + 26], fill=col)
                draw.text((lx + 3, ly + 1), label, fill="white", font=font)
                criteria[str(i)] = f"element {i}: {describe(cd, at)}"
                if is_pos:
                    gold = str(i)
            criteria["none"] = "none of the marked elements is the right target"
            name = f"{ann}_{uid}.png"
            crop.save(out / "img" / name)
            history = list(reprs)[: int(tidx)] if tidx and str(tidx).isdigit() else []
            items.append({"id": f"{ann}_{uid}", "image": f"img/{name}", "task": task, "history": history,
                          "target_repr": tgt, "op": json.loads(op).get("op"), "criteria": criteria, "gold": gold,
                          "view_top": top, "page_size": [W, H]})
    with (out / "items.jsonl").open("w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} items to {out}")


if __name__ == "__main__":
    main()
