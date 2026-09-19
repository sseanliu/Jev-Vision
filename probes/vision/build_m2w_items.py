"""Build `ground` choice items from Multimodal-Mind2Web (teacher test set and student training set).

Each Mind2Web step has a full-page screenshot, a natural task, the previous actions, one or two
positive candidate elements and ~1000 negative candidates with bounding boxes. We crop a viewport
around the target, keep the positive plus K hard negatives inside the crop (clickable, similar
size, nearest first), draw numbered set-of-mark boxes, and emit one choice question:
"which numbered element should be acted on next" with an extra `none` option. With --none-frac,
that share of items drops the positive box so `none` is the gold answer.

Outputs: items.jsonl (audit format) and requests.jsonl (s1 VL request format: image/state/questions/targets).

python build_m2w_items.py --split test_domain --n 300 --out ../../model/data/vision/m2w_items
python build_m2w_items.py --split train --n 8000 --none-frac 0.1 --out /workspace/m2w_train --files 27
"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
import subprocess
from pathlib import Path

import duckdb
from PIL import Image, ImageDraw, ImageFont

VIEW_H = 1000
COLORS = ["#e6194b", "#3cb44b", "#0082c8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#d2f53c",
          "#fabebe", "#008080", "#aa6e28", "#800000", "#808000", "#000080", "#808080", "#000000"]
INSTR = "Which numbered element should be acted on next to make progress on the task?"


def parquet_urls(split: str) -> list[str]:
    out = subprocess.run(["hf", "datasets", "parquet", "osunlp/Multimodal-Mind2Web", "--format", "json"],
                         capture_output=True, text=True, check=True).stdout
    return [f["url"] for f in json.loads(out) if f["split"] == split]


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


def state_text(task: str, history: list[str]) -> str:
    hist = "\n".join(f"  - {h}" for h in history) or "  (none)"
    return f"Task: {task}\nActions already taken:\n{hist}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test_domain")
    ap.add_argument("--files", type=int, default=2, help="how many parquet files of the split to read")
    ap.add_argument("--out", default="../../model/data/vision/m2w_items")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--k", type=int, default=9, help="candidates per item (incl. the positive)")
    ap.add_argument("--none-frac", type=float, default=0.0)
    ap.add_argument("--jitter", action="store_true",
                    help="random viewport offset (target anywhere in the crop) and distractors sampled from the "
                         "nearest 3(K-1) pool instead of strictly nearest: removes the centre-of-cluster cue")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    urls = parquet_urls(a.split)
    rng.shuffle(urls)
    urls = urls[: a.files]
    items, requests = [], []
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    except OSError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        except OSError:
            font = ImageFont.load_default()
    for url in urls:
        if len(items) >= a.n:
            break
        rows = con.execute(
            f"SELECT annotation_id, action_uid, confirmed_task, action_reprs, target_action_index, "
            f"target_action_reprs, operation, pos_candidates, neg_candidates, screenshot FROM '{url}'").fetchall()
        # later-step targets per annotation: a candidate that is the positive of a *later* step is an
        # alternative valid answer now (Mind2Web records one order of several valid ones)
        by_ann = {}
        for r in rows:
            ann_, tidx_, pos_ = r[0], r[4], r[7]
            if tidx_ and str(tidx_).isdigit() and pos_:
                ids_ = {json.loads(p).get("backend_node_id") for p in pos_}
                by_ann.setdefault(ann_, []).append((int(tidx_), ids_))
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
            if a.jitter:
                margin = 60
                lo, hi = cy - VIEW_H + margin, cy - margin
                top = int(max(0, min(H - VIEW_H, rng.uniform(lo, hi))))
            else:
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
            none_item = rng.random() < a.none_frac
            need = a.k if none_item else a.k - 1
            if len(negs) < need:
                continue
            negs.sort(key=lambda t: t[0])
            chosen = rng.sample(negs[: 3 * need], need) if a.jitter else negs[:need]
            cands = [(b, cd, at, False) for _, b, cd, at in chosen]
            if not none_item:
                cands.append((pb, json.loads(pos[0]), pa, True))
            rng.shuffle(cands)
            crop = im.crop(view)
            draw = ImageDraw.Draw(crop)
            criteria, gold = {}, "none"
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
            cur = int(tidx) if tidx and str(tidx).isdigit() else -1
            later_ids = set().union(*[ids_ for t_, ids_ in by_ann.get(ann, []) if t_ > cur]) if cur >= 0 else set()
            alt_gold = [str(i) for i, (b, cd, at, is_pos) in enumerate(cands, 1)
                        if not is_pos and cd.get("backend_node_id") in later_ids]
            name = f"{ann}_{uid}.png"
            crop.save(out / "img" / name)
            history = list(reprs)[: int(tidx)] if tidx and str(tidx).isdigit() else []
            iid = f"{ann}_{uid}"
            opj = json.loads(op)
            m = re.match(r"\[(\w+)\]", tgt or "")
            items.append({"id": iid, "image": f"img/{name}", "task": task, "history": history,
                          "target_repr": tgt, "op": opj.get("op"), "value": opj.get("value", ""),
                          "target_tag": m.group(1) if m else None,
                          "step_index": int(tidx) if tidx and str(tidx).isdigit() else None, "n_steps": len(list(reprs)),
                          "criteria": criteria, "gold": gold, "alt_gold": alt_gold, "view_top": top, "page_size": [W, H]})
            requests.append({"id": iid, "image": f"img/{name}", "state": state_text(task, history),
                             "questions": [{"qid": "ground", "qtype": "choice", "instructions": INSTR, "criteria": criteria}],
                             "targets": {"ground": gold}})
    with (out / "items.jsonl").open("w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with (out / "requests.jsonl").open("w") as f:
        for r in requests:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} items to {out} (none gold: {sum(i['gold'] == 'none' for i in items)})")


if __name__ == "__main__":
    main()
