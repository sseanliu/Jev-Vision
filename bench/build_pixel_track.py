"""Pixel track: the same items as the candidate track, rendered as an unmarked screenshot with one red marker
at the action point and no candidate table.

Needs a recording made with raw screenshots and element boxes (record_triplets.py after 2026-09-20).
Reads bench/<name>_items.jsonl (candidate track, built by build_release.py) and the recorder's steps.jsonl,
writes bench/<name>_items_pixel.jsonl and release/<name>/pixel/*.png.

  skip   : raw before screenshot, marker on the candidate element; state says "the element under the marker"
  effect : raw before screenshot with marker on the executed element + raw after screenshot
  done   : raw screenshot of the judged state, no marker
  ground : not in the pixel track (needs a coordinate output; v1 model work)

python bench/build_pixel_track.py --name v1 --steps model/data/vision/triplets/run2_val/steps.jsonl
"""

import argparse
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def marker(src, dst, box):
    im = Image.open(src).convert("RGB"); d = ImageDraw.Draw(im)
    cx, cy = box["x"] + box["w"] / 2, box["y"] + box["h"] / 2
    r = 11
    d.ellipse([cx - r - 3, cy - r - 3, cx + r + 3, cy + r + 3], fill="white")
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(230, 30, 30))
    im.save(dst, format="PNG")


def rewrite_state(state, question):
    s = re.sub(r"Candidate \d+: element \d+: [^\n]*", "Candidate: the element under the red marker", state)
    s = re.sub(r"Last action: (\w+) on element \d+: [^\n]*", r"Last action: \1 at the red marker", s)
    if question == "effect" and "Last action:" not in s:
        s = s.rstrip("\n") + "\nLast action: at the red marker\n"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--steps", required=True)
    a = ap.parse_args()
    steps = {}
    for l in open(a.steps):
        s = json.loads(l)
        if "id" in s:
            steps[s["id"]] = s
    rec_dir = Path(a.steps).parent
    items = [json.loads(l) for l in open(ROOT / "bench" / f"{a.name}_items.jsonl")]
    out_dir = ROOT / "release" / a.name / "pixel"; out_dir.mkdir(parents=True, exist_ok=True)
    out, skipped = [], 0
    for it in items:
        q = it["question"]
        if q == "ground":
            continue
        st = steps.get(it["step_id"])
        if not st or "raw_before_img" not in st:
            skipped += 1; continue
        raw_b, raw_a = rec_dir / st["raw_before_img"], rec_dir / st["raw_after_img"]
        boxes = st["boxes_before"]
        pit = dict(it); pit["id"] = "pixel:" + it["id"]; pit["track"] = "pixel"
        pit.pop("elements_before", None); pit.pop("elements_after", None); pit.pop("criteria", None)
        pit["state"] = rewrite_state(it["state"], q)
        if q == "skip":
            idx = int(it["candidate"]) - 1
            if idx >= len(boxes):
                skipped += 1; continue
            dst = out_dir / f"{it['step_id']}_before_c{idx + 1}.png"
            if not dst.exists():
                marker(raw_b, dst, boxes[idx])
            pit["images"] = [f"pixel/{dst.name}"]; pit["marker"] = boxes[idx]
        elif q == "effect":
            idx = int(st["chosen"]) - 1
            if idx >= len(boxes):
                skipped += 1; continue
            dst = out_dir / f"{it['step_id']}_before_act.png"
            if not dst.exists():
                marker(raw_b, dst, boxes[idx])
            dst_a = out_dir / f"{it['step_id']}_after.png"
            if not dst_a.exists():
                Image.open(raw_a).convert("RGB").save(dst_a, format="PNG")
            pit["images"] = [f"pixel/{dst.name}", f"pixel/{dst_a.name}"]; pit["marker"] = boxes[idx]
        else:  # done
            which = it.get("which", "before"); src = raw_a if which == "after" else raw_b
            dst = out_dir / f"{it['step_id']}_{which}.png"
            if not dst.exists():
                Image.open(src).convert("RGB").save(dst, format="PNG")
            pit["images"] = [f"pixel/{dst.name}"]
        out.append(pit)
    with open(ROOT / "bench" / f"{a.name}_items_pixel.jsonl", "w") as f:
        for it in out:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"{len(out)} pixel items ({dict(Counter(i['question'] for i in out))}), {skipped} skipped, images in {out_dir}")


if __name__ == "__main__":
    main()
