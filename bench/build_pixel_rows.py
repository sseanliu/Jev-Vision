"""Training rows for the pixel track: the same state questions as build_state_rows_record.py, rendered on the raw
screenshot with one red marker (skip: the candidate; effect: the executed element; done: no marker) and with the
candidate table removed from the state text. Needs a recording with raw/ screenshots and boxes (record_triplets.py
after 2026-09-20). Output rows are in the train_vl.py format with absolute image paths.

Typical use: mix these rows with the candidate-track rows so one model serves both tracks.

python bench/build_pixel_rows.py --steps model/data/vision/triplets/run2_train/steps.jsonl \
    --rows model/data/vision/state_rows/rec2train.train.jsonl --out model/data/vision/state_rows/rec2train_pixel.train.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_pixel_track import marker, rewrite_state  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", required=True, help="recorder steps.jsonl with raw_before_img / boxes_before")
    ap.add_argument("--rows", required=True, help="candidate-track rows built from the same steps (build_state_rows_record.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-dir", default=None, help="where marked images go (default: <recording>/pixel)")
    a = ap.parse_args()
    rec_dir = Path(a.steps).parent
    img_dir = Path(a.img_dir) if a.img_dir else rec_dir / "pixel"; img_dir.mkdir(exist_ok=True)
    steps = {}
    for l in open(a.steps):
        s = json.loads(l)
        if "id" in s:
            steps[s["id"]] = s
    n = Counter(); skipped = 0
    with open(a.out, "w") as f:
        for l in open(a.rows):
            r = json.loads(l); q = r["questions"][0]; qid = q["qid"]; m = r["meta"]
            if qid == "ground":
                continue
            st = steps.get(m["step_id"])
            if not st or "raw_before_img" not in st:
                skipped += 1; continue
            raw_b, raw_a = rec_dir / st["raw_before_img"], rec_dir / st["raw_after_img"]; boxes = st["boxes_before"]
            if qid == "skip":
                idx = int(m["candidate"]) - 1
                if idx >= len(boxes):
                    skipped += 1; continue
                dst = img_dir / f"{m['step_id']}_before_c{idx + 1}.png"
                if not dst.exists():
                    marker(raw_b, dst, boxes[idx])
                images = [str(dst.resolve())]
            elif qid == "effect":
                idx = int(st["chosen"]) - 1
                if idx >= len(boxes):
                    skipped += 1; continue
                dst = img_dir / f"{m['step_id']}_before_act.png"
                if not dst.exists():
                    marker(raw_b, dst, boxes[idx])
                dst_a = img_dir / f"{m['step_id']}_after.png"
                if not dst_a.exists():
                    Image.open(raw_a).convert("RGB").save(dst_a, format="PNG")
                images = [str(dst.resolve()), str(dst_a.resolve())]
            else:
                which = m.get("which", "before"); src = raw_a if which == "after" else raw_b
                dst = img_dir / f"{m['step_id']}_{which}.png"
                if not dst.exists():
                    Image.open(src).convert("RGB").save(dst, format="PNG")
                images = [str(dst.resolve())]
            row = {"images": images, "state": rewrite_state(r["state"], qid), "questions": [q], "targets": r["targets"],
                   "meta": {**{k: v for k, v in m.items() if k not in ("text_before", "text_after")}, "track": "pixel"}}
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); n[qid] += 1
    print(f"{sum(n.values())} pixel rows {dict(n)}, {skipped} skipped -> {a.out}")


if __name__ == "__main__":
    main()
