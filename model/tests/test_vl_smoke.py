"""CPU smoke test for the vision decision model on a real (small) Qwen3-VL checkpoint.

Checks: (1) a forward pass runs with image tokens in the state prefix; (2) sibling isolation —
the logits of question A do not change when question B is added or altered; (3) branch
order invariance. Run: python -m tests.test_vl_smoke --base Qwen/Qwen3-VL-2B-Instruct
"""

from __future__ import annotations

import argparse
import time

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor

from s1.packing import SPECIAL_TOKENS, Question
from s1.vl import VLDecisionModel, VLPacker


def make_image():
    im = Image.new("RGB", (512, 384), "white")
    d = ImageDraw.Draw(im)
    for i, (x, y, col) in enumerate([(40, 40, "red"), (300, 60, "blue"), (120, 250, "green")], 1):
        d.rectangle([x, y, x + 120, y + 50], outline=col, width=4)
        d.text((x + 6, y + 6), f"{i} button {i}", fill="black")
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--max-pixels", type=int, default=256 * 256)
    a = ap.parse_args()
    torch.manual_seed(0)
    proc = AutoProcessor.from_pretrained(a.base)
    packer = VLPacker(proc, max_pixels=a.max_pixels)
    tok = proc.tokenizer
    model = VLDecisionModel(a.base, rank=64, new_vocab_size=len(tok), attn_implementation="sdpa",
                            torch_dtype=torch.float32).eval()
    img = make_image()
    state = "Task: click the blue button.\nCandidates:\n  1: element 1\n  2: element 2\n  3: element 3\n"
    crit = {"1": "element 1", "2": "element 2", "3": "element 3", "none": "none of the marked elements"}
    qa = Question("a", "choice", "Which numbered element should be acted on next?", crit)
    qb = Question("b", "noul", "Is the task already complete?")
    qc = Question("c", "choice", "Which element is red?", crit)

    t0 = time.time()
    with torch.no_grad():
        za = model(packer.pack(img, state, [qa]))[0]
        zab = model(packer.pack(img, state, [qa, qb]))
        zac = model(packer.pack(img, state, [qa, qc]))
        zca = model(packer.pack(img, state, [qc, qa]))
    print(f"forward ok, {time.time()-t0:.1f}s for 4 passes; state tokens = {packer.pack(img, state, [qa]).state_len}")
    print("p(a):", torch.softmax(za, -1).tolist())
    d1 = (za - zab[0]).abs().max().item()
    d2 = (za - zac[0]).abs().max().item()
    d3 = (zac[0] - zca[1]).abs().max().item()
    d4 = (zac[1] - zca[0]).abs().max().item()
    print(f"isolation |dz| add noul: {d1:.2e}  add choice: {d2:.2e}  order swap a: {d3:.2e}  c: {d4:.2e}")
    # two images in the state (before / after): packs, positions are valid, forward runs, isolation still holds
    img2 = make_image(); ImageDraw.Draw(img2).rectangle([40, 40, 160, 90], fill="red")
    qe = Question("e", "noul", "Did the last action change the page?")
    with torch.no_grad():
        p2 = packer.pack([img, img2], state + "Last action: click on element 1\n", [qe, qa])
        z2 = model(p2); z2b = model(packer.pack([img, img2], state + "Last action: click on element 1\n", [qe]))
    print(f"two images: n_images={p2.n_images} grid rows={p2.image_grid_thw.shape[0]} state tokens={p2.state_len} "
          f"p(effect)={torch.sigmoid(z2[0]).item():.3f} isolation |dz|={(z2[0]-z2b[0]).abs().max().item():.2e}")
    ok = max(d1, d2, d3, d4) < 1e-3
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
