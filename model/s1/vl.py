"""Vision variant of the decision model on a Qwen3-VL backbone.

The state prefix is [<|vision_start|> image tokens <|vision_end|> text]; question branches
are text-only and identical to the text model (tree mask, branch-relative positions, pointer
readout). Only two things change:
  - packing: the state is tokenised by the Qwen3-VL processor (image placeholders expanded),
    and pixel_values / image_grid_thw / mm_token_type_ids ride along;
  - positions: Qwen3-VL uses M-RoPE (3 position streams). State positions come from the
    backbone's get_rope_index; every branch restarts at max(state position) + 1 on all three
    streams, so branches stay position-identical and order-invariant as before.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from PIL import Image

from .mask import tree_mask
from .model import DecisionModel
from .packing import Packed, Packer, Question

IMAGE_PREFIX = "<|vision_start|><|image_pad|><|vision_end|>"


@dataclass
class PackedVL(Packed):
    pixel_values: torch.Tensor | None = None  # [n_patches, patch_dim]
    image_grid_thw: torch.Tensor | None = None  # [n_images, 3]
    mm_token_type_ids: list[int] = field(default_factory=list)  # state part only
    state_len: int = 0


class VLPacker:
    def __init__(self, processor, max_pixels: int | None = None):
        self.proc = processor
        self.text = Packer(processor.tokenizer)  # adds <|q|> <|opt|> <|/opt|> <|read|>
        self.max_pixels = max_pixels

    def pack(self, image: Image.Image | str, state_text: str, questions: list[Question],
             max_pixels: int | None = None) -> PackedVL:
        img = Image.open(image).convert("RGB") if isinstance(image, str) else image
        # transformers 5.x Qwen3-VL processors ignore a bare `max_pixels`; the budget goes through `size`
        budget = max_pixels or self.max_pixels
        kw = {"size": {"shortest_edge": 3136, "longest_edge": budget}} if budget else {}
        enc = self.proc(text=[IMAGE_PREFIX + state_text], images=[img], return_tensors="pt",
                        add_special_tokens=False, images_kwargs=kw)
        state_ids = enc["input_ids"][0].tolist()
        p = self.text.pack_ids(state_ids, questions)
        mm = enc["mm_token_type_ids"][0].tolist() if "mm_token_type_ids" in enc else \
            [1 if t == self.proc.image_token_id else 0 for t in state_ids]
        return PackedVL(p.input_ids, p.segment_ids, p.position_ids, p.read_index, p.option_index,
                        p.option_keys, p.qtypes, pixel_values=enc["pixel_values"],
                        image_grid_thw=enc["image_grid_thw"], mm_token_type_ids=mm, state_len=len(state_ids))


class VLDecisionModel(DecisionModel):
    """DecisionModel whose backbone is a Qwen3VLModel (vision tower + LM)."""

    def _mrope_positions(self, ids: torch.Tensor, pos1d: torch.Tensor, seg: torch.Tensor,
                         state_lens: list[int], mm_tt: list[list[int]], grid: torch.Tensor) -> torch.Tensor:
        """[3, B, L] positions: M-RoPE over the state, branch tokens continue after it."""
        bb = self.backbone.get_base_model() if hasattr(self.backbone, "get_base_model") else self.backbone
        B, L = ids.shape
        out = torch.zeros((3, B, L), dtype=torch.long, device=ids.device)
        g = 0
        for b in range(B):
            S = state_lens[b]
            tt = torch.tensor([mm_tt[b]], device=ids.device)
            n_img = int((tt == 1).any())  # one image per request in v1
            p3, _ = bb.get_rope_index(ids[b:b + 1, :S], tt, image_grid_thw=grid[g:g + n_img] if n_img else None)
            g += n_img
            out[:, b, :S] = p3[:, 0, :]
            smax = int(p3.max())
            branch = seg[b] > 0
            out[:, b, branch] = (smax + 1 + (pos1d[b, branch] - S)).unsqueeze(0).expand(3, -1)
        return out

    def hidden_vl(self, ids, seg, pos1d, pixel_values, grid, state_lens, mm_tt):
        mask = tree_mask(seg, self.backbone.dtype)
        pos3 = self._mrope_positions(ids, pos1d, seg, state_lens, mm_tt, grid)
        out = self.backbone(input_ids=ids, pixel_values=pixel_values, image_grid_thw=grid,
                            attention_mask=mask, position_ids=pos3, use_cache=False)
        return out.last_hidden_state

    def forward(self, packed: PackedVL, device=None):
        dev = device or next(self.parameters()).device
        ids = torch.tensor([packed.input_ids], device=dev)
        seg = torch.tensor([packed.segment_ids], device=dev)
        pos = torch.tensor([packed.position_ids], device=dev)
        h = self.hidden_vl(ids, seg, pos, packed.pixel_values.to(dev, self.backbone.dtype),
                           packed.image_grid_thw.to(dev), [packed.state_len], [packed.mm_token_type_ids])[0]
        return self.readout(h, packed)


def collate_vl(examples, pad_id: int):
    """examples: objects with .packed (PackedVL) and .targets."""
    L = max(len(e.packed.input_ids) for e in examples)
    B = len(examples)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    seg = torch.full((B, L), -1, dtype=torch.long)
    pos = torch.zeros((B, L), dtype=torch.long)
    for b, e in enumerate(examples):
        n = len(e.packed.input_ids)
        ids[b, :n] = torch.tensor(e.packed.input_ids)
        seg[b, :n] = torch.tensor(e.packed.segment_ids)
        pos[b, :n] = torch.tensor(e.packed.position_ids)
    pix = torch.cat([e.packed.pixel_values for e in examples], 0)
    grid = torch.cat([e.packed.image_grid_thw for e in examples], 0)
    return ids, seg, pos, pix, grid, [e.packed.state_len for e in examples], [e.packed.mm_token_type_ids for e in examples], examples
