"""Tree attention mask: causal, and a token may attend to position j only if
j is in the state (segment 0) or in the same segment as the token."""

from __future__ import annotations

import torch


def tree_mask(segment_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """segment_ids: [B, L] long. Returns additive mask [B, 1, L, L] (0 = attend,
    dtype.min = blocked), the form transformers passes through unchanged."""
    B, L = segment_ids.shape
    i = torch.arange(L, device=segment_ids.device)
    causal = i[None, :] <= i[:, None]  # [L, L]  query i, key j
    same = segment_ids[:, :, None] == segment_ids[:, None, :]  # [B, L, L]
    key_is_state = (segment_ids == 0)[:, None, :]  # [B, 1, L]
    allowed = causal[None] & (same | key_is_state)
    mask = torch.zeros(B, 1, L, L, dtype=dtype, device=segment_ids.device)
    mask.masked_fill_(~allowed[:, None], torch.finfo(dtype).min)
    return mask


def tree_mask_bool(segment_ids: torch.Tensor) -> torch.Tensor:
    """Boolean variant [B, 1, L, L], True = attend (for sdpa/flex backends)."""
    B, L = segment_ids.shape
    i = torch.arange(L, device=segment_ids.device)
    causal = i[None, :] <= i[:, None]
    same = segment_ids[:, :, None] == segment_ids[:, None, :]
    key_is_state = (segment_ids == 0)[:, None, :]
    return (causal[None] & (same | key_is_state))[:, None]
