"""Decision model: causal LM backbone + tree mask + typed readouts.

Choice/Score: pointer scores  s_k = <W_q h_read, W_k h_opt_k> / sqrt(d_r)
              softmax within the question -> probabilities over options.
Noul:         sigmoid(w . h_read + b).
No text is generated; the LM head is unused.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from transformers import AutoModel

from .mask import tree_mask
from .packing import Packed


class DecisionModel(nn.Module):
    def __init__(self, backbone_name_or_config, rank: int = 256, new_vocab_size: int | None = None,
                 attn_implementation: str = "eager", torch_dtype=None):
        super().__init__()
        if isinstance(backbone_name_or_config, str):
            self.backbone = AutoModel.from_pretrained(
                backbone_name_or_config, attn_implementation=attn_implementation,
                dtype=torch_dtype or torch.float32)
        else:
            self.backbone = AutoModel.from_config(
                backbone_name_or_config, attn_implementation=attn_implementation)
        if new_vocab_size is not None:
            self.backbone.resize_token_embeddings(new_vocab_size)
        d = self.backbone.config.hidden_size
        self.rank = rank
        self.q_proj = nn.Linear(d, rank, bias=False)
        self.k_proj = nn.Linear(d, rank, bias=False)
        self.noul_head = nn.Linear(d, 1)
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)

    def hidden(self, input_ids: torch.Tensor, segment_ids: torch.Tensor,
               position_ids: torch.Tensor) -> torch.Tensor:
        mask = tree_mask(segment_ids, self.backbone.dtype)
        out = self.backbone(input_ids=input_ids, attention_mask=mask,
                            position_ids=position_ids, use_cache=False)
        return out.last_hidden_state  # [B, L, d]

    def forward(self, packed: Packed, device=None):
        """Single request. Returns per-question logits: list of tensors
        (choice/score: [K]; noul: scalar logit)."""
        dev = device or next(self.parameters()).device
        ids = torch.tensor([packed.input_ids], device=dev)
        seg = torch.tensor([packed.segment_ids], device=dev)
        pos = torch.tensor([packed.position_ids], device=dev)
        h = self.hidden(ids, seg, pos)[0]  # [L, d]
        return self.readout(h, packed)

    def readout(self, h: torch.Tensor, packed: Packed) -> list[torch.Tensor]:
        h = h.float()  # heads run in fp32 regardless of backbone dtype
        outs = []
        for qi, qtype in enumerate(packed.qtypes):
            h_read = h[packed.read_index[qi]]
            if qtype == "noul":
                outs.append(self.noul_head(h_read).squeeze(-1))
            else:
                h_opts = h[packed.option_index[qi]]  # [K, d]
                q = self.q_proj(h_read)  # [r]
                k = self.k_proj(h_opts)  # [K, r]
                outs.append((k @ q) / math.sqrt(self.rank))
        return outs

    @staticmethod
    def probabilities(logits: list[torch.Tensor], packed: Packed) -> list[dict]:
        res = []
        for qi, qtype in enumerate(packed.qtypes):
            z = logits[qi]
            if qtype == "noul":
                res.append({"type": "noul", "noul": float(torch.sigmoid(z))})
            else:
                p = torch.softmax(z.float(), -1)
                keys = packed.option_keys[qi]
                probs = {k: float(v) for k, v in zip(keys, p)}
                top = max(probs, key=lambda kk: probs[kk])
                K = len(keys)
                conf = (probs[top] - 1 / K) / (1 - 1 / K) if K > 1 else 1.0
                entry = {"type": qtype, "probabilities": probs, "confidence": round(conf, 4)}
                if qtype == "choice":
                    entry["choice"] = top
                else:
                    entry["score"] = float(sum(p_i * i for i, p_i in enumerate(p)))
                res.append(entry)
        return res
