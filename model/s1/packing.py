"""Pack one request (state + N typed questions) into a single token sequence
with segment ids, branch-relative position ids, and readout indices.

Layout (one sequence):

    [state] [Q1: <|q|> instr <|opt|> k1: d1 <|/opt|> <|opt|> k2: d2 <|/opt|> ... <|read|>] [Q2 ...]

- segment id 0 = state; q >= 1 = question q. The attention mask (mask.py)
  lets a token see the state and its own segment only.
- position ids restart at len(state) for every question, so branches are
  position-identical and question order cannot matter.
- Choice/Score read: pointer between the hidden state at <|read|> and the
  hidden state at each option's <|/opt|>. Noul read: hidden state at <|read|>.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SPECIAL_TOKENS = ["<|q|>", "<|opt|>", "<|/opt|>", "<|read|>"]


@dataclass
class Question:
    qid: str
    qtype: str  # "choice" | "noul" | "score"
    instructions: str
    criteria: dict[str, str] | list[str] | None = None  # choice: {key: desc}; score: [level descs]


@dataclass
class Packed:
    input_ids: list[int]
    segment_ids: list[int]
    position_ids: list[int]
    read_index: list[int]  # per question: index of <|read|>
    option_index: list[list[int]]  # per question: indices of <|/opt|> (empty for noul)
    option_keys: list[list[str]]
    qtypes: list[str] = field(default_factory=list)


class Packer:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
        self.added = added
        self.q_id, self.opt_id, self.optend_id, self.read_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    def _enc(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def pack(self, state: str, questions: list[Question]) -> Packed:
        return self.pack_ids(self._enc(state), questions)

    def pack_ids(self, state_ids: list[int], questions: list[Question]) -> Packed:
        """Same layout, but the state arrives pre-tokenised (e.g. from a multimodal processor
        with image placeholder tokens already expanded)."""
        ids = list(state_ids)
        seg = [0] * len(ids)
        pos = list(range(len(ids)))
        S = len(ids)
        read_index, option_index, option_keys, qtypes = [], [], [], []
        for q_no, q in enumerate(questions, start=1):
            branch: list[int] = [self.q_id] + self._enc(q.instructions)
            opt_idx: list[int] = []
            keys: list[str] = []
            if q.qtype == "noul":
                branch += self._enc(" Answer yes or no.")
            elif q.qtype == "choice":
                assert isinstance(q.criteria, dict) and len(q.criteria) >= 1
                for k, d in q.criteria.items():
                    branch += [self.opt_id] + self._enc(f" {k}: {d}") + [self.optend_id]
                    opt_idx.append(len(ids) + len(branch) - 1)
                    keys.append(k)
            elif q.qtype == "score":
                assert isinstance(q.criteria, list) and len(q.criteria) >= 2
                for i, d in enumerate(q.criteria):
                    branch += [self.opt_id] + self._enc(f" level {i}: {d}") + [self.optend_id]
                    opt_idx.append(len(ids) + len(branch) - 1)
                    keys.append(str(i))
            else:
                raise ValueError(q.qtype)
            branch.append(self.read_id)
            read_index.append(len(ids) + len(branch) - 1)
            ids += branch
            seg += [q_no] * len(branch)
            pos += list(range(S, S + len(branch)))
            option_index.append(opt_idx)
            option_keys.append(keys)
            qtypes.append(q.qtype)
        return Packed(ids, seg, pos, read_index, option_index, option_keys, qtypes)
