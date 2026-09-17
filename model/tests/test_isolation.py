"""Behavioural fingerprints the architecture must satisfy by construction,
checked on a tiny random-init Qwen3 backbone on CPU.

1. Sibling isolation: a question's logits do not change when other questions
   are added, removed, or reordered.
2. Sequential equivalence: packed forward == separate forward of state + one
   question.
3. Option order may change the distribution (listwise), but the option set is
   visible to the readout: the last option can influence earlier scores.
4. Gradients from the readout reach the backbone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3Config

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s1.model import DecisionModel  # noqa: E402
from s1.packing import Packer, Question  # noqa: E402

torch.manual_seed(0)
TOK = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")  # same family vocab, cached quickly
PACKER = Packer(TOK)
CFG = Qwen3Config(vocab_size=len(TOK), hidden_size=64, intermediate_size=128,
                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                  head_dim=16, max_position_embeddings=4096)
MODEL = DecisionModel(CFG, rank=32).eval()

STATE = "Payouts to my bank have been failing since Tuesday. My bank says nothing is blocked."
Q1 = Question("who", "choice", "Who most likely caused the payout failures?",
              {"bank": "The bank caused it", "provider": "The payment provider caused it",
               "customer": "The customer caused it", "unknown": "Cannot tell"})
Q2 = Question("urgent", "noul", "Does this message require urgent attention?")
Q3 = Question("anger", "score", "How angry is the customer?", ["calm", "annoyed", "furious"])


def logits_for(questions):
    with torch.no_grad():
        return [z.detach().clone() for z in MODEL(PACKER.pack(STATE, questions))]


def test_sibling_isolation():
    alone = logits_for([Q1])[0]
    with_siblings = logits_for([Q2, Q1, Q3])[1]
    reordered = logits_for([Q3, Q1, Q2])[1]
    assert torch.allclose(alone, with_siblings, atol=1e-5), (alone, with_siblings)
    assert torch.allclose(alone, reordered, atol=1e-5)
    n_alone = logits_for([Q2])[0]
    n_with = logits_for([Q1, Q3, Q2])[2]
    assert torch.allclose(n_alone, n_with, atol=1e-5)


def test_state_is_visible():
    a = logits_for([Q1])[0]
    global STATE
    old = STATE
    STATE = "The weather is nice today and the park is full of people."
    b = logits_for([Q1])[0]
    STATE = old
    assert not torch.allclose(a, b, atol=1e-4), "state change must change the answer"


def test_last_option_can_influence_earlier_scores():
    base = logits_for([Q1])[0]
    extended = Question("who", "choice", Q1.instructions,
                        {**Q1.criteria, "weather": "Bad weather caused it"})
    ext = logits_for([extended])[0]
    # The first four raw scores need not be identical because the read vector
    # saw the fifth option; that is the listwise property Jev exhibits.
    assert not torch.allclose(base, ext[:4], atol=1e-6)
    assert ext.shape[0] == 5


def test_gradients_reach_backbone():
    MODEL.train()
    packed = PACKER.pack(STATE, [Q1, Q2])
    z = MODEL(packed)
    loss = torch.nn.functional.cross_entropy(z[0][None], torch.tensor([2])) + \
        torch.nn.functional.binary_cross_entropy_with_logits(z[1], torch.tensor(1.0))
    loss.backward()
    grads = [p.grad for n, p in MODEL.backbone.named_parameters() if "layers.0" in n and p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
    MODEL.zero_grad()
    MODEL.eval()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
