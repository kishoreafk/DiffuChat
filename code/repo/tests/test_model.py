"""Model tests (spec §9): shapes + BIDIRECTIONALITY.

The inverse of the AR causality test: perturbing a FUTURE token must change
present-position logits (a causal model would be blind to it).
"""

from __future__ import annotations

import torch

from diffuchat.model import DenoiseLM


def test_shapes(tiny_model, tiny_cfg):
    x = torch.randint(0, tiny_cfg.vocab_size, (2, tiny_cfg.block_size))
    out = tiny_model(x)
    assert out.shape == (2, tiny_cfg.block_size, tiny_cfg.vocab_size)


def test_rejects_overlong_sequence(tiny_model, tiny_cfg):
    x = torch.randint(0, tiny_cfg.vocab_size, (1, tiny_cfg.block_size + 1))
    try:
        tiny_model(x)
        raised = False
    except ValueError:
        raised = True
    assert raised, "model must refuse sequences longer than block_size"


def test_bidirectional_future_perturbation_changes_present_logits(tiny_model, tiny_cfg):
    torch.manual_seed(0)
    x = torch.randint(0, tiny_cfg.vocab_size - 1, (1, tiny_cfg.block_size))
    x[0, -1] = tiny_cfg.mask_token_id
    with torch.no_grad():
        base = tiny_model(x.clone())[0, 0]  # logits at position 0
    # perturb a FUTURE token (arbitrary swap of two future ids)
    x2 = x.clone()
    x2[0, 30], x2[0, 31] = x2[0, 31], x2[0, 30]
    with torch.no_grad():
        pert = tiny_model(x2)[0, 0]
    assert not torch.allclose(base, pert), (
        "position-0 logits must react to future-token changes — "
        "model is not bidirectional (causal mask leaked in?)"
    )


def test_bidirectional_attention_matrix_has_no_causal_mask(tiny_model):
    """A position must attend to positions AFTER it: swap the last token and
    check position-0 logits react (end-to-end residual-stream probe)."""
    m = tiny_model
    x = torch.zeros(1, m.cfg.block_size, dtype=torch.long)
    x[0, -1] = 1  # distinct token at the END
    with torch.no_grad():
        out_a = m(x.clone())
        x[0, -1] = 2
        out_b = m(x)
    assert not torch.allclose(out_a[0, 0], out_b[0, 0]), (
        "position 0 ignores the last token — attention cannot be bidirectional"
    )


def test_weight_tying_and_param_accounting(tiny_cfg):
    m = DenoiseLM(tiny_cfg)
    # head shares storage with the token embedding (nanoGPT-style tying)
    assert m.head.weight is m.tok_emb.weight
    total = sum(p.numel() for p in m.parameters())
    assert m.num_params(non_embedding=False) == total
    # non-embedding count subtracts exactly the (tied) embedding tables
    d, V, T = tiny_cfg.d_model, tiny_cfg.vocab_size, tiny_cfg.block_size
    assert m.num_params() == total - (V * d + T * d)
