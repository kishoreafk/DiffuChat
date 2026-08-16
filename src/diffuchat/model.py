"""DenoiseLM: the bidirectional denoiser backbone (spec §5.3).

Same nanoGPT-style block stack as a char-GPT EXCEPT:

- attention is FULLY BIDIRECTIONAL — there is no ``tril`` causal mask,
  because the model must see the whole canvas including *future* ``[MASK]``
  positions;
- the head predicts the ORIGINAL token id at every position (the loss decides
  which positions count — see ``train.masked_ce_loss``).

Deliberate design note: no ``[CLS]``, no segment embeddings — prompt and
answer share one stream; conditioning is by position + content only, exactly
like LLaDA's shared-canvas formulation at miniature scale.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DenoiseLMConfig

# Relative-position-bias initialization: strong neighbour-locality prior
# (gain 3.0, exponential decay with scale 2).  Without it, a char-scale
# fully-bidirectional masked LM never escapes the unigram plateau: [MASK]
# queries carry no content, so attention collapses to the (near-constant)
# context mean and per-window gradients cancel (verified experimentally —
# see README "Training findings").  Seeding attention on |i-j|<=2 neighbours
# gives the head immediate bigram/trigram signal; the bias stays LEARNABLE
# (T5-style) and diverges per head during training.
_BIAS_GAIN = 3.0
_BIAS_DECAY = 2.0


class Block(nn.Module):
    """Pre-LN transformer block with fully bidirectional attention.

    No causal mask.  Optionally adds a learned T5-style relative-position
    bias to the attention scores (``cfg.rel_pos_bias``, default on).
    """

    def __init__(self, cfg: DenoiseLMConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0, "d_model must divide n_head"
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        self.attn_dropout = cfg.dropout
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )
        self.rel_pos_bias = cfg.rel_pos_bias
        if self.rel_pos_bias:
            T2 = 2 * cfg.block_size - 1
            d = np.arange(-(T2 // 2), T2 // 2 + 1)
            init = _BIAS_GAIN * np.exp(-np.abs(d) / _BIAS_DECAY)
            self.rel_bias = nn.Parameter(
                torch.tensor(init, dtype=torch.float32).repeat(cfg.n_head, 1)
            )
            idx = torch.arange(cfg.block_size)
            self.register_buffer(
                "rel_idx", (idx[None, :] - idx[:, None]) + T2 // 2, persistent=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)  # (B,nh,T,hs)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        # NOTE: no causal mask — bidirectional by construction (spec §5.3).
        attn_mask = None
        if self.rel_pos_bias:
            attn_mask = self.rel_bias[:, self.rel_idx[:T, :T]].unsqueeze(0)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.resid_drop(self.proj(y))
        x = x + self.mlp(self.ln2(x))
        return x


class DenoiseLM(nn.Module):
    """Masked-diffusion LM: forward(x) -> logits (B, T, V).

    The loss lives in ``train.py`` (MDLM objective, masked positions only).
    """

    def __init__(self, cfg: DenoiseLMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # weight tying (nanoGPT-style): the head reuses the token embedding
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx (B, T) int64 -> logits (B, T, vocab_size)."""
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence {T} exceeds block_size {self.cfg.block_size}")
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel() + self.pos_emb.weight.numel()
        return n


def positionally_encode_stub() -> None:  # pragma: no cover - placeholder doc
    """Sinusoidal encodings are NOT used: learned absolute positions suffice
    at this scale and match the sibling project's conventions."""
    return None
