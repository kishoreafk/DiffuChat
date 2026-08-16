"""Matched autoregressive (AR) char-GPT baseline (spec §5.8).

A vendored, dependency-free causal transformer — the same nanoGPT-style stack
as ``model.DenoiseLM`` EXCEPT attention is causal (lower-triangular mask).
Architecture-matched to the diffusion model (layers/heads/d_model/block) and
trained for the SAME number of steps on the SAME data mix, so evaluate.py's
experiment 1 compares bidirectional diffusion vs causal AR on infilling.

The AR baseline never sees [MASK]: it trains on clean text with the plain
65-char vocab (an extra unused row in the embedding keeps the two checkpoints
shape-identical).

CLI:  python -m diffuchat.ar_baseline [--steps N] [--batch-size N] [--lr F]
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DenoiseLMConfig, TrainConfig, as_dict
from .data import CharTokenizer, encode_corpus, mine_dialogue_pairs, mixed_window, shakespeare, split_ids
from .train import cosine_lr

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = _REPO / "artifacts" / "ar_baseline.pt"


class ARBlock(nn.Module):
    """Pre-LN transformer block with CAUSAL attention (the only difference
    from diffuchat.model.Block)."""

    def __init__(self, cfg: DenoiseLMConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # CAUSAL
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(y)
        x = x + self.mlp(self.ln2(x))
        return x


class ARCharGPT(nn.Module):
    """Causal char-GPT matched to DenoiseLM's shape."""

    def __init__(self, cfg: DenoiseLMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList([ARBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel() + self.pos_emb.weight.numel()
        return n


def train_ar(
    cfg: DenoiseLMConfig,
    tcfg: TrainConfig,
    out_path: str | Path = DEFAULT_OUT,
    verbose: bool = True,
) -> dict:
    """Train the AR baseline on the same data mix for the same schedule."""
    torch.set_num_threads(tcfg.threads)
    torch.manual_seed(tcfg.seed)
    rng = np.random.default_rng(tcfg.seed)
    device = torch.device(
        "cuda" if (tcfg.device == "cuda" or
                   (tcfg.device == "auto" and torch.cuda.is_available()))
        else "cpu"
    )

    text = shakespeare()
    tok = CharTokenizer(text)
    cfg.vocab_size = tok.vocab_size
    cfg.mask_token_id = tok.mask_token_id
    ids = encode_corpus(tok, text)
    train_ids, val_ids = split_ids(ids, tcfg.val_frac)
    dial_ids = encode_corpus(tok, "".join(mine_dialogue_pairs(text, tcfg.dialogue_pairs, tcfg.seed)))

    model = ARCharGPT(cfg).to(device)
    if verbose:
        print(f"[ar] {model.num_params(False):,} params | device {device}")
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr,
                            betas=(0.9, 0.95), weight_decay=0.1)

    def next_ce(x: torch.Tensor) -> torch.Tensor:
        logits = model(x[:, :-1])
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))

    best_val = float("inf")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for step in range(tcfg.steps):
        lr = cosine_lr(step, tcfg.steps, tcfg.lr, tcfg.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        x = torch.from_numpy(
            mixed_window(train_ids, dial_ids, cfg.block_size, tcfg.batch_size,
                         rng, tcfg.dialogue_weight)
        ).to(device)
        loss = next_ce(x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % tcfg.eval_every == 0 or step + 1 == tcfg.steps:
            model.eval()
            tot, n = 0.0, 0
            with torch.no_grad():
                vrng = np.random.default_rng(1234)
                for _ in range(tcfg.val_batches):
                    xv = torch.from_numpy(
                        mixed_window(val_ids, None, cfg.block_size,
                                     tcfg.val_batch_size, vrng, 0.0)
                    ).to(device)
                    l = next_ce(xv)
                    tot += l.item()
                    n += 1
            model.train()
            val_loss = tot / max(n, 1)
            if verbose:
                print(f"[ar] step {step + 1:>5}/{tcfg.steps} train {loss.item():.4f} "
                      f"VAL {val_loss:.4f} ({time.time() - t0:.0f}s)")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {"model": model.state_dict(), "cfg": as_dict(cfg),
                     "tok": tok.state(), "val_loss": val_loss, "step": step + 1},
                    out_path,
                )
    if verbose:
        print(f"[ar] done | best val {best_val:.4f} -> {out_path}")
    return {"best_val_loss": best_val, "out_path": str(out_path)}


def load_ar(path: str | Path = DEFAULT_OUT, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    mcfg = DenoiseLMConfig(**{k: v for k, v in ck["cfg"].items()
                              if k in DenoiseLMConfig.__dataclass_fields__})
    model = ARCharGPT(mcfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, CharTokenizer.from_state(ck["tok"]), ck


@torch.no_grad()
def greedy_infill(
    model: ARCharGPT,
    context_ids: list[int],
    span_len: int,
) -> list[int]:
    """Greedy left-to-right fill of ``span_len`` tokens after the context.

    The AR model conditions only on the PAST (prefix) — it cannot see the text
    after the span; that handicap vs the bidirectional diffusion model is the
    whole point of evaluate.py experiment 1.
    """
    device = next(model.parameters()).device
    ids = list(context_ids)
    out: list[int] = []
    for _ in range(span_len):
        x = torch.tensor([ids[-model.cfg.block_size + 1:]], dtype=torch.long,
                         device=device)
        nxt = int(model(x)[0, -1].argmax())
        out.append(nxt)
        ids.append(nxt)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the matched AR baseline")
    ap.add_argument("--steps", type=int, default=1250,
                    help="match the diffusion model's actual step count")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    # config-matched to the cpu preset DenoiseLM (spec §5.8)
    cfg = DenoiseLMConfig()
    tcfg = TrainConfig(steps=args.steps, batch_size=args.batch_size, lr=args.lr,
                       warmup=100, eval_every=args.eval_every)
    train_ar(cfg, tcfg, args.out)


if __name__ == "__main__":
    main()
