"""MDLM-style training for the DiffuChat denoiser (spec §5.4, §7.1).

Training step:
    x  <- window(B, T)                      # clean text (70% plain / 30% dialogue)
    t_b ~ U(0.05, 1.0)                      # per-sample mask ratio
    m[b,i] ~ Bernoulli(t_b)                 # mask indicator
    x~[b,i] = mask_id if m[b,i] else x[b,i]
    loss = sum(CE(logits[b,i], x[b,i]) * m[b,i]) / (sum(m) + eps)

AdamW(0.9/0.95, wd 0.1), cosine LR, warmup, grad clip 1.0, best-val checkpoint
saved as {"model", "cfg", "tok", "val_loss"} -> artifacts/diffusion_lm.pt.

CLI:  python -m diffuchat.train --preset cpu [--steps N] [--out PATH]
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import DenoiseLMConfig, TrainConfig, as_dict, get_preset
from .data import (
    CharTokenizer,
    encode_corpus,
    mine_dialogue_pairs,
    mixed_window,
    shakespeare,
    split_ids,
)
from .model import DenoiseLM

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = _REPO / "artifacts" / "diffusion_lm.pt"


def masked_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Cross-entropy averaged over MASKED positions ONLY (spec §7.1).

    logits (B,T,V), targets (B,T) int64, mask (B,T) bool/float.
    Unmasked positions contribute exactly zero gradient — this is the property
    ``test_train.py`` checks with an adversarial batch.
    """
    ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
    ).view(targets.shape)
    return (ce * mask).sum() / (mask.sum() + eps)


def sample_mask(
    batch: int, block: int, tcfg: TrainConfig, rng: np.random.Generator, device
) -> torch.Tensor:
    """m[b,i] ~ Bernoulli(t_b), t_b ~ U(t_min, t_max) per sample."""
    t = rng.uniform(tcfg.mask_ratio_min, tcfg.mask_ratio_max, size=(batch, 1))
    m = rng.random((batch, block)) < t
    # rows that drew zero masks (possible at t=0.05): mask one random position
    empty = ~m.any(axis=1)
    if empty.any():
        m[empty, rng.integers(0, block, size=int(empty.sum()))] = True
    return torch.from_numpy(m.astype(np.float32)).to(device)


def teacher_forced_eval(
    model: DenoiseLM,
    val_ids: np.ndarray,
    tok: CharTokenizer,
    cfg: DenoiseLMConfig,
    tcfg: TrainConfig,
    rng: np.random.Generator,
    device,
) -> tuple[float, float]:
    """Return (val masked CE, exact-recovery rate of masked tokens)."""
    model.eval()
    totals, correct, n_mask = 0.0, 0, 0
    with torch.no_grad():
        for _ in range(tcfg.val_batches):
            x = torch.from_numpy(
                mixed_window(
                    val_ids, None, cfg.block_size, tcfg.val_batch_size, rng, 0.0
                )
            ).to(device)
            m = sample_mask(x.shape[0], cfg.block_size, tcfg, rng, device)
            xm = torch.where(m.bool(), tok.mask_token_id, x)
            logits = model(xm)
            ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), x.reshape(-1), reduction="none"
            ).view(x.shape)
            mb = m.bool()
            totals += (ce * m).sum().item()
            n_mask += m.sum().item()
            pred = logits.argmax(-1)
            correct += ((pred == x) & mb).sum().item()
    model.train()
    return totals / max(n_mask, 1.0), correct / max(n_mask, 1)


def cosine_lr(step: int, total: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))


def train(
    cfg: DenoiseLMConfig,
    tcfg: TrainConfig,
    out_path: str | Path = DEFAULT_OUT,
    verbose: bool = True,
) -> dict:
    """Train DenoiseLM with the MDLM objective; save the best-val checkpoint."""
    torch.set_num_threads(tcfg.threads)
    torch.manual_seed(tcfg.seed)
    np_seed = np.random.default_rng(tcfg.seed)

    device = torch.device("cuda" if tcfg.device == "cuda" else "cpu")
    if tcfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- data -----------------------------------------------------------
    text = shakespeare()
    tok = CharTokenizer(text)
    cfg.vocab_size = tok.vocab_size
    cfg.mask_token_id = tok.mask_token_id  # always vocab_size - 1
    ids = encode_corpus(tok, text)
    assert tok.mask_token_id not in set(ids.tolist()), "corpus must never emit [MASK]"
    train_ids, val_ids = split_ids(ids, tcfg.val_frac)

    pairs = mine_dialogue_pairs(text, tcfg.dialogue_pairs, tcfg.seed)
    dial_ids = encode_corpus(tok, "".join(pairs))

    model = DenoiseLM(cfg).to(device)
    n_params = model.num_params(non_embedding=False)
    if verbose:
        print(
            f"[train] preset model: {n_params:,} params | device {device} "
            f"| threads {torch.get_num_threads()} | dialogue pairs {len(pairs)}"
        )

    opt = torch.optim.AdamW(
        model.parameters(), lr=tcfg.lr, betas=(0.9, 0.95), weight_decay=0.1
    )

    best_val = float("inf")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    t0 = time.time()

    for step in range(tcfg.steps):
        lr = cosine_lr(step, tcfg.steps, tcfg.lr, tcfg.warmup)
        for g in opt.param_groups:
            g["lr"] = lr

        x = torch.from_numpy(
            mixed_window(
                train_ids,
                dial_ids,
                cfg.block_size,
                tcfg.batch_size,
                np_seed,
                tcfg.dialogue_weight,
            )
        ).to(device)
        m = sample_mask(x.shape[0], cfg.block_size, tcfg, np_seed, device)
        xm = torch.where(m.bool(), cfg.mask_token_id, x)
        logits = model(xm)
        loss = masked_ce_loss(logits, x, m)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % max(1, tcfg.eval_every // 5) == 0 or step == 0:
            if verbose:
                print(
                    f"[train] step {step + 1:>5}/{tcfg.steps} loss {loss.item():.4f} "
                    f"lr {lr:.2e}"
                )

        if (step + 1) % tcfg.eval_every == 0 or step + 1 == tcfg.steps:
            val_loss, recovery = teacher_forced_eval(
                model, val_ids, tok, cfg, tcfg, np.random.default_rng(1234), device
            )
            history.append(
                {"step": step + 1, "train_loss": loss.item(), "val_loss": val_loss,
                 "masked_recovery": recovery}
            )
            if verbose:
                el = time.time() - t0
                print(
                    f"[train] step {step + 1:>5} VAL {val_loss:.4f} "
                    f"recovery {recovery * 100:.1f}% ({el:.0f}s, "
                    f"{el / (step + 1):.3f} s/step)"
                )
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "model": model.state_dict(),
                        "cfg": as_dict(cfg),
                        "tok": tok.state(),
                        "val_loss": val_loss,
                        "step": step + 1,
                        "masked_recovery": recovery,
                        "tcfg": as_dict(tcfg),
                    },
                    out_path,
                )

    if verbose:
        print(
            f"[train] done in {time.time() - t0:.0f}s | best val {best_val:.4f} "
            f"-> {out_path}"
        )
    return {
        "best_val_loss": best_val,
        "val_history": history,
        "n_params": n_params,
        "out_path": str(out_path),
    }


def load_model(path: str | Path = DEFAULT_OUT, device="cpu"):
    """Load a trained checkpoint -> (model in eval mode, CharTokenizer, meta)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    mcfg = DenoiseLMConfig(**{k: v for k, v in ck["cfg"].items()
                              if k in DenoiseLMConfig.__dataclass_fields__})
    tok = CharTokenizer.from_state(ck["tok"])
    model = DenoiseLM(mcfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, tok, ck


def main() -> None:
    ap = argparse.ArgumentParser(description="Train DiffuChat (MDLM objective)")
    ap.add_argument("--preset", default="cpu", choices=["smoke", "cpu", "gpu"])
    ap.add_argument("--steps", type=int, default=None,
                    help="override preset step count (calibration / tests)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    mcfg, tcfg = get_preset(args.preset)
    if args.steps:
        tcfg.steps = args.steps
        tcfg.eval_every = max(10, min(tcfg.eval_every, args.steps // 3 or 1))
    if args.batch_size:
        tcfg.batch_size = args.batch_size
    if args.lr:
        tcfg.lr = args.lr
    train(mcfg, tcfg, args.out)


if __name__ == "__main__":
    main()
