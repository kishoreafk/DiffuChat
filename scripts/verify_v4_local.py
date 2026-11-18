#!/usr/bin/env python3
"""Local CPU verification of the v4 checkpoints pulled from Kaggle.

1. Load diffusion_best.pt + ar_best.pt strictly into repo classes.
2. Run the improved diffusion sampler (temp 0.8 + top-p 0.95 + annealed
   confidence + remask-refine) on two prompts.
3. Run AR generation on the same prompts for comparison.
4. Re-eval a small masked batch to confirm finite losses.
"""
import json
import sys

import torch

sys.path.insert(0, "/home/z/my-project/work/diffuchat/src")
from diffuchat.config import DenoiseLMConfig  # noqa: E402
from diffuchat.model import DenoiseLM  # noqa: E402
from diffuchat.ar_baseline import ARCharGPT  # noqa: E402

CKPT_DIR = "/home/z/my-project/download/diffuchat-v4/weights"
TOK_PATH = "/home/z/my-project/download/diffuchat_artifacts/tokenizer.json"
MASK_ID = 0
DEVICE = "cpu"


def load_diffusion():
    ck = torch.load(f"{CKPT_DIR}/diffusion_best.pt", map_location="cpu", weights_only=False)
    mcfg = DenoiseLMConfig(**{k: v for k, v in ck["mcfg"].items()
                              if k in DenoiseLMConfig.__dataclass_fields__})
    model = DenoiseLM(mcfg)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    print(f"[LOAD] diffusion best: step={ck['step']} val={ck['val_loss']:.4f} "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    return model, mcfg


def load_ar():
    ck = torch.load(f"{CKPT_DIR}/ar_best.pt", map_location="cpu", weights_only=False)
    acfg = DenoiseLMConfig(**{k: v for k, v in ck["mcfg"].items()
                              if k in DenoiseLMConfig.__dataclass_fields__})
    model = ARCharGPT(acfg)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    print(f"[LOAD] ar best: step={ck['step']} val={ck['val_loss']:.4f}")
    return model, acfg


def encode(tok, text, block):
    ids = tok.encode(text).ids[-(block - 64):]
    return ids


def improved_generate(model, mcfg, prompt_ids, gen_len=48, seed=0):
    """Mirror of the kernel's improved sampler: 24 passes, temp 0.8, top-p 0.95,
    annealed confidence threshold, one remask-refine pass."""
    g = torch.Generator().manual_seed(seed)
    block = mcfg.block_size
    vocab = mcfg.vocab_size
    n_passes = 24
    prompt_len = len(prompt_ids)
    total = prompt_len + gen_len
    ids = list(prompt_ids) + [MASK_ID] * gen_len

    def logits_of(cur):
        x = torch.tensor([cur[-block:]], dtype=torch.long)
        with torch.no_grad():
            return model(x)[0]  # [T, V]

    for p in range(n_passes):
        remaining = int((ids[prompt_len:] == [MASK_ID] * 1)[0].sum()) if False else sum(1 for t in ids[prompt_len:] if t == MASK_ID)
        if remaining == 0:
            break
        logits = logits_of(ids)
        # progress fraction for annealing: threshold rises over passes
        frac = p / max(1, n_passes - 1)
        thr = 0.7 + 0.29 * frac  # annealed confidence threshold
        mask_pos = [i for i in range(total) if ids[i] == MASK_ID]
        probs = torch.softmax(logits[1 - 1:][: total] if False else logits[:total] * 1.0 / 0.8, dim=-1)
        conf, pred = probs.max(dim=-1)
        # top-p filter per masked position
        new_tokens = []
        for i in mask_pos:
            lp = probs[i]
            sp, sidx = torch.sort(lp, descending=True)
            csum = torch.cumsum(sp, dim=-1)
            keep = csum - sp < 0.95
            if not keep.any():
                keep[0] = True
            filt = torch.zeros_like(lp)
            filt[sidx[keep]] = sp[keep]
            filt = filt / filt.sum()
            nxt = torch.multinomial(filt, 1, generator=g).item()
            new_tokens.append((i, nxt, float(conf[i])))
        # unmask positions whose confidence exceeds annealed threshold
        # (always unmask at least the most confident one)
        new_tokens.sort(key=lambda t: -t[2])
        n_unmask = max(1, sum(1 for t in new_tokens if t[2] >= thr))
        for i, tok_id, _ in new_tokens[:n_unmask]:
            ids[i] = tok_id
    # remask-refine: 1 extra pass on the 15% least-confident answer tokens
    logits = logits_of(ids)
    probs = torch.softmax(logits[:total] / 0.8, dim=-1)
    conf, pred = probs.max(dim=-1)
    ans_pos = list(range(prompt_len, total))
    ans_pos.sort(key=lambda i: conf[i])
    n_remask = max(1, int(0.15 * len(ans_pos)))
    for i in ans_pos[:n_remask]:
        ids[i] = MASK_ID
    # final refine pass: unmask everything greedily with temp 0.8 top-p
    for _ in range(3):
        mask_pos = [i for i in range(total) if ids[i] == MASK_ID]
        if not mask_pos:
            break
        logits = logits_of(ids)
        probs = torch.softmax(logits[:total] / 0.8, dim=-1)
        conf, pred = probs.max(dim=-1)
        mask_pos.sort(key=lambda i: -float(conf[i]))
        half = max(1, len(mask_pos) // 2)
        for i in mask_pos[:half]:
            lp = probs[i]
            sp, sidx = torch.sort(lp, descending=True)
            csum = torch.cumsum(sp, dim=-1)
            keep = csum - sp < 0.95
            if not keep.any():
                keep[0] = True
            filt = torch.zeros_like(lp)
            filt[sidx[keep]] = sp[keep]
            filt = filt / filt.sum()
            ids[i] = torch.multinomial(filt, 1, generator=g).item()
    return ids[prompt_len:]


def ar_generate(model, acfg, prompt_ids, gen_len=48, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = list(prompt_ids)
    block = acfg.block_size
    assert block is not None
    for _ in range(gen_len):
        x = torch.tensor([ids[-block:]], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)[:, -1, :]
        probs = torch.softmax(logits / 0.8, dim=-1)[0]
        sp, sidx = torch.sort(probs, descending=True)
        csum = torch.cumsum(sp, dim=-1)
        keep = csum - sp < 0.95
        if not keep.any():
            keep[0] = True
        filt = torch.zeros_like(probs)
        filt[sidx[keep]] = sp[keep]
        filt = filt / filt.sum()
        ids.append(torch.multinomial(filt, 1, generator=g).item())
    return ids[len(prompt_ids):]


def main():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOK_PATH)

    dmodel, mcfg = load_diffusion()
    amodel, acfg = load_ar()

    prompts = [
        "Once upon a time, there was a little girl named",
        "The city was founded in the year",
    ]
    for pi, p in enumerate(prompts):
        pids = encode(tok, p, mcfg.block_size)
        d_out = improved_generate(dmodel, mcfg, pids, seed=42 + pi)
        a_out = ar_generate(amodel, acfg, pids, seed=42 + pi)
        print(f"\nPROMPT: {p!r}")
        print(f"  DIFFUSION v4: {tok.decode(d_out)!r}")
        print(f"  AR        v4: {tok.decode(a_out)!r}")

    # masked-loss sanity on real text
    text = ("The quick brown fox jumps over the lazy dog. "
            "Machine learning models learn patterns from data.")
    ids = tok.encode(text).ids
    x = torch.tensor([ids], dtype=torch.long)
    T = x.shape[1]
    mask = torch.rand(x.shape, generator=torch.Generator().manual_seed(1)) < 0.3
    x_m = x.clone()
    x_m[mask] = MASK_ID
    with torch.no_grad():
        logits = dmodel(x_m)
    tgt = x[mask]
    ce = torch.nn.functional.cross_entropy(logits[mask], tgt)
    print(f"\n[VAL] diffusion masked CE on toy text: {float(ce):.3f} (finite={torch.isfinite(ce).item()})")
    print("OK")


if __name__ == "__main__":
    main()
