"""Generation / editing engine for DiffuChat (spec §5.5, §7.2, §7.3, §7.4).

The canvas is the single source of truth: prompt tokens occupy
``[0, answer_start)`` and are never re-masked; the answer region
``[answer_start, len)`` starts all-[MASK] and is denoised iteratively.
``locked`` flags user pins (edits) that are NEVER re-masked or replaced —
the invariant that makes the editable-chat product possible (spec §7.4).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .config import GenConfig
from .data import CharTokenizer
from .model import DenoiseLM

# hard safety bound on denoise passes (guarantees termination)
_MAX_PASSES = 1024


@dataclass
class Canvas:
    """Current sequence + bookkeeping for the answer region and user locks."""

    ids: list[int]
    answer_start: int
    mask_id: int = -1
    locked: list[bool] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.locked:
            self.locked = [False] * len(self.ids)
        assert len(self.locked) == len(self.ids), "locked must align with ids"
        assert self.mask_id >= 0, "Canvas needs a valid mask_token_id"

    @property
    def answer_len(self) -> int:
        return len(self.ids) - self.answer_start

    def masked_positions(self) -> list[int]:
        """Masked, unlockable positions in the answer region."""
        return [
            i
            for i in range(self.answer_start, len(self.ids))
            if self.ids[i] == self.mask_id and not self.locked[i]
        ]


def new_canvas(prompt_ids: list[int], answer_len: int, mask_id: int) -> Canvas:
    """prompt + [MASK]*answer_len with an empty lock vector."""
    return Canvas(
        ids=list(prompt_ids) + [mask_id] * answer_len,
        answer_start=len(prompt_ids),
        mask_id=mask_id,
        locked=[False] * (len(prompt_ids) + answer_len),
    )


@dataclass
class GenerationResult:
    text: str                     # full decoded canvas (prompt + answer)
    answer_text: str              # decoded answer region only
    frames: list[list[int]]       # canvas snapshots (initial + one per pass)
    n_forward_passes: int
    wall_time_s: float
    canvas: Canvas | None = None  # final canvas (for chaining edits)


# ---------------------------------------------------------------------------
# one denoise pass (spec §7.2)
# ---------------------------------------------------------------------------

def denoise_step(
    model: DenoiseLM,
    canvas: Canvas,
    cfg: GenConfig,
    rng: np.random.Generator,
    k: int,
    allowed: list[int] | None = None,
) -> list[int]:
    """One forward pass; unmask k masked positions.

    strategy "confidence": the k masked positions with the highest
      max-softmax probability (ties -> leftmost) are set to their argmax token
      (temperature 0) or a temperature sample (temperature > 0).
    strategy "random": k uniformly-sampled masked positions.
    ``allowed`` restricts unmasking to a position subset (semi-ar blocks).

    Returns the positions actually unmasked (for frame display).  Locked
    positions are never chosen and never changed (spec §7.4 invariant).
    """
    device = next(model.parameters()).device
    x = torch.tensor([canvas.ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x)[0].float()  # (T, V)
    # an UNMASK operation must never write [MASK] (LLaDA-style): a random or
    # under-trained model would otherwise "unmask" positions to the mask token
    logits[:, canvas.mask_id] = float("-inf")

    cands = list(allowed) if allowed is not None else canvas.masked_positions()
    if not cands or k <= 0:
        return []

    if cfg.temperature > 0.0:
        picks = []
        confs = []
        for p in cands:
            row = torch.softmax(logits[p] / cfg.temperature, dim=-1).numpy()
            tok_id = int(rng.choice(len(row), p=row / row.sum()))
            picks.append(tok_id)
            confs.append(float(row[tok_id]))
    else:
        probs = torch.softmax(logits, dim=-1)
        picks = [int(probs[p].argmax()) for p in cands]
        confs = [float(probs[p, t]) for p, t in zip(cands, picks)]

    if cfg.strategy == "random":
        order = list(rng.permutation(len(cands)))
    else:  # confidence (also the inner loop of semi_ar)
        order = sorted(range(len(cands)), key=lambda j: (-confs[j], cands[j]))

    chosen = order[: min(k, len(order))]
    for j in chosen:
        canvas.ids[cands[j]] = picks[j]
    return [cands[j] for j in chosen]


def _fill_remaining(model: DenoiseLM, canvas: Canvas) -> int:
    """Safety net: force-argmax any residual [MASK] (never fires when the
    k = ceil(remaining/steps) arithmetic already emptied the canvas; exists to
    guarantee termination under all rounding paths)."""
    left = [
        i
        for i in range(canvas.answer_start, len(canvas.ids))
        if canvas.ids[i] == canvas.mask_id
    ]
    if not left:
        return 0
    device = next(model.parameters()).device
    x = torch.tensor([canvas.ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x)[0].float()
    logits[:, canvas.mask_id] = float("-inf")  # never fill with [MASK]
    for i in left:
        canvas.ids[i] = int(logits[i].argmax())
    return len(left)


# ---------------------------------------------------------------------------
# full generation loop (strategies per spec §7.2 / §7.3)
# ---------------------------------------------------------------------------

def generate(
    model: DenoiseLM,
    tok: CharTokenizer,
    prompt: str,
    cfg: GenConfig,
    canvas: Canvas | None = None,
) -> GenerationResult:
    """Denoise an all-[MASK] answer region appended after the prompt.

    ``canvas`` may be passed in (the edit() path) — only its unlocked,
    still-masked answer positions are denoised; locked tokens are frozen.
    """
    t0 = time.time()
    model.eval()
    rng = np.random.default_rng(cfg.seed)
    mask_id = tok.mask_token_id

    if canvas is None:
        prompt_ids = tok.encode(prompt)
        block = model.cfg.block_size
        answer_len = max(1, min(cfg.answer_len, block - 1))
        # truncate the prompt (keep its END, nearest the answer) if too long
        prompt_ids = prompt_ids[-(block - answer_len) :]
        canvas = new_canvas(prompt_ids, answer_len, mask_id)

    frames: list[list[int]] = [list(canvas.ids)]
    n_pass = 0

    if cfg.strategy == "semi_ar":
        blocks = [
            (a, min(a + cfg.semi_ar_block, len(canvas.ids)))
            for a in range(canvas.answer_start, len(canvas.ids), cfg.semi_ar_block)
        ]
        per_block = max(2, cfg.steps // max(1, len(blocks)))
        for a, b in blocks:
            allowed = [
                i for i in range(a, b)
                if canvas.ids[i] == canvas.mask_id and not canvas.locked[i]
            ]
            k = max(1, -(-len(allowed) // per_block)) if allowed else 0
            while allowed and n_pass < _MAX_PASSES:
                denoise_step(model, canvas, cfg, rng, min(k, len(allowed)),
                             allowed=allowed)
                n_pass += 1
                frames.append(list(canvas.ids))
                allowed = [
                    i for i in range(a, b)
                    if canvas.ids[i] == canvas.mask_id and not canvas.locked[i]
                ]
    else:
        # k = ceil(L0 / steps) computed ONCE from the initial answer length
        # (spec §7.2); the final pass may unmask fewer if L0 % k != 0.
        n_masked0 = len(canvas.masked_positions())
        k = max(1, -(-n_masked0 // max(1, cfg.steps)))
        while n_pass < _MAX_PASSES:
            left = canvas.masked_positions()
            if not left:
                break
            denoise_step(model, canvas, cfg, rng, min(k, len(left)))
            n_pass += 1
            frames.append(list(canvas.ids))

    filled = _fill_remaining(model, canvas)
    n_pass += filled
    if filled:
        frames.append(list(canvas.ids))

    return GenerationResult(
        text=tok.decode(canvas.ids),
        answer_text=tok.decode(canvas.ids[canvas.answer_start :]),
        frames=frames,
        n_forward_passes=n_pass,
        wall_time_s=time.time() - t0,
        canvas=canvas,
    )


# ---------------------------------------------------------------------------
# edit -> re-denoise (the product primitive, spec §7.4)
# ---------------------------------------------------------------------------

def edit(
    model: DenoiseLM,
    tok: CharTokenizer,
    canvas: Canvas,
    edits: list[tuple[int, int]],
    cfg: GenConfig,
) -> GenerationResult:
    """Apply user edits, re-mask unlocked answer tokens, re-denoise (§7.4).

    edits: list of (pos, new_id); pos is in ABSOLUTE canvas coordinates and
    must lie inside the answer region.  For each edit:
        canvas.ids[pos] = new_id ; canvas.locked[pos] = True
    Then every answer-region token that is NOT locked is reset to [MASK]
    (or, if cfg.remask_window > 0, only tokens within +-remask_window of an
    edit).  Locked tokens are never re-masked and never replaced (invariant).
    """
    if not edits:
        raise ValueError("edit() needs at least one (pos, new_id) edit")
    for pos, new_id in edits:
        if not (canvas.answer_start <= pos < len(canvas.ids)):
            raise IndexError(
                f"edit position {pos} outside answer region "
                f"[{canvas.answer_start}, {len(canvas.ids)})"
            )
        canvas.ids[pos] = int(new_id)
        canvas.locked[pos] = True

    if cfg.remask_window > 0:
        targets: set[int] = set()
        for pos, _ in edits:
            lo = max(canvas.answer_start, pos - cfg.remask_window)
            hi = min(len(canvas.ids), pos + cfg.remask_window + 1)
            targets.update(i for i in range(lo, hi) if not canvas.locked[i])
    else:
        targets = {
            i
            for i in range(canvas.answer_start, len(canvas.ids))
            if not canvas.locked[i]
        }
    for i in targets:
        canvas.ids[i] = canvas.mask_id

    return generate(model, tok, "", cfg, canvas=canvas)
