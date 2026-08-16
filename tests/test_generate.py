"""Generation tests (spec §9): termination, determinism, semi-ar ordering,
frame accounting."""

from __future__ import annotations

import pytest
import torch

from diffuchat.config import GenConfig
from diffuchat.generate import generate, new_canvas


def _cfg(**kw):
    base = dict(strategy="confidence", steps=20, answer_len=96, temperature=0.0,
                seed=42)
    base.update(kw)
    return GenConfig(**base)


def test_generation_terminates_with_zero_masks(tiny_model, tok):
    res = generate(tiny_model, tok, "you : hello\nai :", _cfg(answer_len=48))
    assert tok.mask_token_id not in res.canvas.ids
    # budget: one forward pass per denoise step + safety fill
    assert res.n_forward_passes <= _cfg().steps + 2
    assert res.answer_text  # non-empty answer
    assert res.wall_time_s >= 0


def test_frames_length_matches_steps(tiny_model, tok):
    # 40 tokens / 20 steps -> k=2 -> exactly 20 passes: frames = init + 20
    res = generate(tiny_model, tok, "you : hi\nai :", _cfg(answer_len=40))
    assert len(res.frames) == 21  # initial canvas + one snapshot per pass


def test_determinism_temp0_seed(tiny_model, tok):
    a = generate(tiny_model, tok, "you : once more\nai :", _cfg())
    b = generate(tiny_model, tok, "you : once more\nai :", _cfg())
    assert a.canvas.ids == b.canvas.ids
    # different seed must still terminate (random strategy uses the rng)
    c = generate(tiny_model, tok, "you : once more\nai :",
                 _cfg(strategy="random", seed=7))
    assert tok.mask_token_id not in c.canvas.ids


def test_random_strategy_terminates_and_differs(tiny_model, tok):
    a = generate(tiny_model, tok, "ai :", _cfg(strategy="random", seed=1))
    b = generate(tiny_model, tok, "ai :", _cfg(strategy="random", seed=2))
    assert tok.mask_token_id not in a.canvas.ids
    assert a.canvas.ids != b.canvas.ids, "different seeds should diverge"


def test_semi_ar_fills_blocks_left_to_right(tiny_model, tok):
    cfg = _cfg(strategy="semi_ar", semi_ar_block=24)
    res = generate(tiny_model, tok, "ai :", cfg)
    assert tok.mask_token_id not in res.canvas.ids
    start, block = res.canvas.answer_start, cfg.semi_ar_block
    # frame index at which each position first becomes un-masked
    first_seen = {}
    for fi, frame in enumerate(res.frames):
        for i in range(start, len(frame)):
            if frame[i] != tok.mask_token_id and i not in first_seen:
                first_seen[i] = fi
    rel = sorted(i - start for i in first_seen)
    blocks_seen = [p // block for p in rel]
    # monotone non-decreasing in block order (later blocks appear later or same)
    assert all(b2 >= b1 for b1, b2 in zip(blocks_seen, blocks_seen[1:])), (
        f"semi_ar filled blocks out of order: {blocks_seen}"
    )
    # and each block is COMPLETE before the next block gains any token
    for fi, frame in enumerate(res.frames):
        got = sorted({p // block for p in range(start, len(frame))
                      if frame[p] != tok.mask_token_id})
        if got:
            assert got == list(range(max(got) + 1)), (
                f"frame {fi} has block {max(got)} filled but gaps below: {got}"
            )


def test_prompt_tokens_never_masked_or_changed(tiny_model, tok):
    prompt = "you : the king said\nai :"
    res = generate(tiny_model, tok, prompt, _cfg(answer_len=32))
    assert res.canvas.ids[: res.canvas.answer_start] == tok.encode(prompt)
    assert res.canvas.answer_start == len(tok.encode(prompt))


def test_long_prompt_is_truncated_not_crashed(tiny_model, tok):
    long_prompt = "blah " * 200
    res = generate(tiny_model, tok, long_prompt, _cfg())
    assert len(res.canvas.ids) <= tiny_model.cfg.block_size
    assert tok.mask_token_id not in res.canvas.ids


def test_new_canvas_geometry(tok):
    c = new_canvas([1, 2, 3], 4, tok.mask_token_id)
    assert c.answer_start == 3
    assert c.answer_len == 4
    assert c.ids == [1, 2, 3, 65, 65, 65, 65]
    assert c.masked_positions() == [3, 4, 5, 6]
    assert not any(c.locked)
