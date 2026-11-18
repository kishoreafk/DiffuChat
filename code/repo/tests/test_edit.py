"""Edit tests (spec §9): locked-token survival (100%), neighborhood re-flow,
re-mask window boundaries."""

from __future__ import annotations

import pytest

from diffuchat.config import GenConfig
from diffuchat.generate import Canvas, edit, generate


def _cfg(**kw):
    base = dict(strategy="confidence", steps=12, answer_len=32, temperature=0.0,
                seed=42)
    base.update(kw)
    return GenConfig(**base)


@pytest.fixture()
def base_canvas(tiny_model, tok):
    res = generate(tiny_model, tok, "you : the king\nai :", _cfg())
    # clone it (fresh, unlocked) so mutations never leak across tests
    return Canvas(ids=list(res.canvas.ids), answer_start=res.canvas.answer_start,
                  mask_id=tok.mask_token_id, locked=[False] * len(res.canvas.ids))


def test_locked_tokens_survive_100pct(tiny_model, tok, base_canvas):
    canvas = base_canvas
    # pin three scattered answer positions to unusual tokens
    pins = [(canvas.answer_start + 3, tok.stoi["q"]),
            (canvas.answer_start + 10, tok.stoi["z"]),
            (canvas.answer_start + 20, tok.stoi["x"])]
    res = edit(tiny_model, tok, canvas, pins, _cfg())
    for pos, want in pins:
        assert res.canvas.ids[pos] == want, f"pin at {pos} did not survive"
        assert res.canvas.locked[pos] is True
    assert tok.mask_token_id not in res.canvas.ids  # fully re-denoised


def test_edit_changes_neighborhood(tiny_model, tok, base_canvas):
    canvas = base_canvas
    original = list(canvas.ids)
    pos = canvas.answer_start + canvas.answer_len // 2
    res = edit(tiny_model, tok, canvas, [(pos, tok.stoi["q"])], _cfg())
    assert res.canvas.ids != original, "re-denoise must differ from pre-edit canvas"
    changed = sum(1 for a, b in zip(original, res.canvas.ids) if a != b)
    assert changed >= 1 and res.canvas.ids[pos] == tok.stoi["q"]


def test_edit_rejects_outside_answer_region(tiny_model, tok, base_canvas):
    with pytest.raises(IndexError):
        edit(tiny_model, tok, base_canvas, [(0, tok.stoi["a"])], _cfg())
    with pytest.raises(ValueError):
        edit(tiny_model, tok, base_canvas, [], _cfg())


def test_remask_window_respects_boundary(tiny_model, tok, base_canvas):
    """With remask_window=2, only positions within +/-2 of the edit are
    re-masked; everything else outside the window must keep its token."""
    canvas = base_canvas
    original = list(canvas.ids)
    pos = canvas.answer_start + 15
    w = 2
    cfg = _cfg(remask_window=w)
    res = edit(tiny_model, tok, canvas, [(pos, tok.stoi["k"])], cfg)
    lo, hi = pos - w, pos + w
    for i in range(canvas.answer_start, len(original)):
        if not (lo <= i <= hi):
            assert res.canvas.ids[i] == original[i], (
                f"position {i} outside window +/-{w} was re-masked"
            )
    # the pin survives, as always
    assert res.canvas.ids[pos] == tok.stoi["k"]
    # inside the window at least the neighbors re-flowed
    changed_inside = sum(
        1 for i in range(lo, hi + 1) if res.canvas.ids[i] != original[i]
    )
    assert changed_inside >= 1
