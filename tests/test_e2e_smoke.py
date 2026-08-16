"""End-to-end smoke (spec §9): train (tiny) -> generate -> edit, no crashes,
all invariants hold."""

from __future__ import annotations

from diffuchat.config import DenoiseLMConfig, GenConfig, TrainConfig
from diffuchat.data import CharTokenizer, shakespeare
from diffuchat.generate import edit, generate
from diffuchat.train import load_model, train


def test_full_pipeline_smoke(tmp_path):
    # ---- train a tiny model ---------------------------------------------
    cfg = DenoiseLMConfig(block_size=48, n_layer=2, n_head=2, d_model=64)
    tcfg = TrainConfig(steps=40, batch_size=32, lr=1e-3, warmup=10,
                       eval_every=40, val_batches=1, val_batch_size=8,
                       device="cpu")
    ckpt = tmp_path / "e2e.pt"
    out = train(cfg, tcfg, ckpt, verbose=False)
    assert out["best_val_loss"] < 4.1  # learned *something* in 60 steps

    # ---- reload from disk (product path) --------------------------------
    model, tok, ck = load_model(ckpt)
    assert ck["val_loss"] == out["best_val_loss"]
    assert tok.mask_token_id == 65

    # ---- generate --------------------------------------------------------
    gcfg = GenConfig(strategy="confidence", steps=8, answer_len=24,
                     temperature=0.0, seed=42)
    res = generate(model, tok, "you : the king said\nai :", gcfg)
    assert tok.mask_token_id not in res.canvas.ids
    assert len(res.frames) >= 2
    assert res.n_forward_passes >= 1

    # ---- edit one token; invariants -------------------------------------
    canvas = res.canvas
    pos = canvas.answer_start + canvas.answer_len // 2
    new_id = tok.stoi["q"]
    res2 = edit(model, tok, canvas, [(pos, new_id)],
                GenConfig(**{**vars(gcfg), "remask_window": 0}))
    assert res2.canvas.ids[pos] == new_id           # pin survived
    assert tok.mask_token_id not in res2.canvas.ids  # fully re-denoised
    assert res2.answer_text  # non-empty

    # ---- all three strategies terminate ----------------------------------
    for strat in ("random", "semi_ar"):
        r = generate(model, tok, "ai :",
                     GenConfig(strategy=strat, steps=8, answer_len=24,
                               semi_ar_block=8, temperature=0.0, seed=1))
        assert tok.mask_token_id not in r.canvas.ids, f"{strat} left masks"
