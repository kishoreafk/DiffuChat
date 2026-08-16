"""Tokenizer tests (spec §9): round-trip, [MASK] id = 65 = last, corpus purity."""

from __future__ import annotations

import numpy as np

from diffuchat.data import encode_corpus


def test_mask_id_is_last_and_65(tok):
    assert tok.vocab_size == 66
    assert tok.mask_token_id == 65
    assert tok.mask_token_id == tok.vocab_size - 1  # always the LAST id


def test_round_trip(tok, corpus):
    sample = corpus[:2000]
    ids = tok.encode(sample)
    assert tok.decode(ids) == sample  # lossless round-trip


def test_corpus_never_emits_mask(tok, corpus):
    ids = encode_corpus(tok, corpus)
    assert int(ids.max()) < tok.mask_token_id
    assert tok.mask_token_id not in set(ids.tolist())
    # and encode of plain text never returns the mask id either
    assert tok.mask_token_id not in tok.encode(corpus[:5000])


def test_decode_renders_mask_glyph(tok):
    text = tok.decode([tok.mask_token_id, tok.mask_token_id])
    assert text == "\u2593\u2593"


def test_unknown_char_maps_to_space(tok):
    out = tok.encode("a\u2603b")  # snowman is not in shakespeare vocab
    assert out == [tok.stoi["a"], tok.stoi[" "], tok.stoi["b"]]


def test_state_round_trip(tok):
    from diffuchat.data import CharTokenizer
    t2 = CharTokenizer.from_state(tok.state())
    assert t2.decode(t2.encode("hello world")) == "hello world"
    assert t2.mask_token_id == 65


def test_window_sampler_shapes(tok, corpus, rng):
    ids = encode_corpus(tok, corpus)
    w = rng  # placeholder to use fixture
    from diffuchat.data import random_window
    batch = random_window(ids, block=128, batch=3, rng=rng)
    assert batch.shape == (3, 128)
    assert batch.dtype == np.int64
