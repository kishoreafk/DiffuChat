"""Data utilities for DiffuChat (spec §5.2, §6).

- ``CharTokenizer``: character tokenizer with an appended ``[MASK]`` entry as
  the LAST vocab item (id = len(chars), i.e. 65 for TinyShakespeare).
- ``shakespeare(data_dir)``: loads TinyShakespeare; reuses the sibling
  project's download if present (polite fallback), else downloads it.
- ``random_window``: batch-of-windows sampler.
- ``mine_dialogue_pairs``: synthesises "you : <prompt>\\nai : <completion>"
  pairs from character speeches (the v1 instruction-tuning stand-in, spec §6).
- ``mixed_window``: samples 70% plain-text windows / 30% dialogue-pair windows.

Only stdlib + numpy + torch are used.
"""

from __future__ import annotations

import shutil
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)

# Repo root = <root>/diffuchat; sibling = <root>/learnatask (read-only peek).
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = _REPO_ROOT / "data"
_SIBLING_DATA = _REPO_ROOT.parent / "learnatask" / "data" / "input.txt"

# SSRF hardening: downloads are restricted to this https host allowlist and
# no user-controlled URL ever reaches the fetch path.
_ALLOWED_HOSTS = {"raw.githubusercontent.com"}


def _secure_download(url: str, dest: Path, timeout: int = 60) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in _ALLOWED_HOSTS:
        raise ValueError(
            f"refusing to download from non-allowlisted source: {url!r} "
            f"(allowed: https://{sorted(_ALLOWED_HOSTS)}/*)")
    req = urllib.request.Request(url, headers={"User-Agent": "diffuchat/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if urllib.parse.urlparse(r.geturl()).hostname not in _ALLOWED_HOSTS:
            raise ValueError("redirect left the allowlisted host; aborting")
        raw = r.read()
    dest.write_bytes(raw)


def shakespeare(data_dir: str | Path | None = None) -> str:
    """Return the TinyShakespeare corpus text, downloading/copying if needed.

    Order of preference (spec §5.2): local ``data/input.txt``, then the
    sibling ``../learnatask/data/input.txt`` (copied so both repos stay
    independent), then a fresh download from the canonical URL.
    """
    data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    path = data_dir / "input.txt"
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        if _SIBLING_DATA.exists():
            shutil.copy(_SIBLING_DATA, path)
            print(f"[data] copied sibling corpus {_SIBLING_DATA} -> {path}")
        else:
            print(f"[data] downloading TinyShakespeare from {DATA_URL} ...")
            _secure_download(DATA_URL, path)
            print(f"[data] saved {path} ({path.stat().st_size:,} bytes)")
    return path.read_text(encoding="utf-8")


class CharTokenizer:
    """Char-level tokenizer with a trailing ``[MASK]`` vocab entry.

    ids [0, len(chars)) map to corpus characters; id ``len(chars)`` (65 for
    TinyShakespeare) is the mask and is NEVER produced by ``encode`` of normal
    text — it only enters canvases via ``mask_token_id`` directly.  ``decode``
    renders the mask as ``'\u2593'`` (block shade) for frame display.
    """

    mask_char = "\u2593"  # display glyph for [MASK]; not part of corpus vocab

    def __init__(self, text: str):
        chars = sorted(set(text))
        if self.mask_char in chars:
            raise ValueError("corpus must not contain the mask display glyph")
        self.chars: list[str] = chars
        self.stoi: dict[str, int] = {c: i for i, c in enumerate(chars)}
        self.mask_token_id = len(chars)

    @property
    def vocab_size(self) -> int:
        return len(self.chars) + 1  # + [MASK]

    def encode(self, text: str, strict: bool = False) -> list[int]:
        """Text -> ids.  Unknown chars raise (strict) or map to space."""
        out: list[int] = []
        stoi = self.stoi
        for ch in text:
            idx = stoi.get(ch)
            if idx is None:
                if strict:
                    raise ValueError(f"character {ch!r} not in vocab")
                idx = stoi[" "]
            out.append(idx)
        return out

    def decode(self, ids: Iterable[int]) -> str:
        itos = {i: c for i, c in enumerate(self.chars)}
        itos[self.mask_token_id] = self.mask_char
        return "".join(itos[int(i)] for i in ids)

    def state(self) -> dict:
        return {"chars": "".join(self.chars), "mask_token_id": self.mask_token_id}

    @classmethod
    def from_state(cls, d: dict) -> "CharTokenizer":
        tok = cls.__new__(cls)
        tok.chars = list(d["chars"])
        tok.stoi = {c: i for i, c in enumerate(tok.chars)}
        tok.mask_token_id = int(d["mask_token_id"])
        assert tok.mask_token_id == len(tok.chars), "mask id must be vocab-1"
        return tok


def encode_corpus(tok: CharTokenizer, text: str) -> np.ndarray:
    """Corpus text -> int64 numpy array (fast ASCII LUT path)."""
    if text.isascii():
        lut = np.full(256, -1, dtype=np.int64)  # char code -> id
        for i, c in enumerate(tok.chars):
            lut[ord(c)] = i
        codes = np.frombuffer(text.encode("ascii"), dtype=np.uint8).astype(np.int64)
        ids = lut[codes]
        if (ids < 0).any():
            bad = chr(int(codes[ids < 0][0]))
            raise ValueError(f"corpus contains char {bad!r} outside vocab")
        return ids
    return np.array(tok.encode(text, strict=True), dtype=np.int64)


def split_ids(ids: np.ndarray, val_frac: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Contiguous 90/10 split (first part train, last part val)."""
    n_val = int(len(ids) * val_frac)
    return ids[:-n_val], ids[-n_val:]


def random_window(
    ids: np.ndarray, block: int, batch: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample ``batch`` random windows of length ``block`` -> (batch, block)."""
    if len(ids) <= block:
        raise ValueError("corpus shorter than block size")
    ix = rng.integers(0, len(ids) - block, size=batch)
    return np.stack([ids[i : i + block] for i in ix])


# --------------------------------------------------------------------------
# Dialogue-pair mining (spec §6): "you : <prompt>\nai : <completion>" pairs
# mined from character speeches; ~5k pairs mixed in at 30% weight.
# --------------------------------------------------------------------------

def _iter_speeches(text: str) -> Iterable[str]:
    """Yield speech bodies from TinyShakespeare play format.

    A speaker line is a short line ending in ':' with no sentence punctuation;
    the speech lines follow until a blank line.
    """
    cur: list[str] = []
    in_speech = False
    for raw in text.split("\n"):
        line = raw.strip()
        is_speaker = (
            line.endswith(":")
            and 1 < len(line) < 40
            and not any(p in line[:-1] for p in ".,;!?")
            and any(c.isalpha() for c in line)
        )
        if is_speaker:
            in_speech = True
            cur = []
        elif line == "":
            if in_speech and cur:
                yield " ".join(cur)
            in_speech = False
            cur = []
        elif in_speech:
            cur.append(line)
    if in_speech and cur:
        yield " ".join(cur)


def _split_at_word(text: str, pos: int) -> tuple[str, str]:
    """Split text at ~pos, snapped outward to whitespace (prompt, completion)."""
    if pos <= 0:
        return "", text
    if pos >= len(text):
        return text, ""
    left = text.rfind(" ", 0, pos)
    right = text.find(" ", pos)
    cands = [p for p in (left, right) if p != -1]
    cut = min(cands, key=lambda p: abs(p - pos)) if cands else pos
    return text[:cut], text[cut + 1 :]


def mine_dialogue_pairs(
    text: str, n_pairs: int = 5000, seed: int = 42, min_len: int = 24
) -> list[str]:
    """Mine prompt/completion pairs from character speeches (spec §6)."""
    pairs: list[str] = []
    for speech in _iter_speeches(text):
        speech = " ".join(speech.split())  # normalise whitespace
        if len(speech) < min_len * 2:
            continue
        prompt, completion = _split_at_word(speech, len(speech) // 2)
        if not prompt or not completion:
            continue
        pairs.append(f"you : {prompt}\nai : {completion}\n\n")
    if len(pairs) > n_pairs:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pairs), size=n_pairs, replace=False)
        pairs = [pairs[i] for i in sorted(idx)]
    return pairs


def mixed_window(
    main_ids: np.ndarray,
    dialogue_ids: np.ndarray | None,
    block: int,
    batch: int,
    rng: np.random.Generator,
    dialogue_weight: float = 0.3,
) -> np.ndarray:
    """Batch of windows: each row from the dialogue stream w.p. ``weight``."""
    if dialogue_ids is None or dialogue_weight <= 0 or len(dialogue_ids) <= block:
        return random_window(main_ids, block, batch, rng)
    n_dial = int(rng.binomial(batch, dialogue_weight))
    n_main = batch - n_dial
    rows = []
    if n_main:
        rows.append(random_window(main_ids, block, n_main, rng))
    if n_dial:
        rows.append(random_window(dialogue_ids, block, n_dial, rng))
    out = np.concatenate(rows, axis=0)
    return out[rng.permutation(batch)]


def chat_prefix(user_text: str) -> str:
    """Wrap free user text in the trained dialogue format (for generation)."""
    return f"you : {user_text}\nai :"


def most_common_char(ids: Sequence[int] | np.ndarray) -> int:
    """Most frequent token id (frequency-baseline filler for eval)."""
    ids = np.asarray(ids)
    counts = np.bincount(ids)
    return int(np.argmax(counts))
