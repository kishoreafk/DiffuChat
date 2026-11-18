"""Configuration dataclasses and presets for DiffuChat (spec §5.1).

All hyperparameters live here so training / generation / evaluation can never
drift apart.  Presets: ``smoke`` (fast sanity), ``cpu`` (the laptop preset used
for the shipped artifacts), ``gpu`` (Colab / RTX 3050 preset).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class DenoiseLMConfig:
    """Architecture of the bidirectional denoiser (DenoiseLM)."""

    block_size: int = 128
    vocab_size: int = 66      # 65 shakespeare chars + [MASK]; set from tokenizer
    mask_token_id: int = 65   # always = vocab_size - 1 (tokenizer reserves it)
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128        # cpu preset ~0.83M params; gpu preset 384/12/8
    dropout: float = 0.0
    # T5-style learned relative-position bias with a neighbour-locality init.
    # Required at char scale: without it the fully-bidirectional masked LM
    # never escapes the unigram plateau (see model.py docstring / README).
    rel_pos_bias: bool = True


@dataclass
class TrainConfig:
    """MDLM training hyperparameters (spec §7.1)."""

    steps: int = 3000
    batch_size: int = 64
    lr: float = 3e-4
    warmup: int = 100
    seed: int = 42
    device: str = "auto"          # "auto" | "cpu" | "cuda"
    eval_every: int = 250
    val_frac: float = 0.1
    # masking: per-sample ratio t ~ U(t_min, 1.0)  (MDLM-style; never 0)
    mask_ratio_min: float = 0.05
    mask_ratio_max: float = 1.0
    # dialogue-pair mix (spec §6): ~5k "you :/ai :" pairs at 30% weight
    dialogue_pairs: int = 5000
    dialogue_weight: float = 0.3
    # shared host has 16 cores shared with the sibling project: stay at 4
    threads: int = 4
    # eval knobs
    val_batches: int = 8
    val_batch_size: int = 32


@dataclass
class GenConfig:
    """Decoding / edit configuration (spec §5.1, §7.2-§7.4)."""

    strategy: str = "confidence"   # "confidence" | "random" | "semi_ar"
    steps: int = 20                # denoising steps
    answer_len: int = 96           # masked span appended after prompt
    semi_ar_block: int = 24        # tokens per block when strategy == "semi_ar"
    temperature: float = 0.0       # 0 => argmax unmask (greedy); >0 sample
    seed: int | None = 42
    # edit() re-masking mode: 0 => re-mask the WHOLE unlocked answer region
    # (default); n > 0 => only tokens within +/- n of an edit position.
    remask_window: int = 0

    def __post_init__(self) -> None:
        if self.strategy not in ("confidence", "random", "semi_ar"):
            raise ValueError(
                f"unknown strategy {self.strategy!r}; "
                "expected 'confidence' | 'random' | 'semi_ar'"
            )


_PRESETS = {
    # name: (DenoiseLMConfig, TrainConfig)
    "smoke": (
        DenoiseLMConfig(block_size=64, n_layer=2, n_head=2, d_model=64),
        TrainConfig(steps=300, batch_size=32, warmup=25, eval_every=50),
    ),
    "cpu": (DenoiseLMConfig(), TrainConfig()),
    "gpu": (
        DenoiseLMConfig(n_layer=12, n_head=8, d_model=384),
        TrainConfig(steps=8000, batch_size=64, warmup=200, eval_every=500),
    ),
}


def get_preset(name: str) -> tuple[DenoiseLMConfig, TrainConfig]:
    """Return fresh (model_cfg, train_cfg) copies for a preset name."""
    try:
        mcfg, tcfg = _PRESETS[name]
    except KeyError:
        raise ValueError(
            f"unknown preset {name!r}; expected one of {sorted(_PRESETS)}"
        ) from None
    import copy

    return copy.deepcopy(mcfg), copy.deepcopy(tcfg)


def as_dict(obj) -> dict:
    """Dataclass -> plain dict (for JSON-safe checkpoint metadata)."""
    return asdict(obj)
