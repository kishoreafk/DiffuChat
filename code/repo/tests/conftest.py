"""Shared fixtures for the DiffuChat test suite (CPU, fast)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diffuchat.config import DenoiseLMConfig  # noqa: E402
from diffuchat.data import CharTokenizer, shakespeare  # noqa: E402
from diffuchat.model import DenoiseLM  # noqa: E402

torch.set_num_threads(4)


@pytest.fixture(scope="session")
def corpus() -> str:
    return shakespeare()


@pytest.fixture(scope="session")
def tok(corpus: str) -> CharTokenizer:
    return CharTokenizer(corpus)


@pytest.fixture()
def tiny_cfg() -> DenoiseLMConfig:
    return DenoiseLMConfig(block_size=64, n_layer=2, n_head=2, d_model=64)


@pytest.fixture()
def tiny_model(tiny_cfg: DenoiseLMConfig) -> DenoiseLM:
    torch.manual_seed(42)
    m = DenoiseLM(tiny_cfg)
    m.eval()
    return m


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)
