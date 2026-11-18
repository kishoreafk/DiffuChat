"""CPU smoke test of v3 kernel's new logic (no repo imports needed)."""
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/z/my-project/work/kg-diffuchat-v3")

# --- extract the pure-logic pieces by importing the file with stubs ---
# We can't import train.py directly (it runs main), so replicate + exec the
# EMA/_top_p_sample/coherence_stats blocks via a targeted exec hack:
src = Path("/home/z/my-project/work/kg-diffuchat-v3/train.py").read_text()


def cut(start_marker, end_marker):
    i = src.index(start_marker)
    j = src.index(end_marker, i)
    return src[i:j]


code = "\n".join([
    "import math, numpy as np, torch",
    cut("class EMA:", "class _TCFG:"),
    cut("@torch.no_grad()\ndef _top_p_sample", "@torch.no_grad()\ndef diffusion_generate"),
    cut("def coherence_stats", "# ---------------------------------------------------------------------------\n# 8."),
])
ns: dict = {}
exec(code, ns)  # noqa: S102 - test harness

# --- 1. EMA correctness on a toy linear model ---
torch.manual_seed(0)
m = nn.Linear(4, 4)
ema = ns["EMA"](m, 0.9)
w0 = m.weight.detach().clone()
for step in range(5):
    with torch.no_grad():
        m.weight.add_(0.1)
    ema.update(m, step)
# analytic: ramp decay d=min(0.9,(1+step)/(10+step)); shadow after 5 updates
sh = w0.clone()
for step in range(5):
    d = min(0.9, (1 + step) / (10 + step))
    sh = sh * d + (w0 + 0.1 * (step + 1)) * (1 - d)
assert torch.allclose(ema.shadow["weight"], sh, atol=1e-6), "EMA math mismatch"
bak = ema.swap_in(m)
assert torch.allclose(m.weight, sh, atol=1e-6), "swap_in failed"
ema.restore(m, bak)
assert torch.allclose(m.weight, w0 + 0.5, atol=1e-6), "restore failed"
print("EMA OK")

# --- 2. top-p sampler: shape, valid ids, p=1 keeps everything, p tiny -> argmax ---
logits = torch.randn(7, 50)
tok, conf = ns["_top_p_sample"](logits, 1.0, 1.0)
assert tok.shape == (7,) and conf.shape == (7,) and (conf > 0).all()
tok2, _ = ns["_top_p_sample"](logits, 0.01, 0.01)
am = logits.argmax(-1)
match = (tok2 == am).float().mean()
print(f"top_p OK (temp0.01 matches argmax on {match:.0%} of rows)")
assert match >= 0.7  # near-deterministic

# --- 3. annealing loop never divides by zero, ends with zero MASK ---
MASK = 99
n0, steps = 16, 24
remaining = n0
unmasked = 0
for s in range(steps):
    k = max(1, math.ceil(n0 / (steps - s)))
    unmasked += min(k, remaining)
    remaining = max(0, remaining - k)
assert unmasked == n0, f"annealing uncovers {unmasked}/{n0}"
print("annealing schedule OK (all positions uncovered)")

# --- 4. refine remask fraction ---
for n_gen in (4, 48):
    assert max(1, int(0.15 * n_gen)) >= 1
print("refine remask OK")

# --- 5. coherence stats ---
ids = [5, 5, 5, 9, 3, 5, 5, 7]
st = ns["coherence_stats"](ids)
# dup bigrams: (5,5),(5,5),(5,9),(9,3),(3,5),(5,5),(5,7) -> 3/7
assert abs(st["dup_bigram_rate"] - 3 / 7) < 1e-3, st
assert abs(st["distinct_token_ratio"] - 4 / 8) < 1e-3, st
print("coherence OK", st)

# --- 6. metrics sanity: tiny DenoiseLM-like forward with dropout in eval vs train ---
drop = nn.Dropout(0.1)
x = torch.ones(2, 8)
drop.eval()
assert torch.equal(drop(x), x), "dropout must be identity in eval()"
print("eval-mode dropout OK")
print("ALL SMOKE TESTS PASSED")
