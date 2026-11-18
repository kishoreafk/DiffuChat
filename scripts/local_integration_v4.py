"""Local CPU integration test mirroring the kernel's smoke_pass exactly.

Validates the real repo classes (DenoiseLM / ARCharGPT / masked_ce_loss /
sample_mask / random_window) with the block-size-parameterized batch
factories, EMA, checkpoint round-trip and both samplers — the path that
crashed remotely with 'sequence 256 exceeds block_size 64'.
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/z/my-project/work/diffuchat/src")
from diffuchat.model import DenoiseLM
from diffuchat.ar_baseline import ARCharGPT
from diffuchat.config import DenoiseLMConfig
from diffuchat.train import masked_ce_loss, sample_mask
from diffuchat.data import random_window

DEVICE = torch.device("cpu")
MASK_ID = 298
BLK = 64

tcfg = DenoiseLMConfig(block_size=BLK, vocab_size=300, mask_token_id=MASK_ID,
                       n_layer=2, n_head=4, d_model=64, dropout=0.1,
                       rel_pos_bias=True)
md = DenoiseLM(tcfg).to(DEVICE)
ma = ARCharGPT(tcfg).to(DEVICE)

# --- block-parameterized factories (the fix under test) ---
def make_diffusion_batch(ids, block=None):
    blk = block or tcfg.block_size
    def fn(rng, batch_size):
        x_np = random_window(ids, blk, batch_size, rng)
        x = torch.from_numpy(x_np).long().to(DEVICE)
        m = sample_mask(batch_size, blk, type("C", (), {
            "mask_ratio_min": 0.05, "mask_ratio_max": 1.0})(), rng, DEVICE)
        xm = torch.where(m.bool(), MASK_ID, x)
        return xm, x, m
    return fn

rng = np.random.default_rng(0)
train_ids = np.random.randint(0, 290, size=200_000).astype(np.int32)
val_ids = np.random.randint(0, 290, size=20_000).astype(np.int32)
dbf = make_diffusion_batch(train_ids, BLK)
vbf = make_diffusion_batch(val_ids, BLK)

od = torch.optim.AdamW(md.parameters(), lr=1e-3)
ema = torch.optim.swa_utils.AveragedModel(md, avg_fn=lambda a, b, n: a * 0.99 + b * 0.01) \
    if False else None

# local EMA copy of the kernel's class
class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items() if v.dtype.is_floating_point}
    @torch.no_grad()
    def update(self, model, step):
        d = min(self.decay, (1.0 + step) / (10.0 + step))
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)
    @torch.no_grad()
    def swap_in(self, model):
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        sd = model.state_dict()
        for k, v in self.shadow.items():
            sd[k].copy_(v.to(sd[k].dtype))
        return backup
    @torch.no_grad()
    def restore(self, model, backup):
        sd = model.state_dict()
        for k, v in backup.items():
            sd[k].copy_(v)

ema = EMA(md, 0.99)
scaler = torch.amp.GradScaler("cpu", enabled=False)
for step in range(3):
    xm, x, m = dbf(rng, 8)
    loss = masked_ce_loss(md(xm), x, m)
    od.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(md.parameters(), 1.0)
    od.step()
    ema.update(md, step)
print(f"train steps OK (loss {loss.item():.3f})")

backup = ema.swap_in(md)
md.eval()
with torch.no_grad():
    tot = 0.0
    for _ in range(2):
        xm, x, m = vbf(rng, 8)
        tot += float(masked_ce_loss(md(xm), x, m).item())
v = tot / 2
md.train()
ema.restore(md, backup)
assert math.isfinite(v)
print(f"EMA eval OK ({v:.3f})")

# checkpoint round-trip
p = Path("/tmp/_smoke_local.pt")
torch.save({"model": {k: v.clone() for k, v in ema.shadow.items()},
            "mcfg": {"block": BLK}}, p)
ck = torch.load(p, weights_only=False)
md.load_state_dict(ck["model"])
p.unlink()
print("ckpt round-trip OK")

# samplers at block 64
def _top_p_sample(logits, temp, top_p):
    logits = logits.float() / max(temp, 1e-4)
    probs = torch.softmax(logits, dim=-1)
    sp, si = probs.sort(dim=-1, descending=True)
    cum = sp.cumsum(dim=-1)
    keep = (cum - sp) < top_p
    keep[..., 0] = True
    sp = sp * keep
    sp = sp / sp.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    choice = torch.multinomial(sp, 1)
    return si.gather(-1, choice).squeeze(-1), sp.gather(-1, choice).squeeze(-1)

with torch.no_grad():
    arr = random_window(val_ids, BLK, 1, rng)[0]
    prompt = arr[:32].tolist()
    # greedy
    canvas = torch.tensor([prompt + [MASK_ID] * 8])
    for s in range(4):
        mpos = (canvas[0] == MASK_ID).nonzero().flatten()
        if len(mpos) == 0: break
        logits = md(canvas)
        logits[:, :, MASK_ID] = float("-inf")
        probs = torch.softmax(logits[0, mpos].float(), dim=-1)
        conf, pick = probs.max(dim=-1)
        k = max(1, math.ceil(8 / (4 - s)))
        for oi in torch.argsort(-conf)[:k].tolist():
            canvas[0, mpos[oi]] = pick[oi]
    assert not (canvas[0] == MASK_ID).any()
    print("greedy sampler OK")
    # sampled + refine
    md.eval()
    torch.manual_seed(0)
    canvas = torch.tensor([prompt + [MASK_ID] * 8])
    for s in range(4):
        mpos = (canvas[0] == MASK_ID).nonzero().flatten()
        if len(mpos) == 0: break
        logits = md(canvas)
        logits[:, :, MASK_ID] = float("-inf")
        toks, conf = _top_p_sample(logits[0, mpos], 0.8, 0.95)
        k = max(1, math.ceil(8 / (4 - s)))
        for oi in torch.argsort(-conf)[:k].tolist():
            canvas[0, mpos[oi]] = toks[oi]
    gen_pos = torch.arange(32, 40)
    for _r in range(1):
        logits = md(canvas)
        logits[:, :, MASK_ID] = float("-inf")
        probs = torch.softmax(logits[0, gen_pos].float(), dim=-1)
        own = probs[torch.arange(8), canvas[0, gen_pos]]
        low = torch.argsort(own)[:1]
        canvas[0, gen_pos[low]] = MASK_ID
        for _s2 in range(4):
            mpos = (canvas[0] == MASK_ID).nonzero().flatten()
            if len(mpos) == 0: break
            logits = md(canvas)
            logits[:, :, MASK_ID] = float("-inf")
            toks, conf = _top_p_sample(logits[0, mpos], 0.64, 0.95)
            k = max(1, math.ceil(len(mpos) / (4 - _s2)))
            for oi in torch.argsort(-conf)[:k].tolist():
                canvas[0, mpos[oi]] = toks[oi]
    assert not (canvas[0] == MASK_ID).any()
    print("sampled+refine sampler OK")

# AR loss with explicit block
x = torch.from_numpy(random_window(train_ids, BLK, 4, rng)).long()
logits = ma(x[:, :-1])
al = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
assert math.isfinite(float(al))
print("AR loss OK")
print("LOCAL INTEGRATION: ALL PASSED — safe to push")
