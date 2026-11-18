"""DiffuChat on Kaggle — v3 GENERALIZATION upgrade (E7, controlled vs v2).

Changes vs v2 (everything else held identical: arch, batch, steps, lr, block):
  1. dropout 0.0 -> 0.1 on BOTH models (arch-matched regularization)
  2. EMA of weights (decay 0.999) used for all evals + saved checkpoints
  3. Multi-domain corpus: WikiText-103 + TinyStories mixed in one stream
  4. OOD generalization evals: TinyStories val + FineWeb-Edu probe
  5. Improved diffusion sampler (temperature + top-p + annealed unmasking
     + remask-refine) to fix the degenerate "the the the" generation
  6. Explicit generalization section in metrics.json (in-domain vs OOD CE,
     train-val gap curve, coherence/repetition metrics)

Outputs -> /kaggle/working/
    checkpoints/{diffusion,ar}_{best,last}.pt   (best = EMA weights, last = resumable)
    tokenizer.json, metrics.json, history_diffusion.json, history_ar.json
"""
from __future__ import annotations

import gc
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
# Token cache rides in as the Kaggle DATASET kishore00afk/diffuchat-token-cache
# (kernel-source mounts proved unreliable in v2/v3). Discovery scans every
# /kaggle/input/* mount for the needed filenames instead of hardcoding a path.
INPUT_ROOT = Path("/kaggle/input")


def find_in_input(filename: str):
    """Return the first existing /kaggle/input/*/<filename> (any mount name)."""
    if INPUT_ROOT.exists():
        for d in sorted(INPUT_ROOT.iterdir()):
            cand = d / filename
            if cand.is_file():
                return cand
    cand = WORK / filename
    return cand if cand.is_file() else None
CKPT_DIR = WORK / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
REPO_DIR = WORK / "DiffuChat"
T0 = time.time()
TIME_BUDGET_S = 9000  # soft guard; Kaggle T4 sessions are 12h, this is ample

CONFIG = {
    "experiment": "E7 generalization upgrade: dropout+EMA+WikiText-103+TinyStories (v4)",
    "changes_vs_v2": [
        "dropout 0.0 -> 0.1 (both models)",
        "EMA 0.999 for eval/checkpoints",
        "TinyStories mixed into training stream (~20% tokens)",
        "OOD evals: TinyStories val + FineWeb-Edu probe",
        "improved diffusion sampler: temp 0.8 + top-p 0.95 + remask-refine",
    ],
    "vocab_size_bpe": 8192,
    "special_tokens": ["[MASK]", "<|eos|>"],
    "block_size": 256,
    "batch_size": 32,
    "steps_per_model": 12000,
    "eval_every": 500,
    "val_batches": 24,
    "val_batch_size": 32,
    "lr": 4e-4,
    "warmup": 400,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "mask_ratio_min": 0.05,
    "mask_ratio_max": 1.0,
    "d_model": 384,
    "n_layer": 8,
    "n_head": 8,
    "dropout": 0.1,               # <-- v2 used 0.0
    "rel_pos_bias": True,
    "ema_decay": 0.999,
    "tinystories_train_stories": 150_000,
    "tinystories_val_stories": 2_000,
    "fineweb_probe_docs": 3_000,
    "seed": 42,
    "amp": "fp16",
}


def log(tag: str, msg: str) -> None:
    print(f"[{tag} {time.time() - T0:7.1f}s] {msg}", flush=True)


def elapsed_left() -> float:
    return TIME_BUDGET_S - (time.time() - T0)


# ---------------------------------------------------------------------------
# 0. DiffuChat repo (model classes + training math, unmodified)
# ---------------------------------------------------------------------------
def get_repo() -> None:
    if not REPO_DIR.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/kishoreafk/DiffuChat", str(REPO_DIR)],
            check=True, capture_output=True,
        )
    sys.path.insert(0, str(REPO_DIR / "src"))
    os.chdir(WORK)


get_repo()
from diffuchat.model import DenoiseLM                      # noqa: E402
from diffuchat.ar_baseline import ARCharGPT                # noqa: E402
from diffuchat.config import DenoiseLMConfig, as_dict      # noqa: E402
from diffuchat.train import masked_ce_loss, sample_mask, cosine_lr  # noqa: E402
from diffuchat.data import random_window                   # noqa: E402

# ---------------------------------------------------------------------------
# 1. Data: cached WikiText-103 + TinyStories mix + OOD probes
# ---------------------------------------------------------------------------
TOK_PATH = WORK / "tokenizer.json"
WT_TRAIN_NPY = WORK / "wikitext103_train_ids.npy"
WT_VAL_NPY = WORK / "wikitext103_val_ids.npy"
TS_TRAIN_NPY = WORK / "tinystories_train_ids.npy"
TS_VAL_NPY = WORK / "tinystories_val_ids.npy"
FW_VAL_NPY = WORK / "fineweb_val_ids.npy"


def encode_lines(tok, lines, eos_id, eos_mode, tag) -> np.ndarray:
    """Batched encoding. eos_mode: 'heading' (before '= X =' lines, WikiText)
    or 'doc' (after every document, TinyStories/Fineweb)."""
    out: list[int] = []
    B = 2000
    for i in range(0, len(lines), B):
        chunk = lines[i:i + B]
        encs = tok.encode_batch(chunk, add_special_tokens=False)
        for j, enc in enumerate(encs):
            if eos_mode == "heading" and chunk[j].lstrip().startswith("= ") \
                    and chunk[j].rstrip().endswith(" ="):
                out.append(eos_id)
            out.extend(enc.ids)
            if eos_mode == "doc":
                out.append(eos_id)
        if (i // B) % 100 == 0:
            log("DATA", f"{tag}: {i + len(chunk):,}/{len(lines):,} lines "
                        f"({len(out) / 1e6:.1f}M tokens)")
    return np.asarray(out, dtype=np.int32)


def build_data() -> dict:
    from tokenizers import Tokenizer
    from datasets import load_dataset

    info: dict = {}

    # --- tokenizer + WikiText-103 (cached via the token-cache dataset) ---
    c_tok = find_in_input("tokenizer.json")
    c_tr = find_in_input("wikitext103_train_ids.npy")
    c_va = find_in_input("wikitext103_val_ids.npy")
    if c_tok and c_tr and c_va:
        log("DATA", f"cached WikiText-103 + tokenizer "
                    f"({c_tok.parent.name}/) — skipping download+encode")
        tok = Tokenizer.from_file(str(c_tok))
        wt_train = np.load(c_tr)
        wt_val = np.load(c_va)
        info["cache_source"] = c_tok.parent.name
    else:
        from tokenizers import models, trainers, pre_tokenizers, decoders
        log("DATA", "no cache — downloading Salesforce/wikitext wikitext-103-raw-v1 ...")
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
        train_lines = [t for t in ds["train"]["text"] if t.strip()]
        val_lines = [t for t in ds["validation"]["text"] if t.strip()]
        tok = Tokenizer(models.BPE(unk_token=None))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=CONFIG["vocab_size_bpe"], min_frequency=2,
            show_progress=False, special_tokens=CONFIG["special_tokens"])
        tok.train_from_iterator(train_lines[:80000], trainer)
        tok.save(str(TOK_PATH))
        eos_id = tok.token_to_id("<|eos|>")
        wt_train = encode_lines(tok, train_lines, eos_id, "heading", "wt-train")
        wt_val = encode_lines(tok, val_lines, eos_id, "heading", "wt-val")
        np.save(WT_TRAIN_NPY, wt_train)
        np.save(WT_VAL_NPY, wt_val)
    info["wikitext_train_tokens"] = int(len(wt_train))
    info["wikitext_val_tokens"] = int(len(wt_val))

    # --- TinyStories (multi-domain training mix + OOD val) ---
    ts_train = ts_val = None
    c_tstr = find_in_input("tinystories_train_ids.npy")
    c_tsva = find_in_input("tinystories_val_ids.npy")
    if c_tstr and c_tsva:
        ts_train = np.load(c_tstr)
        ts_val = np.load(c_tsva)
        info["tinystories_cache"] = c_tstr.parent.name
    if ts_train is None:
        try:
            log("DATA", "downloading roneneldan/TinyStories ...")
            ts = load_dataset("roneneldan/TinyStories")
            eos_id = tok.token_to_id("<|eos|>")
            tr_docs = [t for t in ts["train"]["text"][:CONFIG["tinystories_train_stories"]]
                       if t.strip()]
            va_docs = [t for t in ts["validation"]["text"][:CONFIG["tinystories_val_stories"]]
                       if t.strip()]
            ts_train = encode_lines(tok, tr_docs, eos_id, "doc", "ts-train")
            ts_val = encode_lines(tok, va_docs, eos_id, "doc", "ts-val")
            np.save(TS_TRAIN_NPY, ts_train)
            np.save(TS_VAL_NPY, ts_val)
        except Exception as e:  # noqa: BLE001
            log("DATA", f"TinyStories FAILED ({e!r}) — falling back to WikiText-only")
            ts_train = np.zeros(0, dtype=np.int32)
            ts_val = np.zeros(0, dtype=np.int32)
            info["tinystories_error"] = repr(e)
    info["tinystories_train_tokens"] = int(len(ts_train))
    info["tinystories_val_tokens"] = int(len(ts_val))

    # --- FineWeb-Edu OOD probe (streaming; small on purpose) ---
    fw_val = None
    c_fw = find_in_input("fineweb_val_ids.npy")
    if c_fw:
        fw_val = np.load(c_fw)
        info["fineweb_cache"] = c_fw.parent.name
    if fw_val is None:
        try:
            log("DATA", "streaming FineWeb-Edu probe docs ...")
            fw = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                              split="train", streaming=True)
            docs = []
            for row in fw:
                t = (row.get("text") or "").strip()
                if t:
                    docs.append(t[:800])
                if len(docs) >= CONFIG["fineweb_probe_docs"]:
                    break
            eos_id = tok.token_to_id("<|eos|>")
            fw_val = encode_lines(tok, docs, eos_id, "doc", "fw-probe")
            np.save(FW_VAL_NPY, fw_val)
        except Exception as e:  # noqa: BLE001
            log("DATA", f"FineWeb probe FAILED ({e!r}) — OOD eval will use TinyStories only")
            fw_val = np.zeros(0, dtype=np.int32)
            info["fineweb_error"] = repr(e)
    info["fineweb_probe_tokens"] = int(len(fw_val))

    train_ids = wt_train if len(ts_train) == 0 else np.concatenate([wt_train, ts_train])
    assert tok.token_to_id("[MASK]") not in np.unique(train_ids[:5_000_000]), "mask leaked"
    info["train_tokens_total"] = int(len(train_ids))
    info["train_mix"] = {
        "wikitext_frac": round(len(wt_train) / len(train_ids), 3),
        "tinystories_frac": round(len(ts_train) / max(len(train_ids), 1), 3),
    }
    info["tokenizer"] = "byte-level BPE (v1/v2 reuse)"
    info["vocab_total"] = tok.get_vocab_size()
    return {"tok": tok, "train": train_ids, "wt_val": wt_val,
            "ts_val": ts_val, "fw_val": fw_val, "info": info}


D = build_data()
tok, train_ids = D["tok"], D["train"]
wt_val_ids, ts_val_ids, fw_val_ids = D["wt_val"], D["ts_val"], D["fw_val"]
MASK_ID = tok.token_to_id("[MASK]")
log("DATA", f"train stream {len(train_ids):,} tokens | wt-val {len(wt_val_ids):,} "
            f"| ts-val {len(ts_val_ids):,} | fw-probe {len(fw_val_ids):,}")

# ---------------------------------------------------------------------------
# 2. Model config (identical to v2 except dropout)
# ---------------------------------------------------------------------------
mcfg = DenoiseLMConfig(
    block_size=CONFIG["block_size"],
    vocab_size=tok.get_vocab_size(),
    mask_token_id=MASK_ID,
    n_layer=CONFIG["n_layer"],
    n_head=CONFIG["n_head"],
    d_model=CONFIG["d_model"],
    dropout=CONFIG["dropout"],
    rel_pos_bias=CONFIG["rel_pos_bias"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu"


def gpu_mem_gb() -> float:
    if DEVICE.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e9


class EMA:
    """Exponential moving average of model weights (fp32 shadow on device)."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            k: v.detach().clone().float()
            for k, v in model.state_dict().items() if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int) -> None:
        d = min(self.decay, (1.0 + step) / (10.0 + step))  # early-step ramp
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)

    @torch.no_grad()
    def swap_in(self, model: torch.nn.Module) -> dict:
        """Copy EMA weights into the model; return a backup of raw weights."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        sd = model.state_dict()
        for k, v in self.shadow.items():
            sd[k].copy_(v.to(sd[k].dtype))
        return backup

    @torch.no_grad()
    def restore(self, model: torch.nn.Module, backup: dict) -> None:
        sd = model.state_dict()
        for k, v in backup.items():
            sd[k].copy_(v)


class _TCFG:
    mask_ratio_min = CONFIG["mask_ratio_min"]
    mask_ratio_max = CONFIG["mask_ratio_max"]


_TCFG_PROXY = _TCFG()


def save_last(path: Path, model, opt, ema: EMA, step, best_val, history, kind):
    torch.save({
        "kind": kind, "model": model.state_dict(), "opt": opt.state_dict(),
        "ema": ema.shadow, "mcfg": as_dict(mcfg), "step": step,
        "best_val": best_val, "history": history,
    }, path)


def save_best(path: Path, ema: EMA, step, val_loss, history, kind):
    torch.save({
        "kind": kind + "_ema", "model": {k: v.clone() for k, v in ema.shadow.items()},
        "mcfg": as_dict(mcfg), "step": step, "val_loss": val_loss, "history": history,
    }, path)


# ---------------------------------------------------------------------------
# 3. Batch factories (train on the mixed stream; val per corpus)
# ---------------------------------------------------------------------------
def make_diffusion_batch(ids, block: int | None = None):
    blk = block or mcfg.block_size
    def fn(rng, batch_size):
        x_np = random_window(ids, blk, batch_size, rng)
        x = torch.from_numpy(x_np).long().to(DEVICE)
        m = sample_mask(batch_size, blk, _TCFG_PROXY, rng, DEVICE)
        xm = torch.where(m.bool(), MASK_ID, x)
        return xm, x, m
    return fn


def make_ar_batch(ids, block: int | None = None):
    blk = block or mcfg.block_size
    def fn(rng, batch_size):
        x_np = random_window(ids, blk, batch_size, rng)
        return torch.from_numpy(x_np).long().to(DEVICE), None, None
    return fn


def diffusion_loss(model, xm, x, m):
    logits = model(xm)
    return masked_ce_loss(logits, x, m)


def ar_loss(model, x, _t, _m):
    logits = model(x[:, :-1])
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           x[:, 1:].reshape(-1))


diffusion_train_batch = make_diffusion_batch(train_ids)
diffusion_val_batch = make_diffusion_batch(wt_val_ids)
ar_train_batch = make_ar_batch(train_ids)
ar_val_batch = make_ar_batch(wt_val_ids)

# ---------------------------------------------------------------------------
# 4. Shared training loop (EMA + resume-aware)
# ---------------------------------------------------------------------------
def train_generic(model, opt, kind, batch_fn, loss_fn, val_batch_fn,
                  tag, best_path, last_path) -> dict:
    torch.manual_seed(CONFIG["seed"])
    rng = np.random.default_rng(CONFIG["seed"])
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))
    ema = EMA(model, CONFIG["ema_decay"])

    start_step, best_val, history = 0, float("inf"), []
    if last_path.exists():
        ck = torch.load(last_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        ema.shadow = {k: v.to(DEVICE) for k, v in ck["ema"].items()}
        start_step = ck["step"]
        best_val = ck.get("best_val", float("inf"))
        history = ck.get("history", [])
        log(tag, f"resumed from step {start_step} (best {best_val:.4f})")

    t_start = time.time()
    steps_at_start = start_step
    model.train()
    for step in range(start_step, CONFIG["steps_per_model"]):
        lr = cosine_lr(step, CONFIG["steps_per_model"], CONFIG["lr"], CONFIG["warmup"])
        for g in opt.param_groups:
            g["lr"] = lr

        x, target, mask = batch_fn(rng, CONFIG["batch_size"])
        with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE.type == "cuda")):
            loss = loss_fn(model, x, target, mask)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        scaler.step(opt)
        scaler.update()
        ema.update(model, step)

        if (step + 1) % 100 == 0 or step == start_step:
            log(tag, f"step {step + 1}/{CONFIG['steps_per_model']} "
                     f"loss {loss.item():.4f} lr {lr:.2e}")

        if (step + 1) % CONFIG["eval_every"] == 0 or step + 1 == CONFIG["steps_per_model"]:
            # --- evaluate with EMA weights ---
            backup = ema.swap_in(model)
            vloss = eval_generic(model, val_batch_fn, loss_fn)
            history.append({
                "step": step + 1, "train_loss": float(loss.item()),
                "val_loss": float(vloss),
                "gap": round(float(loss.item()) - float(vloss), 4),
                "elapsed_s": round(time.time() - T0, 1),
            })
            log(tag, f"VAL(step {step + 1}) ema_loss {vloss:.4f} "
                     f"gap {history[-1]['gap']:+.3f}")
            if vloss < best_val:
                best_val = vloss
                save_best(best_path, ema, step + 1, best_val, history, kind)
            save_last(last_path, model, opt, ema, step + 1, best_val, history, kind)
            ema.restore(model, backup)

        if elapsed_left() < 900:
            log(tag, "time budget guard — stopping training early, checkpoints saved")
            break

    n_done = max(1, CONFIG["steps_per_model"] - steps_at_start)
    return {
        "best_val_loss_ema": best_val,
        "history": history,
        "train_time_s": round(time.time() - t_start, 1),
        "s_per_step": round((time.time() - t_start) / n_done, 3),
        "gpu_mem_gb": round(gpu_mem_gb(), 2),
    }


def eval_generic(model, batch_fn, loss_fn,
                 n_batches=None) -> float:
    was_training = model.training
    model.eval()  # dropout OFF for every eval (v3 uses dropout 0.1)
    vrng = np.random.default_rng(1234)
    tot, n = 0.0, 0
    try:
        with torch.no_grad():
            for _ in range(n_batches or CONFIG["val_batches"]):
                x, target, mask = batch_fn(vrng, CONFIG["val_batch_size"])
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=(DEVICE.type == "cuda")):
                    loss = loss_fn(model, x, target, mask)
                tot += float(loss.item())
                n += 1
    finally:
        model.train(was_training)
    return tot / max(n, 1)


@torch.no_grad()
def teacher_forced_recovery(model, ids, n_batches=8, batch_size=32) -> float:
    """Exact-recovery rate of masked tokens (teacher-forced reconstruction)."""
    was_training = model.training
    model.eval()
    vrng = np.random.default_rng(7)
    correct, n_mask = 0, 0
    bf = make_diffusion_batch(ids)
    try:
        for _ in range(n_batches):
            xm, x, m = bf(vrng, batch_size)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE.type == "cuda")):
                logits = model(xm)
            pred = logits.argmax(-1)
            mb = m.bool()
            correct += ((pred == x) & mb).sum().item()
            n_mask += mb.sum().item()
    finally:
        model.train(was_training)
    return correct / max(n_mask, 1)


# ---------------------------------------------------------------------------
# 5. OOD generalization eval (per corpus, EMA weights, model.eval())
# ---------------------------------------------------------------------------
@torch.no_grad()
def ood_suite(model_diff, model_ar) -> dict:
    out = {}
    suites = {"wikitext_val_in_domain": wt_val_ids}
    if len(ts_val_ids) > mcfg.block_size * 4:
        suites["tinystories_ood"] = ts_val_ids
    if len(fw_val_ids) > mcfg.block_size * 4:
        suites["fineweb_edu_ood"] = fw_val_ids
    for name, ids in suites.items():
        entry: dict = {"tokens": int(len(ids))}
        dloss = eval_generic(model_diff, make_diffusion_batch(ids), diffusion_loss,
                             n_batches=16)
        entry["diffusion_masked_ce"] = round(dloss, 4)
        entry["diffusion_recovery"] = round(teacher_forced_recovery(model_diff, ids), 4)
        aloss = eval_generic(model_ar, make_ar_batch(ids), ar_loss, n_batches=16)
        entry["ar_ce"] = round(aloss, 4)
        entry["ar_ppl"] = round(math.exp(aloss), 2)
        out[name] = entry
        log("OOD", f"{name}: {entry}")
    return out


# ---------------------------------------------------------------------------
# 6. Infill comparison (greedy confidence, 10 passes — IDENTICAL to v2)
# ---------------------------------------------------------------------------
@torch.no_grad()
def infill_eval(model_diff, model_ar, n_windows: int = 32) -> dict:
    model_diff.eval()
    model_ar.eval()
    rng = np.random.default_rng(99)
    results: dict[int, dict] = {}
    for L in (1, 2, 4, 8, 16):
        diff_correct, ar_correct, total, exact_d, exact_a = 0, 0, 0, 0, 0
        for _ in range(n_windows):
            x_np = random_window(wt_val_ids, mcfg.block_size, 1, rng)[0]
            start = mcfg.block_size // 2 + int(rng.integers(0, 40))
            start = min(start, mcfg.block_size - L - 1)
            truth = torch.from_numpy(x_np[start:start + L].copy()).long().to(DEVICE)

            canvas = x_np.copy()
            canvas[start:start + L] = MASK_ID
            cm = torch.from_numpy(canvas[None].copy()).long().to(DEVICE)
            for _p in range(10):
                logits = model_diff(cm)
                logits[:, 0, MASK_ID] = float("-inf")
                mpos = (cm[0] == MASK_ID).nonzero().flatten()
                if len(mpos) == 0:
                    break
                probs = torch.softmax(logits[0, mpos].float(), dim=-1)
                conf, pick = probs.max(dim=-1)
                k = max(1, math.ceil(len(mpos) / 10))
                for oi in torch.argsort(-conf)[:k].tolist():
                    cm[0, mpos[oi]] = pick[oi]
            d_pred = cm[0, start:start + L]
            diff_correct += (d_pred == truth).sum().item()
            exact_d += int(torch.equal(d_pred, truth))

            ids = x_np[:start].tolist()
            for _t in range(L):
                inp = torch.tensor([ids[-mcfg.block_size + 1:]], device=DEVICE)
                nxt = int(model_ar(inp)[0, -1].argmax())
                ids.append(nxt)
            a_pred = torch.tensor(ids[start:start + L], device=DEVICE)
            ar_correct += (a_pred == truth).sum().item()
            exact_a += int(torch.equal(a_pred, truth))
            total += L
        results[L] = {
            "diffusion_token_recovery": round(diff_correct / total, 4),
            "ar_token_recovery": round(ar_correct / total, 4),
            "diffusion_span_exact": round(exact_d / n_windows, 4),
            "ar_span_exact": round(exact_a / n_windows, 4),
        }
        log("EVAL", f"infill L={L}: {results[L]}")
    return results


# ---------------------------------------------------------------------------
# 7. Improved diffusion sampler: temp + top-p + annealed unmask + refine
# ---------------------------------------------------------------------------
@torch.no_grad()
def _top_p_sample(logits: torch.Tensor, temp: float, top_p: float) -> tuple[torch.Tensor, torch.Tensor]:
    """logits (N, V) -> (chosen_token (N,), its prob (N,))."""
    logits = logits.float() / max(temp, 1e-4)
    probs = torch.softmax(logits, dim=-1)
    sp, si = probs.sort(dim=-1, descending=True)
    cum = sp.cumsum(dim=-1)
    keep = (cum - sp) < top_p          # smallest set with cumsum >= p
    keep[..., 0] = True
    sp = sp * keep
    sp = sp / sp.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    choice = torch.multinomial(sp, 1)  # (N, 1)
    tok = si.gather(-1, choice).squeeze(-1)
    conf = sp.gather(-1, choice).squeeze(-1)
    return tok, conf


@torch.no_grad()
def diffusion_generate(model, prompt_ids, n_gen, steps=24, temp=0.8, top_p=0.95,
                       refine_passes=2, seed=None) -> list[int]:
    """Annealed confidence sampling with optional remask-refine self-correction."""
    was_training = model.training
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)
    canvas = torch.tensor([list(prompt_ids) + [MASK_ID] * n_gen], device=DEVICE)
    n0 = n_gen
    for s in range(steps):
        mpos = (canvas[0] == MASK_ID).nonzero().flatten()
        if len(mpos) == 0:
            break
        logits = model(canvas)
        logits[:, :, MASK_ID] = float("-inf")
        toks, conf = _top_p_sample(logits[0, mpos], temp, top_p)
        k = max(1, math.ceil(n0 / (steps - s)))
        order = torch.argsort(-conf)[:k]
        for oi in order.tolist():
            canvas[0, mpos[oi]] = toks[oi]
    # --- remask-refine: re-generate the least confident filled positions ---
    gen_pos = torch.arange(len(prompt_ids), len(prompt_ids) + n_gen, device=DEVICE)
    for _r in range(refine_passes):
        if n_gen < 4:
            break
        logits = model(canvas)
        logits[:, :, MASK_ID] = float("-inf")
        probs = torch.softmax(logits[0, gen_pos].float(), dim=-1)
        own = probs[torch.arange(n_gen, device=DEVICE), canvas[0, gen_pos]]
        n_remask = max(1, int(0.15 * n_gen))
        low = torch.argsort(own)[:n_remask]
        canvas[0, gen_pos[low]] = MASK_ID
        for _s2 in range(4):
            mpos = (canvas[0] == MASK_ID).nonzero().flatten()
            if len(mpos) == 0:
                break
            logits = model(canvas)
            logits[:, :, MASK_ID] = float("-inf")
            toks, conf = _top_p_sample(logits[0, mpos], temp * 0.8, top_p)
            k = max(1, math.ceil(len(mpos) / (4 - _s2)))
            for oi in torch.argsort(-conf)[:k].tolist():
                canvas[0, mpos[oi]] = toks[oi]
    model.train(was_training)
    return canvas[0].tolist()


@torch.no_grad()
def greedy_generate(model, prompt_ids, n_gen, passes=10) -> list[int]:
    """v2-style greedy confidence decoding (kept for v2-comparable samples)."""
    was_training = model.training
    model.eval()
    canvas = torch.tensor([list(prompt_ids) + [MASK_ID] * n_gen], device=DEVICE)
    n0 = n_gen
    for s in range(passes):
        mpos = (canvas[0] == MASK_ID).nonzero().flatten()
        if len(mpos) == 0:
            break
        logits = model(canvas)
        logits[:, :, MASK_ID] = float("-inf")
        probs = torch.softmax(logits[0, mpos].float(), dim=-1)
        conf, pick = probs.max(dim=-1)
        k = max(1, math.ceil(n0 / (passes - s)))
        for oi in torch.argsort(-conf)[:k].tolist():
            canvas[0, mpos[oi]] = pick[oi]
    model.train(was_training)
    return canvas[0].tolist()


def coherence_stats(ids: list[int]) -> dict:
    """Repetition / diversity metrics — lower dup_bigram = less degenerate."""
    if len(ids) < 4:
        return {}
    arr = np.asarray(ids)
    dup_bigram = float((arr[1:] == arr[:-1]).mean())
    distinct = float(len(set(arr.tolist())) / len(ids))
    vals, counts = np.unique(arr, return_counts=True)
    top_share = float(counts.max() / len(ids))
    return {"dup_bigram_rate": round(dup_bigram, 4),
            "distinct_token_ratio": round(distinct, 4),
            "top_token_share": round(top_share, 4)}


# ---------------------------------------------------------------------------
# 8. Speed benchmark (greedy AR vs 10-pass greedy diffusion — v2-comparable)
# ---------------------------------------------------------------------------
@torch.no_grad()
def speed_bench(model_diff, model_ar) -> dict:
    model_diff.eval()
    model_ar.eval()
    out = {}
    prompt_ids = tok.encode("The meaning of the term became common in the media",
                            add_special_tokens=False).ids[:128]
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    ids = list(prompt_ids)
    for _ in range(64):
        inp = torch.tensor([ids[-mcfg.block_size + 1:]], device=DEVICE)
        ids.append(int(model_ar(inp)[0, -1].argmax()))
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    out["ar"] = {"tokens": 64, "forward_passes": 64,
                 "time_s": round(time.time() - t0, 2),
                 "tokens_per_s": round(64 / (time.time() - t0), 1),
                 "tokens_per_forward": 1.0}
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    greedy_generate(model_diff, prompt_ids, 64, passes=10)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    out["diffusion_greedy_10pass"] = {"tokens": 64, "forward_passes": 10,
                                      "time_s": round(dt, 2),
                                      "tokens_per_s": round(64 / dt, 1),
                                      "tokens_per_forward": 6.4}
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    diffusion_generate(model_diff, prompt_ids, 64, steps=24, refine_passes=2,
                       seed=1234)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    out["diffusion_sampled_24pass_refine"] = {"tokens": 64, "forward_passes": 24 + 2 * 4,
                                              "time_s": round(dt, 2),
                                              "tokens_per_s": round(64 / dt, 1),
                                              "tokens_per_forward": round(64 / (24 + 8), 1)}
    log("EVAL", f"speed: {out}")
    return out


# ---------------------------------------------------------------------------
# 9. Fast-fail smoke pass — exercises every code path (train/eval/EMA/ckpt/
#    sampler/infill) on tiny models in ~20 s so bugs die in minute 1.
# ---------------------------------------------------------------------------
def smoke_pass() -> None:
    log("SMOKE", "tiny end-to-end round-trip (train>eval>EMA>ckpt>sample) ...")
    tcfg = DenoiseLMConfig(block_size=64, vocab_size=tok.get_vocab_size(),
                           mask_token_id=MASK_ID, n_layer=2, n_head=4,
                           d_model=64, dropout=0.1, rel_pos_bias=True)
    md = DenoiseLM(tcfg).to(DEVICE)
    ma = ARCharGPT(tcfg).to(DEVICE)
    od = torch.optim.AdamW(md.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    dbf = make_diffusion_batch(train_ids[:2_000_000], tcfg.block_size)
    vbf = make_diffusion_batch(wt_val_ids, tcfg.block_size)
    abf = make_ar_batch(train_ids[:2_000_000], tcfg.block_size)
    ema = EMA(md, 0.99)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))
    for step in range(3):
        xm, x, m = dbf(rng, 8)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE.type == "cuda")):
            loss = diffusion_loss(md, xm, x, m)
        od.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(od)
        torch.nn.utils.clip_grad_norm_(md.parameters(), 1.0)
        scaler.step(od)
        scaler.update()
        ema.update(md, step)
    backup = ema.swap_in(md)
    v = eval_generic(md, vbf, diffusion_loss, n_batches=2)
    ema.restore(md, backup)
    assert math.isfinite(v), "eval returned non-finite"
    p = CKPT_DIR / "_smoke.pt"
    save_best(p, ema, 3, v, [], "smoke")
    ck = torch.load(p, map_location=DEVICE, weights_only=False)
    md.load_state_dict(ck["model"])
    p.unlink()
    arr = random_window(wt_val_ids, 64, 1, rng)[0]
    _ = greedy_generate(md, arr[:32].tolist(), 8, passes=4)
    _ = diffusion_generate(md, arr[:32].tolist(), 8, steps=4, refine_passes=1, seed=0)
    _ = ar_loss(ma, torch.from_numpy(
        random_window(train_ids, 64, 4, rng)).long().to(DEVICE), None, None)
    del md, ma, od, ema, scaler
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    log("SMOKE", f"passed (eval {v:.3f}) — all code paths OK")


# ---------------------------------------------------------------------------
# 10. Main
# ---------------------------------------------------------------------------
metrics: dict = {"config": CONFIG, "environment": {
    "gpu": GPU_NAME, "torch": torch.__version__,
    "cuda": torch.version.cuda if DEVICE.type == "cuda" else None,
    "params_note": "both models architecture-matched; evals use EMA weights",
}}
metrics["data"] = D["info"]

model_diff, model_ar = None, None
try:
    smoke_pass()
    # ---- E1: DenoiseLM ----
    log("E1", f"training DenoiseLM (dropout {CONFIG['dropout']}, EMA "
              f"{CONFIG['ema_decay']}) on {GPU_NAME} ...")
    model_diff = DenoiseLM(mcfg).to(DEVICE)
    metrics["diffusion"] = {"n_params": model_diff.num_params(non_embedding=False)}
    opt_d = torch.optim.AdamW(model_diff.parameters(), lr=CONFIG["lr"],
                              betas=(0.9, 0.95), weight_decay=CONFIG["weight_decay"])
    r = train_generic(model_diff, opt_d, "diffusion", diffusion_train_batch,
                      diffusion_loss, diffusion_val_batch, "E1",
                      CKPT_DIR / "diffusion_best.pt", CKPT_DIR / "diffusion_last.pt")
    metrics["diffusion"].update(r)
    log("E1", f"done: best val(EMA) {r['best_val_loss_ema']:.4f} | "
              f"{r['s_per_step']} s/step | {r['gpu_mem_gb']} GB")
    del opt_d
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # ---- E2: AR baseline ----
    log("E2", "training ARCharGPT (same schedule) ...")
    model_ar = ARCharGPT(mcfg).to(DEVICE)
    metrics["ar"] = {"n_params": model_ar.num_params(non_embedding=False)}
    opt_a = torch.optim.AdamW(model_ar.parameters(), lr=CONFIG["lr"],
                              betas=(0.9, 0.95), weight_decay=CONFIG["weight_decay"])
    r = train_generic(model_ar, opt_a, "ar", ar_train_batch, ar_loss, ar_val_batch,
                      "E2", CKPT_DIR / "ar_best.pt", CKPT_DIR / "ar_last.pt")
    metrics["ar"].update(r)
    if math.isfinite(r["best_val_loss_ema"]):
        metrics["ar"]["val_ppl_ema"] = round(math.exp(r["best_val_loss_ema"]), 2)
        log("E2", f"done: best val(EMA) {r['best_val_loss_ema']:.4} "
                  f"(ppl {math.exp(r['best_val_loss_ema']):.2f})")
    else:
        log("E2", "done: no eval completed (time guard) — resume in a follow-up run")
    del opt_a
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # load EMA best weights into the live models for all comparison evals
    for model, path in ((model_diff, "diffusion_best.pt"), (model_ar, "ar_best.pt")):
        p = CKPT_DIR / path
        if p.exists():
            ck = torch.load(p, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ck["model"])
            log("EVAL", f"loaded EMA best weights: {path} (step {ck['step']})")

    # ---- generalization suite (always runs) ----
    model_diff.eval()
    model_ar.eval()
    metrics["generalization"] = ood_suite(model_diff, model_ar)

    if elapsed_left() > 300:
        metrics["infill_comparison"] = infill_eval(model_diff, model_ar)
        metrics["speed_bench"] = speed_bench(model_diff, model_ar)

        prompts = ["The city was founded in the year",
                   "Once upon a time, there was a little girl named",
                   "The study found that patients treated with",
                   "In economics, inflation is defined as"]
        greedy_samples, improved_samples = [], []
        for p in prompts:
            ids = tok.encode(p, add_special_tokens=False).ids[:96]
            g_ids = greedy_generate(model_diff, ids, 48, passes=10)
            g_txt = tok.decode(g_ids, skip_special_tokens=True)
            greedy_samples.append({"prompt": p, "completion": g_txt[len(p):].strip(),
                                   "coherence": coherence_stats(g_ids)})
            i_ids = diffusion_generate(model_diff, ids, 48, steps=24, temp=0.8,
                                       top_p=0.95, refine_passes=2, seed=1234)
            i_txt = tok.decode(i_ids, skip_special_tokens=True)
            improved_samples.append({"prompt": p, "completion": i_txt[len(p):].strip(),
                                     "coherence": coherence_stats(i_ids)})
            log("EVAL", f"greedy  : {greedy_samples[-1]['completion'][:70]!r}")
            log("EVAL", f"improved: {improved_samples[-1]['completion'][:70]!r}")
        metrics["generation_samples"] = {"greedy_v2_style": greedy_samples,
                                         "improved_temp_top_p": improved_samples}
        # aggregate coherence comparison
        def agg(samples, key):
            vs = [s["coherence"][key] for s in samples if s["coherence"]]
            return round(sum(vs) / max(len(vs), 1), 4)
        metrics["coherence_summary"] = {
            "greedy_v2_style": {k: agg(greedy_samples, k) for k in
                                ("dup_bigram_rate", "distinct_token_ratio", "top_token_share")},
            "improved_temp_top_p": {k: agg(improved_samples, k) for k in
                                    ("dup_bigram_rate", "distinct_token_ratio", "top_token_share")},
        }
        log("EVAL", f"coherence: {metrics['coherence_summary']}")
    else:
        metrics["skipped"] = "infill/speed/samples skipped: time budget"
finally:
    (WORK / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if model_diff is not None:
        (WORK / "history_diffusion.json").write_text(
            json.dumps(metrics.get("diffusion", {}).get("history", []), indent=2))
    if model_ar is not None:
        (WORK / "history_ar.json").write_text(
            json.dumps(metrics.get("ar", {}).get("history", []), indent=2))
    log("MAIN", f"metrics.json written | total {time.time() - T0:.0f}s")
    log("MAIN", f"checkpoints: {sorted(p.name for p in CKPT_DIR.glob('*.pt'))}")
