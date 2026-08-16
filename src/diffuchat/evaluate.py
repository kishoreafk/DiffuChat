"""DiffuChat evaluation: the 3 experiments (spec §5.7).

(1) Masked-span reconstruction: mask random spans (len in 1/2/4/8/16) of held-
    out val text; exact token recovery for the diffusion model vs random-guess,
    frequency (most-common-char) and matched-size AR char-LM baselines.
    This is the headline scientific comparison: bidirectional diffusion vs
    causal AR on infilling (the AR model sees only the prefix).
(2) Decoding ablation: recovery vs #denoise steps (5/10/20/40) x strategy
    (confidence / random / semi_ar).
(3) Edit conditioning: pin a token inside a masked span; assert 100% pin
    survival; compare teacher-forced CE of the edited canvas vs unedited.

CLI:  python -m diffuchat.evaluate [--ckpt ...] [--ar-ckpt ...] [--out ...]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .ar_baseline import greedy_infill, load_ar, train_ar
from .config import GenConfig
from .data import CharTokenizer, encode_corpus, shakespeare, split_ids
from .generate import Canvas, edit, generate
from .train import DEFAULT_OUT as DIFFUSION_CKPT, load_model

_REPO = Path(__file__).resolve().parents[2]
AR_CKPT = _REPO / "artifacts" / "ar_baseline.pt"
DEFAULT_OUT = _REPO / "artifacts" / "eval_report.json"


def _sample_windows(
    val_ids: np.ndarray, block: int, n: int, rng: np.random.Generator
) -> list[np.ndarray]:
    starts = rng.integers(0, len(val_ids) - block - 1, size=n)
    return [val_ids[s : s + block] for s in starts]


def diffusion_infill(
    model, tok: CharTokenizer, window: np.ndarray, span_start: int, span_len: int,
    cfg: GenConfig,
) -> list[int]:
    """Re-denoise exactly the masked span, conditioned on BOTH sides."""
    ids = [int(t) for t in window]
    canvas = Canvas(
        ids=ids[:span_start] + [tok.mask_token_id] * span_len + ids[span_start + span_len :],
        answer_start=span_start,
        mask_id=tok.mask_token_id,
        locked=[False] * len(ids),
    )
    # suffix context is inside the canvas but NOT masked -> never touched
    res = generate(model, tok, "", cfg, canvas=canvas)
    return [canvas.ids[span_start + j] for j in range(span_len)]


def experiment_1(model, tok, ar_model, val_ids, rng, n_windows=40, steps=10):
    """Masked-span reconstruction vs three baselines (spec §5.7.1)."""
    spans = [1, 2, 4, 8, 16]
    windows = _sample_windows(val_ids, model.cfg.block_size, n_windows * len(spans), rng)
    vocab_no_mask = tok.mask_token_id  # real chars: 0..64 -> guess rate 1/65
    most_common = int(np.bincount(val_ids).argmax())

    acc = {s: {"diffusion": [], "freq": [], "ar": []} for s in spans}
    exact = {s: {"diffusion": [], "freq": [], "ar": []} for s in spans}
    i = 0
    for s in spans:
        for _ in range(n_windows):
            w = windows[i]
            i += 1
            span_start = int(rng.integers(model.cfg.block_size // 4,
                                          model.cfg.block_size - s - 1))
            truth = [int(t) for t in w[span_start : span_start + s]]
            cfg = GenConfig(strategy="confidence", steps=steps, answer_len=s,
                            temperature=0.0, seed=42)
            pred = diffusion_infill(model, tok, w, span_start, s, cfg)
            ar_pred = greedy_infill(ar_model, [int(t) for t in w[:span_start]], s)
            freq_pred = [most_common] * s
            for name, p in (("diffusion", pred), ("ar", ar_pred), ("freq", freq_pred)):
                acc[s][name].append(sum(a == b for a, b in zip(p, truth)) / s)
                exact[s][name].append(float(p == truth))
    return {
        "n_windows": n_windows,
        "steps": steps,
        "random_guess_rate": 1.0 / vocab_no_mask,
        "most_common_char": tok.decode([most_common]),
        "token_recovery": {str(s): {k: float(np.mean(v)) for k, v in d.items()}
                           for s, d in acc.items()},
        "full_span_exact": {str(s): {k: float(np.mean(v)) for k, v in d.items()}
                            for s, d in exact.items()},
    }


def experiment_2(model, tok, val_ids, rng, n_windows=24, span_len=8):
    """Decoding ablation: steps x strategy (spec §5.7.2)."""
    step_grid = [5, 10, 20, 40]
    strategies = ["confidence", "random", "semi_ar"]
    windows = _sample_windows(val_ids, model.cfg.block_size, n_windows, rng)
    table = {}
    for strat in strategies:
        for steps in step_grid:
            rec = []
            for w in windows:
                span_start = int(rng.integers(model.cfg.block_size // 4,
                                              model.cfg.block_size - span_len - 1))
                truth = [int(t) for t in w[span_start : span_start + span_len]]
                cfg = GenConfig(strategy=strat, steps=steps, answer_len=span_len,
                                semi_ar_block=max(2, span_len // 2),
                                temperature=0.0, seed=42)
                pred = diffusion_infill(model, tok, w, span_start, span_len, cfg)
                rec.append(sum(a == b for a, b in zip(pred, truth)) / span_len)
            table[f"{strat}|steps={steps}"] = float(np.mean(rec))
    return {"n_windows": n_windows, "span_len": span_len,
            "token_recovery": table}


def experiment_3(model, tok, val_ids, rng, n_windows=30, span_len=8, steps=10):
    """Edit conditioning: pin survival (must be 100%) + CE comparison."""
    windows = _sample_windows(val_ids, model.cfg.block_size, n_windows, rng)
    device = next(model.parameters()).device
    ces_edited, ces_unedited, rec_edited, rec_unedited = [], [], [], []
    pin_survivals = []

    for w in windows:
        span_start = int(rng.integers(model.cfg.block_size // 4,
                                      model.cfg.block_size - span_len - 1))
        truth = [int(t) for t in w[span_start : span_start + span_len]]
        pin_rel = span_len // 2
        pin_abs = span_start + pin_rel
        pin_id = truth[pin_rel]

        def make_canvas() -> Canvas:
            ids = [int(t) for t in w]
            return Canvas(
                ids=ids[:span_start] + [tok.mask_token_id] * span_len
                + ids[span_start + span_len :],
                answer_start=span_start,
                mask_id=tok.mask_token_id,
                locked=[False] * len(ids),
            )

        # ---- unedited: plain span denoise ------------------------------
        cfg = GenConfig(strategy="confidence", steps=steps, answer_len=span_len,
                        temperature=0.0, seed=42)
        c0 = make_canvas()
        generate(model, tok, "", cfg, canvas=c0)
        # ---- edited: pin the TRUE middle token, lock the suffix --------
        c1 = make_canvas()
        for p in range(span_start + span_len, len(c1.ids)):
            c1.locked[p] = True  # suffix context must survive edit() re-mask
        edit(model, tok, c1, [(pin_abs, pin_id)], cfg)
        pin_survivals.append(c1.ids[pin_abs] == pin_id)

        # ---- teacher-forced CE of the final canvases (vs truth) --------
        def span_ce(canvas: Canvas) -> float:
            with torch.no_grad():
                x = torch.tensor([canvas.ids], dtype=torch.long, device=device)
                logits = model(x)[0]
                tgt = torch.tensor(truth, dtype=torch.long, device=device)
                rows = logits[span_start : span_start + span_len]
                return float(F.cross_entropy(rows, tgt))

        ces_unedited.append(span_ce(c0))
        ces_edited.append(span_ce(c1))
        rec_unedited.append(
            sum(c0.ids[span_start + j] == truth[j] for j in range(span_len)) / span_len
        )
        rec_edited.append(
            sum(c1.ids[span_start + j] == truth[j] for j in range(span_len)) / span_len
        )

    survival = float(np.mean(pin_survivals))
    assert survival == 1.0, f"pin survival {survival} != 1.0 — §7.4 invariant broken"
    return {
        "n_windows": n_windows,
        "span_len": span_len,
        "pin_survival_rate": survival,
        "mean_ce_unedited": float(np.mean(ces_unedited)),
        "mean_ce_edited": float(np.mean(ces_edited)),
        "token_recovery_unedited": float(np.mean(rec_unedited)),
        "token_recovery_edited": float(np.mean(rec_edited)),
        "ce_improved_fraction": float(np.mean(
            [e <= u + 1e-6 for e, u in zip(ces_edited, ces_unedited)]
        )),
    }


def _md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def to_markdown(report: dict) -> str:
    lines = ["# DiffuChat evaluation report", ""]
    lines += [f"- diffusion checkpoint: `{report['checkpoints']['diffusion']}`"
              f" (val loss {report['diffusion_val_loss']:.4f})",
              f"- AR baseline: `{report['checkpoints']['ar']}`"
              f" (val loss {report['ar_val_loss']:.4f})", ""]

    e1 = report["experiment_1"]
    lines += ["## Experiment 1 — masked-span reconstruction (token recovery)", ""]
    rows = []
    for s in ["1", "2", "4", "8", "16"]:
        d = e1["token_recovery"][s]
        x = e1["full_span_exact"][s]
        rows.append([s, f"{d['diffusion']*100:.1f}%", f"{d['ar']*100:.1f}%",
                     f"{d['freq']*100:.1f}%",
                     f"{e1['random_guess_rate']*100:.2f}%",
                     f"{x['diffusion']*100:.0f}%"])
    lines += [_md_table(
        ["span len", "diffusion", "AR-LM (prefix-only)", "frequency",
         "random guess", "full-span exact (diff.)"], rows), ""]
    lines += ["> The AR baseline cannot see the text after the span (causal);"
              " the diffusion model denoises conditioned on both sides.", ""]

    e2 = report["experiment_2"]
    lines += ["## Experiment 2 — decoding ablation (token recovery, span 8)", ""]
    rows = []
    for strat in ["confidence", "random", "semi_ar"]:
        rows.append([strat] + [
            f"{e2['token_recovery'][f'{strat}|steps={st}']*100:.1f}%"
            for st in [5, 10, 20, 40]])
    lines += [_md_table(["strategy", "steps=5", "steps=10", "steps=20", "steps=40"],
                        rows), ""]

    e3 = report["experiment_3"]
    lines += ["## Experiment 3 — edit conditioning", ""]
    lines += [_md_table(
        ["metric", "unedited", "edited (1 true pin)"],
        [["pin survival", "—", f"{e3['pin_survival_rate']*100:.0f}%"],
         ["mean span CE (vs truth)", f"{e3['mean_ce_unedited']:.4f}",
          f"{e3['mean_ce_edited']:.4f}"],
         ["token recovery", f"{e3['token_recovery_unedited']*100:.1f}%",
          f"{e3['token_recovery_edited']*100:.1f}%"],
         ["CE improved vs unedited", "—",
          f"{e3['ce_improved_fraction']*100:.0f}% of windows"]])]
    return "\n".join(lines) + "\n"


def evaluate(
    ckpt: str | Path = DIFFUSION_CKPT,
    ar_ckpt: str | Path = AR_CKPT,
    out: str | Path = DEFAULT_OUT,
    seed: int = 42,
) -> dict:
    """Run all three experiments; write JSON + markdown reports."""
    torch.set_num_threads(4)
    rng = np.random.default_rng(seed)
    model, tok, ck = load_model(ckpt)
    print(f"[eval] diffusion: val loss {ck['val_loss']:.4f}")

    if not Path(ar_ckpt).exists():
        print("[eval] AR baseline missing — training it now (matched config)")
        from .config import DenoiseLMConfig, TrainConfig
        mcfg = DenoiseLMConfig(**{k: v for k, v in ck["cfg"].items()})
        steps = int(ck.get("tcfg", {}).get("steps", 1250))
        bs = int(ck.get("tcfg", {}).get("batch_size", 48))
        lr = float(ck.get("tcfg", {}).get("lr", 1e-3))
        train_ar(mcfg, TrainConfig(steps=steps, batch_size=bs, lr=lr), ar_ckpt)
    ar_model, ar_tok, ar_ck = load_ar(ar_ckpt)
    print(f"[eval] AR baseline: val loss {ar_ck['val_loss']:.4f}")

    text = shakespeare()
    ids = encode_corpus(tok, text)
    _, val_ids = split_ids(ids, 0.1)

    t0 = time.time()
    report: dict = {
        "checkpoints": {"diffusion": str(ckpt), "ar": str(ar_ckpt)},
        "diffusion_val_loss": float(ck["val_loss"]),
        "ar_val_loss": float(ar_ck["val_loss"]),
    }
    print("[eval] experiment 1: masked-span reconstruction ...")
    report["experiment_1"] = experiment_1(model, tok, ar_model, val_ids, rng)
    print("[eval] experiment 2: decoding ablation ...")
    report["experiment_2"] = experiment_2(model, tok, val_ids, rng)
    print("[eval] experiment 3: edit conditioning ...")
    report["experiment_3"] = experiment_3(model, tok, val_ids, rng)
    report["wall_time_s"] = round(time.time() - t0, 1)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path = out.with_suffix(".md")
    md_path.write_text(to_markdown(report), encoding="utf-8")
    print(f"[eval] wrote {out} and {md_path} ({report['wall_time_s']}s)")
    print(to_markdown(report))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the 3 DiffuChat experiments")
    ap.add_argument("--ckpt", type=Path, default=DIFFUSION_CKPT)
    ap.add_argument("--ar-ckpt", type=Path, default=AR_CKPT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    evaluate(args.ckpt, args.ar_ckpt, args.out)


if __name__ == "__main__":
    main()
