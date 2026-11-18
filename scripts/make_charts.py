#!/usr/bin/env python3
"""Generate report charts from DiffuChat Kaggle run metrics.json.

Outputs PNGs to  /home/z/my-project/download/diffuchat_artifacts/charts/
Palette: DM-1 (Deep Cyan) — AI/tech report palette from the docx design system.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

FONT = "/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf"
try:
    fm.fontManager.addfont(FONT)
except Exception:
    pass
plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ACCENT = "#1B6B7A"      # DM-1 table accent (readable on white)
ACCENT2 = "#D4875A"     # contrast series (AR)
GRID = "#E0E0E0"
TEXT = "#333333"


def line_chart(x, series, title, xlabel, ylabel, output, figsize=(10, 5.2)):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    for i, (name, ys) in enumerate(series.items()):
        ax.plot(x, ys, marker="o", ms=4, lw=2,
                color=[ACCENT, ACCENT2, "#5B8DB8"][i % 3], label=name)
    ax.set_title(title, fontsize=14, pad=12, color=TEXT)
    ax.set_xlabel(xlabel, fontsize=11, color=TEXT)
    ax.set_ylabel(ylabel, fontsize=11, color=TEXT)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3, color=GRID)
    ax.legend(frameon=False, fontsize=10, loc="best")
    fig.savefig(output, dpi=200, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"saved {output}")


def grouped_bar(categories, groups, title, ylabel, output, figsize=(10, 5.2)):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    x = np.arange(len(categories))
    n = len(groups)
    width = 0.8 / n
    for i, (name, vals) in enumerate(groups.items()):
        offset = (i - n / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=name,
                      color=[ACCENT, ACCENT2, "#5B8DB8"][i % 3], edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}" if isinstance(val, float) else str(val),
                    ha="center", va="bottom", fontsize=9, color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_title(title, fontsize=14, pad=12, color=TEXT)
    ax.set_ylabel(ylabel, fontsize=11, color=TEXT)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, color=GRID)
    ax.legend(frameon=False, fontsize=10)
    fig.savefig(output, dpi=200, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"saved {output}")


def main(metrics_path: str, outdir: str) -> None:
    m = json.loads(Path(metrics_path).read_text())
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. validation loss curves (both models)
    dh = m.get("diffusion", {}).get("history", [])
    ah = m.get("ar", {}).get("history", [])
    if dh:
        x = [p["step"] for p in dh]
        series = {"DenoiseLM (masked CE on val)": [p["val_loss"] for p in dh]}
        if ah:
            series["ARBaseline (next-token CE on val)"] = [p["val_loss"] for p in ah]
        line_chart(x, series, "Validation loss during training (WikiText-103)",
                   "Training step", "Validation cross-entropy (nats/token)",
                   out / "loss_curves.png")

    # 2. train vs val for the diffusion model (improvement check)
    if dh:
        line_chart([p["step"] for p in dh],
                   {"train loss": [p["train_loss"] for p in dh],
                    "val loss": [p["val_loss"] for p in dh]},
                   "DenoiseLM: training vs validation loss (improvement check)",
                   "Training step", "Cross-entropy (nats/token)",
                   out / "diffusion_train_val.png")

    # 3. infilling comparison
    infill = m.get("infill_comparison", {})
    if infill:
        spans = sorted(infill.keys(), key=int)
        grouped_bar(spans,
                    {"DenoiseLM (bidirectional)": [infill[s]["diffusion_token_recovery"] for s in spans],
                     "AR baseline (prefix-only)": [infill[s]["ar_token_recovery"] for s in spans]},
                    "Masked-span infilling on validation windows",
                    "Token recovery rate", out / "infill_comparison.png")

    # 4. decode efficiency
    speed = m.get("speed_bench", {})
    if speed.get("diffusion"):
        grouped_bar(["Forward passes", "Wall time (s)"],
                    {"DenoiseLM": [speed["diffusion"]["forward_passes"],
                                   speed["diffusion"]["time_s"]],
                     "AR greedy": [speed["ar"]["forward_passes"], speed["ar"]["time_s"]]},
                    "Decoding 64 tokens: forward passes and wall time",
                    "Count (passes / seconds)", out / "decode_efficiency.png")


def compare_runs(m1_path: str, m2_path: str, outdir: str) -> None:
    """v1 baseline vs v2 improved — DenoiseLM val loss + recovery."""
    m1 = json.loads(Path(m1_path).read_text())
    m2 = json.loads(Path(m2_path).read_text())
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    h1 = m1["diffusion"]["history"]
    h2 = m2["diffusion"]["history"]
    fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    ax.plot([p["step"] for p in h1], [p["val_loss"] for p in h1], marker="o", ms=4,
            lw=2, color=ACCENT2, label="v1 baseline (2.5k steps, batch 16)")
    ax.plot([p["step"] for p in h2], [p["val_loss"] for p in h2], marker="o", ms=4,
            lw=2, color=ACCENT, label="v2 improved (12k steps, batch 32)")
    ax.set_title("DenoiseLM validation loss: baseline vs improved schedule",
                 fontsize=14, pad=12, color=TEXT)
    ax.set_xlabel("Training step", fontsize=11, color=TEXT)
    ax.set_ylabel("Validation masked cross-entropy", fontsize=11, color=TEXT)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3, color=GRID)
    ax.legend(frameon=False, fontsize=10)
    fig.savefig(out / "diffusion_v1_vs_v2.png", dpi=200, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"saved {out / 'diffusion_v1_vs_v2.png'}")

    # AR comparison
    a1 = m1["ar"]["history"]
    a2 = m2["ar"]["history"]
    fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    ax.plot([p["step"] for p in a1], [p["val_loss"] for p in a1], marker="o", ms=4,
            lw=2, color=ACCENT2, label="v1 baseline (2.5k steps, batch 16)")
    ax.plot([p["step"] for p in a2], [p["val_loss"] for p in a2], marker="o", ms=4,
            lw=2, color=ACCENT, label="v2 improved (12k steps, batch 32)")
    ax.set_title("ARBaseline validation loss: baseline vs improved schedule",
                 fontsize=14, pad=12, color=TEXT)
    ax.set_xlabel("Training step", fontsize=11, color=TEXT)
    ax.set_ylabel("Validation next-token cross-entropy", fontsize=11, color=TEXT)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3, color=GRID)
    ax.legend(frameon=False, fontsize=10)
    fig.savefig(out / "ar_v1_vs_v2.png", dpi=200, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"saved {out / 'ar_v1_vs_v2.png'}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
