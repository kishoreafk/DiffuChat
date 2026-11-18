#!/usr/bin/env python3
"""Build the single comprehensive DiffuChat deliverable zip.

Layout:
  diffuchat_complete_package/
    README.md          - contents map + login details + headline results
    report/            - findings report PDF, cover HTML, figures
    metrics/           - metrics.json, loss histories, train_v4.log
    weights/           - diffusion_best.pt, ar_best.pt (EMA, step 12000)
    model/             - diffusion_lm.onnx + tokenizer + meta
    code/kernel/       - exact Kaggle v4 train.py + kernel-metadata.json
    code/repo/         - DiffuChat library (DenoiseLM, ARCharGPT, tests, docs)
    web-app/           - FULL Next.js playground (src, prisma, mini-service, configs)
    scripts/           - analysis / report / export scripts
    archive/v1/        - first-run metrics + charts (reference only)
"""
import json
import os
import shutil
import zipfile

PROJ = "/home/z/my-project"
STAGE = "/tmp/zipstage"
PKG = os.path.join(STAGE, "diffuchat_complete_package")
OUT = os.path.join(PROJ, "download", "diffuchat_complete_package.zip")

EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", "node_modules", ".next", ".git"}
EXCLUDE_EXT = {".pyc", ".pyo", ".log"}  # train_v4.log is added explicitly


def copy_tree(src, dst):
    n = 0
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in EXCLUDE_EXT:
                continue
            s = os.path.join(root, f)
            rel = os.path.relpath(s, src)
            d = os.path.join(dst, rel)
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copy2(s, d)
            n += 1
    return n


def copy_file(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def main():
    # ---- fresh staging ----
    if os.path.exists(STAGE):
        shutil.rmtree(STAGE)
    os.makedirs(PKG)

    P = lambda *a: os.path.join(PROJ, *a)
    K = lambda *a: os.path.join(PKG, *a)

    report = [
        ("diffuchat_findings_report.pdf", "report/diffuchat_findings_report.pdf"),
        ("cover.html", "report/cover.html"),
        ("fig_val_curves.png", "report/fig_val_curves.png"),
        ("fig_speed.png", "report/fig_speed.png"),
        ("fig_infill.png", "report/fig_infill.png"),
        ("fig_ood.png", "report/fig_ood.png"),
        ("fig_coherence.png", "report/fig_coherence.png"),
    ]
    metrics = [
        ("metrics.json", "metrics/metrics.json"),
        ("history_diffusion.json", "metrics/history_diffusion.json"),
        ("history_ar.json", "metrics/history_ar.json"),
        ("train_v4.log", "metrics/train_v4.log"),
    ]
    weights = [
        ("diffusion_best.pt", "weights/diffusion_best.pt"),
        ("ar_best.pt", "weights/ar_best.pt"),
    ]
    model = [
        ("diffusion_lm.onnx", "model/diffusion_lm.onnx"),
        ("tokenizer.json", "model/tokenizer.json"),
        ("model_meta.json", "model/model_meta.json"),
        ("metrics.json", "model/metrics.json"),
    ]
    kernel = [
        ("train.py", "code/kernel/train.py"),
        ("kernel-metadata.json", "code/kernel/kernel-metadata.json"),
    ]

    # report / metrics / weights come from the v4 artifacts dir
    for src, dst in report:
        copy_file(P("download", "diffuchat-v4", "report", src), K(dst))
    for src, dst in metrics:
        copy_file(P("download", "diffuchat-v4", "metrics", src), K(dst))
    for src, dst in weights:
        copy_file(P("download", "diffuchat-v4", "weights", src), K(dst))
    # model files live in the workspace model/ dir
    for src, dst in model:
        copy_file(P("model", src), K(dst))
    # kernel files live in work/kg-diffuchat-v3
    for src, dst in kernel:
        copy_file(P("work", "kg-diffuchat-v3", src), K(dst))

    n_repo = copy_tree(P("work", "diffuchat"), K("code", "repo"))
    n_app = copy_tree(P("src"), K("web-app", "src"))
    n_misc_app = 0
    for d in ("public", "prisma", "mini-services", "db"):
        n_misc_app += copy_tree(P(d), K("web-app", d))
    for f in ("package.json", "bun.lock", "next.config.ts", "tailwind.config.ts",
              "postcss.config.mjs", "tsconfig.json", "eslint.config.mjs",
              "components.json", "Caddyfile", ".gitignore"):
        if os.path.exists(P(f)):
            copy_file(P(f), K("web-app", f))
    with open(K("web-app", ".env.example"), "w") as f:
        f.write('DATABASE_URL="file:./custom.db"\n')

    n_scripts = copy_tree(P("scripts"), K("scripts"))

    # archive of the first run (v1) for reference
    for f in ("metrics.json", "history_diffusion.json", "history_ar.json"):
        copy_file(P("download", "diffuchat_artifacts", f), K("archive", "v1", f))
    n_charts = copy_tree(P("download", "diffuchat_artifacts", "charts"), K("archive", "v1", "charts"))

    # ---- README ----
    readme = """# DiffuChat — Complete Deliverable Package

Masked-diffusion language model (**DenoiseLM**) vs architecture-matched autoregressive
baseline (**ARCharGPT**), trained from scratch on Kaggle free-tier Tesla T4 GPUs.
Kernel: `kishore00afk/diffuchat-generalization-v3` (run v4 — E7 generalization upgrade).

## Package contents

| Folder | What it is |
|---|---|
| `report/` | **Findings report** (`diffuchat_findings_report.pdf`) + cover HTML source + chart PNGs |
| `metrics/` | `metrics.json` (every number in the report), both loss histories, full `train_v4.log` |
| `weights/` | `diffusion_best.pt` + `ar_best.pt` — EMA weights at step 12,000 (config + history included) |
| `model/` | Browser-inference export: `diffusion_lm.onnx` (parity 1.8e-05 vs PyTorch), BPE tokenizer, meta |
| `code/kernel/` | The exact `train.py` + `kernel-metadata.json` of the v4 Kaggle run |
| `code/repo/` | Core library: DenoiseLM, ARCharGPT, canvas generation engine, tests, notebooks |
| `web-app/` | **Full Next.js playground**: landing / login / signup / chat with live diffusion animation / report dashboard, ONNX inference mini-service, Prisma schema, all configs |
| `scripts/` | Analysis, chart, report-generation and ONNX-export scripts used to build this package |
| `archive/v1/` | First-run metrics + charts (kept for reference; v4 supersedes them) |

## Web app login details (demo build — no credential checks)

- **Email:** any valid-format email works — e.g. `demo@diffuchat.ai`
- **Password:** anything — e.g. `demo1234`
- **One-click:** the **Google** or **GitHub** buttons sign you straight in
  (as `demo@google.oauth` / `demo@github.oauth`)
- The **Sign up** page likewise accepts any values; sign-in opens the chat
  playground directly.

## Headline results (v4)

- **Decode efficiency**: greedy diffusion resolves a 64-token answer in **10 forward
  passes at 1,523 tok/s** — **6.8x faster** than the AR baseline (64 passes, 223 tok/s)
  on T4. Quality-first sampled decoding (temp 0.8 + top-p 0.95 + remask-refine) still
  runs ~2x faster than AR.
- **Generalization** (first run to measure it): diffusion masked-token recovery 15.5%
  in-domain (WikiText-103), **29.8% on TinyStories**, 7.5% on the unseen FineWeb-Edu
  probe. AR perplexity: 36.1 / 6.6 / 150.2.
- **Infilling**: diffusion matches AR at spans of 1-8 tokens and is **1.9x better at
  L = 16** — the native bidirectional advantage.
- **Sampler upgrade without retraining**: repeated-bigram rate **19.9% -> 3.0%**,
  top-token share 26.8% -> 16.4%.
- **Val losses (EMA)**: diffusion 5.676 (masked CE), AR 3.575 (next-token CE, ppl 35.7).
  Objectives are not directly comparable; the report explains why.

## Running the web app

```bash
cd web-app
bun install                 # or: npm install

# terminal 1 — inference mini-service (loads ../../../model/diffusion_lm.onnx)
cd web-app/mini-services/diffusion-service
DIFFUCHAT_MODEL_DIR=$(pwd)/../../../model bun run index.ts    # port 3030

# terminal 2 — Next.js app
cd web-app
bun run dev                 # port 3000
```

- `web-app/.env.example` -> copy to `.env` (Prisma SQLite; `db/custom.db` is included,
  otherwise run `bun run db:push`).
- The chat view denoises answers token-by-token in front of you; click any finished
  token to pin-edit it — the pin never re-masks and the rest re-denoises around it.
- The Report page renders `metrics.json` live.

## Running inference offline (Python / PyTorch)

```python
import torch, json
from code.repo.src.diffuchat.model import DenoiseLM, DenoiseLMConfig
ckpt = torch.load("weights/diffusion_best.pt", map_location="cpu")
cfg = DenoiseLMConfig(**ckpt["config"])
model = DenoiseLM(cfg); model.load_state_dict(ckpt["model"])
```

(See `code/repo/src/diffuchat/generate.py` for the full canvas / infill engine.)

## Retraining on Kaggle

Attach dataset `kishore00afk/diffuchat-token-cache` (677 MB pre-tokenized
WikiText-103 + TinyStories + FineWeb probe), enable GPU, and run
`code/kernel/train.py` as a script kernel. It smoke-tests the full pipeline in
~20 s (2-layer block-64 miniature) before starting the 2 x 12,000-step run
(~45 min total), resumes from `_last.pt`, and writes `metrics.json` +
checkpoints + histories at the end.

## Environment

Torch 2.11.0+cu130, CUDA 13.0, fp16 autocast, seed 42. Total GPU time: 44.2 min.
Web app: Next.js 16, React 19, Tailwind 4, shadcn/ui, onnxruntime-node.
"""
    with open(K("README.md"), "w") as f:
        f.write(readme)

    # ---- zip ----
    if os.path.exists(OUT):
        os.remove(OUT)
    total = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for root, dirs, files in os.walk(PKG):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for f in sorted(files):
                s = os.path.join(root, f)
                rel = os.path.relpath(s, STAGE)
                z.write(s, rel)
                total += 1
    print(f"zip written: {OUT}")
    print(f"entries: {total} | repo files: {n_repo} | app src: {n_app} | app misc: {n_misc_app} | scripts: {n_scripts} | v1 charts: {n_charts}")
    print(f"zip size: {os.path.getsize(OUT)/1e6:.1f} MB")

    # ---- verify ----
    with zipfile.ZipFile(OUT) as z:
        bad = z.testzip()
        assert bad is None, f"corrupt entry: {bad}"
        names = z.namelist()
        readme_ok = "diffuchat_complete_package/README.md" in names
        need = [
            "diffuchat_complete_package/report/diffuchat_findings_report.pdf",
            "diffuchat_complete_package/metrics/metrics.json",
            "diffuchat_complete_package/metrics/train_v4.log",
            "diffuchat_complete_package/weights/diffusion_best.pt",
            "diffuchat_complete_package/weights/ar_best.pt",
            "diffuchat_complete_package/model/diffusion_lm.onnx",
            "diffuchat_complete_package/model/tokenizer.json",
            "diffuchat_complete_package/code/kernel/train.py",
            "diffuchat_complete_package/code/repo/src/diffuchat/model.py",
            "diffuchat_complete_package/web-app/src/app/page.tsx",
            "diffuchat_complete_package/web-app/src/components/diffuchat/chat-view.tsx",
            "diffuchat_complete_package/web-app/mini-services/diffusion-service/index.ts",
            "diffuchat_complete_package/web-app/package.json",
            "diffuchat_complete_package/scripts/make_report.js",
            "diffuchat_complete_package/archive/v1/metrics.json",
        ]
        missing = [n for n in need if n not in names]
        assert not missing, f"missing: {missing}"
        m = json.loads(z.read("diffuchat_complete_package/metrics/metrics.json"))
        r = z.read("diffuchat_complete_package/README.md").decode()
        assert "demo@diffuchat.ai" in r and "login" in r.lower()
        print("integrity: OK | entries:", len(names), "| readme:", readme_ok)
        print("metrics.json top-level keys:", sorted(m.keys()) if isinstance(m, dict) else type(m).__name__)


if __name__ == "__main__":
    main()
