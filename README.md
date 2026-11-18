# DiffuChat — Complete Deliverable Package

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

### Windows notes

The scripts are cross-platform (no `tee`/`cp`/`VAR=value` shell syntax):

| Command | What it does |
|---|---|
| `bun install` / `npm install` | installs the web-app deps (831 packages) |
| `bun run db:generate` | generates the Prisma client (needs `.env`) |
| `bun run dev` | Next.js dev server on <http://localhost:3000> |
| `bun run build` | production build + copies `.next/static` and `public` into `.next/standalone` |
| `bun run start` | serves the standalone production bundle on port 3000 |

The inference service needs its model directory. Either export it explicitly —
PowerShell: `$env:DIFFUCHAT_MODEL_DIR = (Resolve-Path ..\..\..\model)` — or rely on
the auto-detection, which probes `<cwd>/model`, `<cwd>/../model`, `<cwd>/../../model`
and `<cwd>/../../../model` for `diffusion_lm.onnx` (the mini-service sits three
levels below the package root, so the shipped model is found automatically).


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
