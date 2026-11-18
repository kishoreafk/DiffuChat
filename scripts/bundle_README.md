# DiffuChat v4 — Complete Deliverable Bundle

Masked-diffusion language model (DenoiseLM) vs architecture-matched autoregressive
baseline (ARCharGPT), trained from scratch on Kaggle free-tier Tesla T4 GPUs.
Kernel: `kishore00afk/diffuchat-generalization-v3` (run v4, E7 generalization upgrade).

## Contents

| Folder | What it is |
|---|---|
| `report/` | **Findings report** (`diffuchat_findings_report.pdf`, 12 pages) + cover HTML source + chart PNGs |
| `metrics/` | `metrics.json` (every number in the report), loss histories, full training log |
| `weights/` | `diffusion_best.pt` + `ar_best.pt` — EMA weights at step 12,000 (config + history included) |
| `model/` | Browser-inference export: `diffusion_lm.onnx` (83 MB, parity 1.8e-05 vs PyTorch), BPE tokenizer, meta |
| `code/kernel/` | The exact `train.py` + `kernel-metadata.json` of the v4 Kaggle run |
| `code/repo/` | Core library: DenoiseLM, ARCharGPT, canvas generation engine, tests |
| `playground/` | Next.js frontend: landing / login / signup / chat with live diffusion animation / report dashboard, plus the ONNX inference mini-service |

## Headline results (v4)

- **Decode efficiency**: greedy diffusion resolves a 64-token answer in **10 forward passes at 1,523 tok/s** — **6.8x faster** than the AR baseline (64 passes, 223 tok/s) on T4. Quality-first sampled decoding (temp 0.8 + top-p 0.95 + remask-refine) still runs ~2x faster than AR.
- **Generalization** (first run to measure it): diffusion masked-token recovery 15.5% in-domain (WikiText-103), **29.8% on TinyStories**, 7.5% on the unseen FineWeb-Edu probe. AR perplexity: 36.1 / 6.6 / 150.2.
- **Infilling**: diffusion matches AR at spans of 1-8 tokens and is **1.9x better at L = 16** — the native bidirectional advantage.
- **Sampler upgrade without retraining**: repeated-bigram rate **19.9% -> 3.0%**, top-token share 26.8% -> 16.4%.
- **Val losses (EMA)**: diffusion 5.676 (masked CE), AR 3.575 (next-token CE, ppl 35.7). Objectives are not directly comparable; the report explains why.

## Running the playground

```bash
# 1. Serve inference (loads model/diffusion_lm.onnx)
cd playground/diffusion-service
DIFFUCHAT_MODEL_DIR=../../model bun run index.ts     # port 3030

# 2. Run the Next.js app (from the project root that contains src/)
bun run dev    # or: bun run build && bun run start   (port 3000)
```

The chat view denoises answers token-by-token in front of you; click any finished
token to pin-edit it — the pin never re-masks and the rest re-denoises around it.
The Report page renders `metrics.json` live.

## Retraining on Kaggle

Attach dataset `kishore00afk/diffuchat-token-cache` (677 MB pre-tokenized
WikiText-103 + TinyStories + FineWeb probe), enable GPU, and run
`code/kernel/train.py` as a script kernel. It smoke-tests the full pipeline in
~20 s (2-layer block-64 miniature) before starting the 2 x 12,000-step run
(~45 min total), resumes from `_last.pt`, and writes `metrics.json` +
checkpoints + histories at the end.

## Environment

Torch 2.11.0+cu130, CUDA 13.0, fp16 autocast, seed 42. Total GPU time: 44.2 min.
