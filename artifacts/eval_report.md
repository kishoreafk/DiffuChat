# DiffuChat evaluation report

- diffusion checkpoint: `E:\Report Writing Project\diffuchat\artifacts\diffusion_lm.pt` (val loss 2.5700)
- AR baseline: `E:\Report Writing Project\diffuchat\artifacts\ar_baseline.pt` (val loss 1.5536)

## Experiment 1 — masked-span reconstruction (token recovery)

| span len | diffusion | AR-LM (prefix-only) | frequency | random guess | full-span exact (diff.) |
|---|---|---|---|---|---|
| 1 | 62.5% | 72.5% | 22.5% | 1.54% | 62% |
| 2 | 41.2% | 50.0% | 17.5% | 1.54% | 28% |
| 4 | 24.4% | 33.1% | 18.1% | 1.54% | 0% |
| 8 | 16.6% | 18.8% | 14.7% | 1.54% | 0% |
| 16 | 10.8% | 14.8% | 12.5% | 1.54% | 0% |

> The AR baseline cannot see the text after the span (causal); the diffusion model denoises conditioned on both sides.

## Experiment 2 — decoding ablation (token recovery, span 8)

| strategy | steps=5 | steps=10 | steps=20 | steps=40 |
|---|---|---|---|---|
| confidence | 14.6% | 19.3% | 17.7% | 15.6% |
| random | 18.2% | 13.5% | 14.6% | 14.1% |
| semi_ar | 17.7% | 16.7% | 15.1% | 20.8% |

## Experiment 3 — edit conditioning

| metric | unedited | edited (1 true pin) |
|---|---|---|
| pin survival | — | 100% |
| mean span CE (vs truth) | 5.1775 | 4.1527 |
| token recovery | 17.1% | 32.1% |
| CE improved vs unedited | — | 80% of windows |
