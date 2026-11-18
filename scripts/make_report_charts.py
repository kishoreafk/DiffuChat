#!/usr/bin/env python3
"""Charts for the DiffuChat findings report (Template 07 Crystal Blue palette).

Style per charts.md: no top/right spines, dashed grid 20% opacity,
legend without frame, first/last/max/min labels only on lines.
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Template 07 body palette
ACCENT = '#2d7ab3'       # diffusion series
DEEP = '#1a4a7a'         # AR series
MUTED = '#5a7a96'
BRIGHT = '#4da8da'
TEXT = '#142840'
BORDER = '#c0d0e2'

plt.rcParams.update({
    'font.size': 11,
    'text.color': TEXT,
    'axes.labelcolor': TEXT,
    'axes.edgecolor': BORDER,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'axes.unicode_minus': False,
})

OUT = '/home/z/my-project/download/diffuchat-v4/report'
import os
os.makedirs(OUT, exist_ok=True)

M = json.load(open('/home/z/my-project/download/diffuchat-v4/metrics/metrics.json'))


def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(BORDER)
    ax.spines['bottom'].set_color(BORDER)
    ax.grid(True, linestyle='--', alpha=0.2, linewidth=0.5)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- fig 1: val curves
d_hist = M['diffusion']['history']
a_hist = M['ar']['history']
steps = [h['step'] for h in d_hist]
d_val = [h['val_loss'] for h in d_hist]
a_val = [h['val_loss'] for h in a_hist]

fig, ax = plt.subplots(figsize=(7.4, 3.6), constrained_layout=True)
style_ax(ax)
ax.plot(steps, d_val, color=ACCENT, lw=2.5, label='DenoiseLM (masked CE)', solid_capstyle='round')
ax.fill_between(steps, d_val, min(d_val) - 0.4, color=ACCENT, alpha=0.10, lw=0)
ax.plot(steps, a_val, color=DEEP, lw=2.5, label='AR baseline (next-token CE)', solid_capstyle='round')
# first / last labels only
for series, col, dy in ((d_val, ACCENT, 10), (a_val, DEEP, -14)):
    ax.annotate(f'{series[0]:.2f}', (steps[0], series[0]), textcoords='offset points',
                xytext=(2, dy), fontsize=9, color=col, fontweight='bold')
    ax.annotate(f'{series[-1]:.2f}', (steps[-1], series[-1]), textcoords='offset points',
                xytext=(-4, dy), fontsize=9, color=col, fontweight='bold', ha='right')
ax.set_xlabel('Training step')
ax.set_ylabel('Validation loss (EMA)')
ax.set_xlim(0, 12600)
ax.legend(loc='upper right', frameon=False, fontsize=10)
fig.savefig(f'{OUT}/fig_val_curves.png', dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- fig 2: OOD recovery
gen = M['generalization']
suites = ['WikiText-103\n(in-domain)', 'TinyStories\n(2nd domain)', 'FineWeb-Edu\n(unseen)']
rec = [gen['wikitext_val_in_domain']['diffusion_recovery'] * 100,
       gen['tinystories_ood']['diffusion_recovery'] * 100,
       gen['fineweb_edu_ood']['diffusion_recovery'] * 100]
ppl = [gen['wikitext_val_in_domain']['ar_ppl'],
       gen['tinystories_ood']['ar_ppl'],
       gen['fineweb_edu_ood']['ar_ppl']]

fig, ax = plt.subplots(figsize=(7.4, 3.3), constrained_layout=True)
style_ax(ax)
bars = ax.bar(suites, rec, width=0.52, color=[ACCENT, BRIGHT, MUTED], edgecolor='none')
for b, v in zip(bars, rec):
    ax.annotate(f'{v:.1f}%', (b.get_x() + b.get_width() / 2, v), xytext=(0, 5),
                textcoords='offset points', ha='center', fontsize=10.5,
                fontweight='bold', color=TEXT)
ax.set_ylabel('Diffusion token recovery (%)')
ax.set_ylim(0, 34)
ax.text(0.99, 0.95, 'AR perplexity: 36.1 / 6.6 / 150.2', transform=ax.transAxes,
        ha='right', va='top', fontsize=9.5, color=MUTED, style='italic')
fig.savefig(f'{OUT}/fig_ood.png', dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- fig 3: speed
speed = M['speed_bench']
names = ['AR baseline\n(greedy, 64 passes)',
         'Diffusion sampled\n(24 passes + refine)',
         'Diffusion greedy\n(10 passes)']
tps = [speed['ar']['tokens_per_s'],
       speed['diffusion_sampled_24pass_refine']['tokens_per_s'],
       speed['diffusion_greedy_10pass']['tokens_per_s']]
fwd = [speed['ar']['forward_passes'],
       speed['diffusion_sampled_24pass_refine']['forward_passes'],
       speed['diffusion_greedy_10pass']['forward_passes']]

fig, ax = plt.subplots(figsize=(7.4, 3.2), constrained_layout=True)
style_ax(ax)
cols = [MUTED, BRIGHT, ACCENT]
bars = ax.barh(names, tps, height=0.55, color=cols, edgecolor='none')
for b, v, f in zip(bars, tps, fwd):
    ax.annotate(f'{v:.0f} tok/s  ·  {f} forward passes', (v, b.get_y() + b.get_height() / 2),
                xytext=(6, 0), textcoords='offset points', va='center',
                fontsize=10, fontweight='bold', color=TEXT)
ax.set_xlabel('Decode throughput on T4 (tokens per second, 64-token answers)')
ax.set_xlim(0, 2050)
ax.xaxis.grid(False)
ax.yaxis.grid(True, linestyle='--', alpha=0.2, linewidth=0.5)
fig.savefig(f'{OUT}/fig_speed.png', dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- fig 4: coherence
coh = M['coherence_summary']
metrics = ['Repeated-bigram rate', 'Top-token share', 'Distinct-token ratio']
greedy = [coh['greedy_v2_style']['dup_bigram_rate'] * 100,
          coh['greedy_v2_style']['top_token_share'] * 100,
          coh['greedy_v2_style']['distinct_token_ratio'] * 100]
improved = [coh['improved_temp_top_p']['dup_bigram_rate'] * 100,
            coh['improved_temp_top_p']['top_token_share'] * 100,
            coh['improved_temp_top_p']['distinct_token_ratio'] * 100]

import numpy as np
x = np.arange(len(metrics))
w = 0.34
fig, ax = plt.subplots(figsize=(7.4, 3.3), constrained_layout=True)
style_ax(ax)
b1 = ax.bar(x - w / 2, greedy, w, color=MUTED, label='Greedy denoising', edgecolor='none')
b2 = ax.bar(x + w / 2, improved, w, color=ACCENT, label='Improved sampler (temp 0.8 + top-p + refine)', edgecolor='none')
for bars in (b1, b2):
    for b in bars:
        ax.annotate(f'{b.get_height():.1f}%', (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 4), textcoords='offset points', ha='center',
                    fontsize=9.5, fontweight='bold', color=TEXT)
ax.set_xticks(x, metrics)
ax.set_ylabel('Share of generated text (%)')
ax.set_ylim(0, 56)
ax.legend(loc='upper right', frameon=False, fontsize=9.5)
fig.savefig(f'{OUT}/fig_coherence.png', dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- fig 5: infill
infill = M['infill_comparison']
spans = sorted(int(k) for k in infill)
d_rec = [infill[str(s)]['diffusion_token_recovery'] * 100 for s in spans]
a_rec = [infill[str(s)]['ar_token_recovery'] * 100 for s in spans]
x = np.arange(len(spans))
w = 0.34
fig, ax = plt.subplots(figsize=(7.4, 3.3), constrained_layout=True)
style_ax(ax)
b1 = ax.bar(x - w / 2, d_rec, w, color=ACCENT, label='Diffusion (bidirectional, 10 passes)', edgecolor='none')
b2 = ax.bar(x + w / 2, a_rec, w, color=DEEP, label='AR (left-context only)', edgecolor='none')
for bars in (b1, b2):
    for b in bars:
        ax.annotate(f'{b.get_height():.1f}', (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 4), textcoords='offset points', ha='center',
                    fontsize=9.5, fontweight='bold', color=TEXT)
ax.set_xticks(x, [f'L = {s}' for s in spans])
ax.set_xlabel('Masked span length (tokens)')
ax.set_ylabel('Token recovery (%)')
ax.set_ylim(0, 36)
ax.legend(loc='upper right', frameon=False, fontsize=9.5)
fig.savefig(f'{OUT}/fig_infill.png', dpi=200)
plt.close(fig)

print('charts written to', OUT)
for f in sorted(os.listdir(OUT)):
    print(' ', f, os.path.getsize(os.path.join(OUT, f)) // 1024, 'KB')
