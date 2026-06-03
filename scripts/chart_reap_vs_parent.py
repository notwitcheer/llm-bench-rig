#!/usr/bin/env python3
"""Gold & Crimson grouped bars: REAP (pruned) vs parent Qwen3.6-35B-A3B.
Honest full 0-100 y-axis (gaps are modest). The real story is the VRAM/speed callout."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BG="#0d0906"; GOLD="#e8c44a"; TEXT="#f5e6d0"; CRIMSON="#e06060"; GRID="#3a2f25"; MUTE="#8a7a64"

cats   = ["MMLU", "ARC-C", "HellaSwag", "GSM8K", "HumanEval"]
parent = [94.7, 97.0, 87.0, 92.0, 98.0]   # Qwen3.6-35B-A3B (full)
reap   = [87.7, 95.0, 82.0, 90.0, 94.0]   # 28B, 20% experts pruned

fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
fig.subplots_adjust(left=0.075, right=0.96, top=0.74, bottom=0.16)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0, labelsize=14, colors=TEXT)
ax.grid(axis="y", color=GRID, linewidth=1, alpha=0.6); ax.set_axisbelow(True)

xs = list(range(len(cats))); w = 0.38
for i,(lbl,vals,c) in enumerate([("Qwen3.6-35B-A3B (full)", parent, GOLD),
                                 ("REAP 28B (-20% experts)", reap, CRIMSON)]):
    off = (i - 0.5) * w
    bars = ax.bar([x+off for x in xs], vals, w, label=lbl, color=c)
    for b,v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.6, f"{v:g}", ha="center", va="bottom",
                color=TEXT, fontsize=11)

ax.set_xticks(xs); ax.set_xticklabels(cats, fontsize=14)
ax.set_ylim(0, 104); ax.set_ylabel("score  (%)", fontsize=13, color=TEXT)
leg = ax.legend(loc="lower center", frameon=False, fontsize=13, ncol=2, bbox_to_anchor=(0.5, -0.16))
for t in leg.get_texts(): t.set_color(TEXT)

fig.add_artist(Rectangle((0.075, 0.90), 0.07, 0.012, color=CRIMSON, transform=fig.transFigure))
fig.text(0.075, 0.835, "20% expert-pruning: VRAM saved, quality spent", fontsize=22, fontweight="bold", color=TEXT)
fig.text(0.075, 0.79, "Qwen3.6-35B-A3B vs its REAP-pruned 28B — same Q6, reasoning on, one RTX 5090 (n≈100/task)", fontsize=12.5, color=GOLD)
# the real story callout
fig.text(0.075, 0.025, "VRAM  27.3 → 21.6 GB  (−6)      ·      speed  260 → 247 tok/s  (no gain)      ·      ~3B active either way",
         fontsize=12.5, color=MUTE)
fig.text(0.96, 0.025, "WITCHEER · RTX 5090", fontsize=11, color=MUTE, ha="right")
fig.savefig("reports/chart-reap-vs-parent.png", facecolor=BG)
print("wrote reports/chart-reap-vs-parent.png")
