#!/usr/bin/env python3
"""Depth A/B chart: Qwen3.8-27B UD-IQ3_XXS vs Q6_K on long-context retrieve-and-use.
Data: results/qwen3-8-27b-{ud-iq3-xxs,q6-k}/longcontext_detail.json (15 tasks/depth).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG    = "#0e1420"
TEXT  = "#eaf0f7"
MUTE  = "#7d8ba0"
GRID  = "#20293a"
QWEN  = "#8a7bff"
AMBER = "#e8a13c"

OUT = "/opt/data/mercury-cards/qwen38-27b/depth-ab-q38.png"
TITLE = "The 1-point gap becomes a 27-point gap at long context"
SUBTITLE = "Qwen3.8-27B \u00b7 buried-fact retrieve-and-use, 15 tasks per depth \u00b7 one RTX 5090 \u00b7 thinking off"

depths = ["16k", "32k", "64k"]
q6  = [100.0, 100.0, 100.0]
iq3 = [73.3, 66.7, 80.0]

x = np.arange(len(depths))
w = 0.34

fig, ax = plt.subplots(figsize=(12, 7.2), dpi=200)
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

b1 = ax.bar(x - w/2, q6,  w, color=QWEN,  label="Q6_K (22.3 GiB peak)", zorder=3)
b2 = ax.bar(x + w/2, iq3, w, color=AMBER, label="UD-IQ3_XXS (12.8 GiB peak)", zorder=3)

for bars, vals in ((b1, q6), (b2, iq3)):
    for r, v in zip(bars, vals):
        ax.text(r.get_x() + r.get_width()/2, v - 6, f"{v:.0f}",
                ha="center", va="top", color=BG, fontsize=13, fontweight="bold", zorder=4)

ax.text(0.99, 0.055,
        "short-context board: 93.7 vs 92.7 (1.0 apart)\nsame task ids recur among the misses: fragility, not depth decay",
        transform=ax.transAxes, color=MUTE, fontsize=10.5, ha="right", va="bottom")

ax.set_xticks(x); ax.set_xticklabels(depths)
ax.set_ylim(0, 108)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_xlabel("context depth (needle buried mid-haystack)", color=MUTE, fontsize=12)
ax.set_ylabel("tasks solved, % (retrieve-and-use)", color=MUTE, fontsize=12)
ax.grid(axis="y", color=GRID, lw=0.7, alpha=0.8, zorder=0)
for s in ax.spines.values(): s.set_color(GRID)
ax.tick_params(colors=MUTE, labelsize=11)
leg = ax.legend(loc="lower left", frameon=False, fontsize=11.5)
for t in leg.get_texts(): t.set_color(TEXT)

ax.set_title(TITLE, color=TEXT, fontsize=21, fontweight="bold", loc="left", pad=26)
ax.text(0, 1.022, SUBTITLE, transform=ax.transAxes, color=MUTE, fontsize=11.5, va="bottom")
fig.text(0.985, 0.968, "WITCHEER \u00b7 RTX 5090", color=MUTE, fontsize=10.5, ha="right")

fig.tight_layout(rect=(0, 0, 1, 0.985))
fig.savefig(OUT, facecolor=BG, bbox_inches="tight")
print("saved", OUT)
