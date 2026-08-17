#!/usr/bin/env python3
"""NVFP4 MTP chart: Qwen3.8-27B base vs MTP-spec decode across prompt depth.
Data: results/qwen3-8-27b-nvfp4/speed_nvfp4.json + speed_mtp.json (medians of 3).
Grouped bars: at every depth the MTP bar is ~1.7-1.8x the base bar.
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
GREEN = "#4fc98f"

OUT = "/opt/data/mercury-cards/qwen38-27b/nvfp4-mtp-q38.png"
TITLE = "The shipped MTP head holds 1.7x to 32k deep"
SUBTITLE = "Qwen3.8-27B NVFP4 \u00b7 vLLM 0.25.1 \u00b7 chat-server decode, median of 3 \u00b7 one RTX 5090 \u00b7 thinking off"

shapes = ["short prompt", "2k prompt", "8k prompt", "32k prompt"]
base = [68.95, 68.83, 68.17, 65.93]
mtp  = [119.79, 124.56, 123.16, 113.62]

x = np.arange(len(shapes))
w = 0.34

fig, ax = plt.subplots(figsize=(12, 7.2), dpi=200)
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

b1 = ax.bar(x - w/2, base, w, color=QWEN,  label="base decode", zorder=3)
b2 = ax.bar(x + w/2, mtp,  w, color=GREEN, label="+ shipped MTP head (spec tokens 2)", zorder=3)

for bars, vals in ((b1, base), (b2, mtp)):
    for r, v in zip(bars, vals):
        ax.text(r.get_x() + r.get_width()/2, v - 4, f"{v:.0f}",
                ha="center", va="top", color=BG, fontsize=13, fontweight="bold", zorder=4)

for i, (b, m) in enumerate(zip(base, mtp)):
    ax.text(x[i], m + 4, f"{m/b:.2f}x", ha="center", va="bottom",
            color=GREEN, fontsize=12.5, fontweight="bold")

ax.text(0.99, 0.05,
        "the trade: quality drops to q_avg 92.5, under the 15.9GB Q4_K_M gguf (93.2)",
        transform=ax.transAxes, color=TEXT, fontsize=13, ha="right", va="bottom",
        fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(shapes)
ax.set_ylim(0, 145)
ax.set_xlabel("prompt depth", color=MUTE, fontsize=12)
ax.set_ylabel("decode, tok/s (completion/(total\u2212TTFT))", color=MUTE, fontsize=12)
ax.grid(axis="y", color=GRID, lw=0.7, alpha=0.8, zorder=0)
for s in ax.spines.values(): s.set_color(GRID)
ax.tick_params(colors=MUTE, labelsize=11)
leg = ax.legend(loc="upper right", frameon=False, fontsize=11.5)
for t in leg.get_texts(): t.set_color(TEXT)

ax.set_title(TITLE, color=TEXT, fontsize=21, fontweight="bold", loc="left", pad=26)
ax.text(0, 1.022, SUBTITLE, transform=ax.transAxes, color=MUTE, fontsize=11.5, va="bottom")
fig.text(0.985, 0.968, "WITCHEER \u00b7 RTX 5090", color=MUTE, fontsize=10.5, ha="right")

fig.tight_layout(rect=(0, 0, 1, 0.985))
fig.savefig(OUT, facecolor=BG, bbox_inches="tight")
print("saved", OUT)
