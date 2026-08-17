#!/usr/bin/env python3
"""GPQA ladder chart: the board's tight ladder splits on a hard benchmark.
Horizontal bars = GPQA-diamond per rung (desc); inline label carries board q_avg.
Data: results/qwen3-8-27b-*/gpqa.json, 2026-08-17.
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

OUT = "/opt/data/mercury-cards/qwen38-27b/gpqa-ladder-q38.png"
TITLE = "A ladder 2.9 points tight splits by 11.1 on GPQA"
SUBTITLE = "Qwen3.8-27B \u00b7 GPQA-diamond, 198 items, zero-shot \u00b7 one RTX 5090 \u00b7 thinking off \u00b7 gaps under ~3 pts are noise"

rungs = [
    ("Q4_K_M", 50.5, 93.2),
    ("Q6_K", 49.0, 93.7),
    ("UD-Q4_K_XL", 49.0, 93.5),
    ("NVFP4 (vLLM)", 47.0, 92.5),
    ("Q8_0", 47.0, 93.7),
    ("UD-IQ3_XXS", 45.0, 92.7),
    ("UD-IQ2_XXS", 42.9, 90.8),
    ("UD-IQ2_M", 39.4, 91.5),
]

names = [r[0] for r in rungs][::-1]
gpqa  = [r[1] for r in rungs][::-1]
board = [r[2] for r in rungs][::-1]
colors = [AMBER if n.startswith("UD-IQ2") else QWEN for n in names]

fig, ax = plt.subplots(figsize=(12, 7.2), dpi=200)
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

y = np.arange(len(names))
bars = ax.barh(y, gpqa, 0.62, color=colors, zorder=3)

for yi, (g, b) in enumerate(zip(gpqa, board)):
    ax.text(g - 0.6, yi, f"{g:.1f}", ha="right", va="center",
            color=BG, fontsize=13, fontweight="bold", zorder=4)
    ax.text(g + 0.6, yi, f"board q_avg {b:.1f}", ha="left", va="center",
            color=MUTE, fontsize=11, zorder=4)

ax.set_yticks(y); ax.set_yticklabels(names, fontsize=12)
ax.set_xlim(0, 62)
ax.set_xlabel("GPQA-diamond accuracy, % (zero-shot)", color=MUTE, fontsize=12)
ax.grid(axis="x", color=GRID, lw=0.7, alpha=0.8, zorder=0)
for s in ax.spines.values(): s.set_color(GRID)
ax.tick_params(colors=MUTE, labelsize=11)
for lbl in ax.get_yticklabels(): lbl.set_color(TEXT)

ax.set_title(TITLE, color=TEXT, fontsize=21, fontweight="bold", loc="left", pad=26)
ax.text(0, 1.022, SUBTITLE, transform=ax.transAxes, color=MUTE, fontsize=11.5, va="bottom")
fig.text(0.985, 0.968, "WITCHEER \u00b7 RTX 5090", color=MUTE, fontsize=10.5, ha="right")

fig.tight_layout(rect=(0, 0.05, 1, 0.985))
fig.text(0.985, 0.015,
         "the five-task board spreads these same rungs by only 2.9 points",
         color=TEXT, fontsize=13, ha="right", va="bottom", fontweight="bold")
fig.savefig(OUT, facecolor=BG, bbox_inches="tight")
print("saved", OUT)
