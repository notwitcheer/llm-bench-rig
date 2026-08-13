#!/usr/bin/env python3
"""AA-style ladder chart: Lightning quant-tax — q_avg vs measured VRAM peak, one RTX 5090.
Every point traces to results/nvidia-nemotron-3-5-lightning-30b-a3b-<rung>/{quality,speed}.json.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BG    = "#0e1420"
TEXT  = "#eaf0f7"
MUTE  = "#7d8ba0"
GRID  = "#20293a"
IDEAL = "#1f9e8a"
HILITE= "#ffffff"
NEMO  = "#e0645f"

OUT = "/opt/data/mercury-cards/nemotron-lightning/ladder-lightning.png"
TITLE = "Seven quants of the fastest 30B, one real casualty"
SUBTITLE = "Nemotron 3.5 Lightning GGUF ladder \u00b7 five-task q_avg vs measured VRAM peak \u00b7 one RTX 5090 \u00b7 thinking off"

# (rung, q_avg, vram_gib, tg128, (dx, dy, ha), subject?)
DATA = [
    ("IQ2_XXS", 82.25, 17.7, 403, (14,  -4, "left"),   False),
    ("IQ2_M",   83.93, 17.7, 399, (-14,  6, "right"),  True),
    ("IQ4_XS",  83.46, 17.7, 395, (14,  -4, "left"),   False),
    ("IQ3_XXS", 84.03, 18.5, 390, (14,   8, "left"),   False),
    ("Q3_K_M",  83.16, 18.6, 391, (14,  -8, "left"),   False),
    ("Q4_K_M",  83.73, 23.6, 377, (0,   14, "center"), False),
    ("Q5_K_M",  84.26, 25.2, 349, (0,   14, "center"), False),
]

fig, ax = plt.subplots(figsize=(12, 7.2), dpi=200)
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

# 24GB-card ceiling
ax.axvline(24, color=MUTE, lw=1.2, ls=(0, (5, 4)), alpha=0.85, zorder=1)
ax.text(23.75, 81.6, "24GB cards stop here", color=MUTE, fontsize=11,
        rotation=90, va="bottom", ha="right")

# ideal corner: small and smart (top-left)
ax.add_patch(Rectangle((16.8, 83.6), 2.6, 1.05, facecolor=IDEAL, alpha=0.13, zorder=0))
ax.text(16.95, 84.52, "small and smart", color=IDEAL, fontsize=11.5, va="top", alpha=0.9)

for rung, q, v, tg, (dx, dy, ha), subj in DATA:
    ax.scatter([v], [q], s=150 if subj else 110, color=NEMO, zorder=3,
               edgecolors=HILITE if subj else BG, linewidths=2.2 if subj else 1.0)
    ax.annotate(f"{rung}\n{tg} tok/s", (v, q), xytext=(dx, dy), textcoords="offset points",
                ha=ha, va="center", color=TEXT if subj else MUTE,
                fontsize=11.5 if subj else 10.5,
                fontweight="bold" if subj else "normal", linespacing=1.25, zorder=4)

ax.annotate("the one benchmark casualty:\nfloor rung drops 1.7 pts, all HellaSwag",
            (17.7, 82.25), xytext=(60, -16), textcoords="offset points",
            color=MUTE, fontsize=10.5, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.9, alpha=0.7))

ax.set_xlim(16.8, 26.6); ax.set_ylim(81.4, 84.9)
ax.set_xlabel("VRAM peak, GiB (fully resident, -ngl 99)", color=MUTE, fontsize=12)
ax.set_ylabel("q_avg (MMLU \u00b7 ARC-C \u00b7 HellaSwag \u00b7 GSM8K \u00b7 HumanEval)", color=MUTE, fontsize=12)
ax.grid(color=GRID, lw=0.7, alpha=0.8)
for s in ax.spines.values(): s.set_color(GRID)
ax.tick_params(colors=MUTE, labelsize=11)

ax.set_title(TITLE, color=TEXT, fontsize=21, fontweight="bold", loc="left", pad=26)
ax.text(0, 1.022, SUBTITLE, transform=ax.transAxes, color=MUTE, fontsize=11.5, va="bottom")
fig.text(0.985, 0.968, "WITCHEER \u00b7 RTX 5090", color=MUTE, fontsize=10.5, ha="right")

fig.tight_layout(rect=(0, 0, 1, 0.985))
fig.savefig(OUT, facecolor=BG, bbox_inches="tight")
print("saved", OUT)
