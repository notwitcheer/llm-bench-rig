#!/usr/bin/env python3
"""AA-style ladder chart: Qwen3.8-27B quant-tax — q_avg vs measured VRAM peak, one RTX 5090.
Every point traces to results/qwen3-8-27b-<rung>/{quality,speed}.json.
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
QWEN  = "#8a7bff"

OUT = "/opt/data/mercury-cards/qwen38-27b/ladder-q38.png"
TITLE = "Seven quants of Qwen3.8-27B, no casualty at all"
SUBTITLE = "GGUF quant-tax ladder \u00b7 five-task q_avg vs measured VRAM peak \u00b7 one RTX 5090 \u00b7 thinking off"

# (rung, q_avg, vram_gib, tg128, (dx, dy, ha), subject?)
DATA = [
    ("UD-IQ2_XXS", 90.82, 10.06, 115, (14,  -2, "left"),   False),
    ("UD-IQ2_M",   91.47, 11.29, 103, (14,  -2, "left"),   False),
    ("UD-IQ3_XXS", 92.66, 12.78,  96, (-16,  4, "right"),  True),
    ("Q4_K_M",     93.17, 17.26,  79, (0,  -20, "center"), False),
    ("UD-Q4_K_XL", 93.46, 17.98,  76, (12,  14, "left"),   False),
    ("Q6_K",       93.66, 22.27,  63, (0,   16, "center"), False),
    ("Q8_0",       93.66, 27.60,  53, (0,   16, "center"), False),
]

fig, ax = plt.subplots(figsize=(12, 7.2), dpi=200)
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

# card ceilings
ax.axvline(16, color=MUTE, lw=1.2, ls=(0, (5, 4)), alpha=0.85, zorder=1)
ax.text(15.75, 90.35, "16GB cards stop here", color=MUTE, fontsize=11,
        rotation=90, va="bottom", ha="right")
ax.axvline(24, color=MUTE, lw=1.2, ls=(0, (5, 4)), alpha=0.85, zorder=1)
ax.text(23.75, 90.35, "24GB cards stop here", color=MUTE, fontsize=11,
        rotation=90, va="bottom", ha="right")

# ideal corner: small and smart (top-left)
ax.add_patch(Rectangle((9.4, 92.3), 4.2, 1.15, facecolor=IDEAL, alpha=0.13, zorder=0))
ax.text(9.6, 93.32, "small and smart", color=IDEAL, fontsize=11.5, va="top", alpha=0.9)

for rung, q, v, tg, (dx, dy, ha), subj in DATA:
    ax.scatter([v], [q], s=150 if subj else 110, color=QWEN, zorder=3,
               edgecolors=HILITE if subj else BG, linewidths=2.2 if subj else 1.0)
    ax.annotate(f"{rung}\n{tg} tok/s", (v, q), xytext=(dx, dy), textcoords="offset points",
                ha=ha, va="center", color=TEXT if subj else MUTE,
                fontsize=11.5 if subj else 10.5,
                fontweight="bold" if subj else "normal", linespacing=1.25, zorder=4)

ax.annotate("Q6_K ties Q8_0 exactly (93.66 both)\nfor 5.3 GiB less and +10 tok/s",
            (22.27, 93.66), xytext=(18, -26), textcoords="offset points",
            color=MUTE, fontsize=10.5, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.9, alpha=0.7))

ax.set_xlim(9.2, 29.6); ax.set_ylim(90.2, 94.4)
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
