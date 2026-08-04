#!/usr/bin/env python3
"""Gold & Crimson: BTL-3-Compact against the honest quant frontier of its own base.

One panel, one question: what does going below 2.5 bits/weight cost. The Qwen3.6-27B
K-quant ladder (t090) is the GOLD reference curve Q8->Q3; BTL-3-Compact, a sub-2.5-bit
AVQ2 finetune of that same base, is the CRIMSON subject sitting far below where the
ladder would extrapolate. x = effective bits/weight (file bytes * 8 / params), y =
composite quality (mean of MMLU/GSM8K/HumanEval, same harness, greedy, think-off).
BTL-3 is a finetune of the base, not the same weights re-quantized, so this is "where
a different method+model lands vs the base's honest frontier", stated on the chart.

Usage: python3 scripts/chart_btl3_frontier.py  (writes reports/chart_btl3_frontier.png)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, GOLD, CRIMSON = "#0d0906", "#e8c44a", "#e06060"
TEXT, GRID, MUTE = "#f5e6d0", "#3a2f25", "#8a7a64"

# Qwen3.6-27B K-quant ladder (foil, GOLD): (bits/weight, composite, label)
LADDER = [
    (8.50, 92.32, "Q8_0"),
    (6.57, 92.32, "Q6_K"),
    (5.72, 92.24, "Q5_K_M"),
    (4.92, 91.96, "Q4_K_M"),
    (3.95, 90.54, "Q3_K_M"),
]
# BTL-3-Compact (subject, CRIMSON): bits/weight, composite
BTL = (2.46, 77.33)

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": TEXT,
    "ytick.color": TEXT, "axes.edgecolor": GRID, "font.size": 12,
})

fig, ax = plt.subplots(figsize=(11.5, 6.8))
fig.subplots_adjust(top=0.80, bottom=0.13, left=0.09, right=0.965)

lx = [r[0] for r in LADDER]
ly = [r[1] for r in LADDER]

# the honest frontier
ax.plot(lx, ly, color=GOLD, lw=2, zorder=2)
for x, y, name in LADDER:
    ax.plot([x], [y], marker="o", markersize=9, color=GOLD, mec=BG, mew=1.5, zorder=4)
    ax.annotate(name, (x, y), textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=10.5, color=TEXT, zorder=5)

# BTL-3-Compact: the subject, off the frontier
bx, by = BTL
ax.plot([bx], [by], marker="D", markersize=15, color=CRIMSON, mec=BG, mew=1.5, zorder=6)
ax.annotate("BTL-3-Compact\n2.46 bits/wt · 8.4 GB · 47 tok/s", (bx, by),
            textcoords="offset points", xytext=(-14, 26), ha="right",
            fontsize=11.5, color=CRIMSON, zorder=7, linespacing=1.3)

# the gap
ax.annotate("", xy=(bx + 0.04, by + 0.6), xytext=(3.90, 90.0),
            arrowprops=dict(arrowstyle="-|>", color=CRIMSON, lw=1.6))
ax.text(3.05, 84.6, "13 pt below Q3_K_M\nat fewer bits:\noff the honest frontier",
        ha="left", fontsize=11, color=CRIMSON, linespacing=1.35)

ax.set_xlim(2.0, 9.1)
ax.set_ylim(74, 94)
ax.invert_xaxis()  # more compressed toward the right, matching "how low can bits go"
ax.grid(True, axis="both", color=GRID, lw=0.8, zorder=0)
ax.tick_params(length=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

ax.set_xlabel("effective bits per weight  (lower = more compressed)", color=MUTE, fontsize=11.5)
ax.set_ylabel("composite quality  (%, mean of 3 suites)", color=MUTE, fontsize=11.5)

fig.suptitle("BTL-3-Compact vs the honest quant frontier of its own base, on one RTX 5090",
             fontsize=13.5, color=TEXT, x=0.09, ha="left")
fig.text(0.09, 0.895,
         "composite = mean(MMLU, GSM8K, HumanEval) · same harness, greedy, think-off",
         fontsize=9.5, color=MUTE)
fig.text(0.09, 0.862,
         "gold: Qwen3.6-27B K-quant ladder (t090) · crimson: BTL-3-Compact, an AVQ2 "
         "finetune of that base (not the same weights re-quantized)",
         fontsize=9.5, color=MUTE)

fig.savefig("reports/chart_btl3_frontier.png", dpi=170)
print("wrote reports/chart_btl3_frontier.png")
