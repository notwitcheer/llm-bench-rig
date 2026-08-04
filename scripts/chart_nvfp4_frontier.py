#!/usr/bin/env python3
"""Gold & Crimson: NVFP4 lands ON the Qwen3.6-27B quant frontier (BTL-3 did not).

Same panel as the BTL-3 chart, one more point. The Qwen3.6-27B K-quant ladder (t090)
is the GOLD frontier Q8->Q3; NVFP4-A is the CRIMSON subject sitting right ON the line
between Q5 and Q6; BTL-3-Compact is the MUTE contrast, 13 pt below. x = effective
bits/weight (file bytes * 8 / params), y = composite quality (mean MMLU/GSM8K/HumanEval,
same harness, greedy, think-off).

Usage: python3 scripts/chart_nvfp4_frontier.py  (writes reports/chart_nvfp4_frontier.png)
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
NVFP4 = (5.88, 92.11)   # subject, CRIMSON, on the line
BTL = (2.46, 77.33)     # contrast, MUTE, off the line

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
    off = (0, 12) if name != "Q5_K_M" else (0, -24)
    ax.annotate(name, (x, y), textcoords="offset points", xytext=off,
                ha="center", fontsize=10.5, color=TEXT, zorder=5)

# NVFP4-A: the subject, on the line. Label sits in the empty mid-band with a leader.
nx, ny = NVFP4
ax.plot([nx], [ny], marker="D", markersize=15, color=CRIMSON, mec=BG, mew=1.5, zorder=6)
ax.annotate(
    "NVFP4-A, on the frontier\n92.1 composite (MMLU + GSM8K identical to Q8)\n18.1 GiB served, 79 tok/s",
    xy=(nx, ny - 0.35), xytext=(7.7, 85.4), textcoords="data", ha="left",
    fontsize=11, color=CRIMSON, zorder=8, linespacing=1.4,
    arrowprops=dict(arrowstyle="-|>", color=CRIMSON, lw=1.4))

# BTL-3-Compact: the contrast, off the line
bx, by = BTL
ax.plot([bx], [by], marker="D", markersize=13, color=MUTE, mec=BG, mew=1.5, zorder=6)
ax.annotate("BTL-3-Compact (AVQ2)\n77.3, off the frontier", (bx, by),
            textcoords="offset points", xytext=(-14, 24), ha="right",
            fontsize=11, color=MUTE, zorder=7, linespacing=1.3)

ax.set_xlim(2.0, 9.1)
ax.set_ylim(74, 94)
ax.invert_xaxis()
ax.grid(True, axis="both", color=GRID, lw=0.8, zorder=0)
ax.tick_params(length=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

ax.set_xlabel("effective bits per weight  (lower = more compressed)", color=MUTE, fontsize=11.5)
ax.set_ylabel("composite quality  (%, mean of 3 suites)", color=MUTE, fontsize=11.5)

fig.suptitle("NVFP4 lands on the Qwen3.6-27B frontier; BTL-3's sub-2.5-bit did not",
             fontsize=13.5, color=TEXT, x=0.09, ha="left")
fig.text(0.09, 0.895,
         "composite = mean(MMLU, GSM8K, HumanEval) · same harness, greedy, think-off · one RTX 5090",
         fontsize=9.5, color=MUTE)
fig.text(0.09, 0.862,
         "gold: Qwen3.6-27B K-quant ladder (t090) · crimson: NVFP4-A (llama.cpp GGUF) · mute: BTL-3-Compact",
         fontsize=9.5, color=MUTE)

fig.savefig("reports/chart_nvfp4_frontier.png", dpi=170)
print("wrote reports/chart_nvfp4_frontier.png")
