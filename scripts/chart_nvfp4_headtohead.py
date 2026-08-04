#!/usr/bin/env python3
"""Gold & Crimson dumbbell: NVFP4-A vs Q6_K head-to-head on one RTX 5090.

Deliberately NOT the frontier line (used for the last three cards). Three metrics,
each a horizontal dumbbell between Q6_K (gold) and NVFP4-A (crimson) with the real
values labelled, and the plain-English takeaway on the right. The story the frontier
line hides: at the same quality, NVFP4 is smaller and faster than the nearest K-quant.

Usage: python3 scripts/chart_nvfp4_headtohead.py  (writes reports/chart_nvfp4_headtohead.png)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, GOLD, CRIMSON = "#0d0906", "#e8c44a", "#e06060"
TEXT, GRID, MUTE = "#f5e6d0", "#3a2f25", "#8a7a64"

# (label, q6_value, nvfp4_value, fmt, takeaway)
ROWS = [
    ("composite quality", 92.32, 92.11, "{:.2f}", "tie  (-0.21, inside noise)"),
    ("decode  (tok/s)",    63.7,  78.6,  "{:.1f}", "+23% faster"),
    ("VRAM served  (GiB)", 21.7,  18.1,  "{:.1f}", "3.6 GiB smaller"),
]

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": TEXT, "font.size": 12,
})
fig, ax = plt.subplots(figsize=(11.5, 5.6))
fig.subplots_adjust(top=0.72, bottom=0.10, left=0.02, right=0.98)

XL, XR = 0.36, 0.62  # dumbbell band (leave room for metric name left, takeaway right)


def place(lo, hi, v):
    span = (hi - lo) or 1.0
    pad = span * 0.7
    a, b = lo - pad, hi + pad
    return XL + (v - a) / (b - a) * (XR - XL)


for i, (label, q6, nv, fmt, take) in enumerate(ROWS):
    y = len(ROWS) - 1 - i
    lo, hi = min(q6, nv), max(q6, nv)
    xq, xn = place(lo, hi, q6), place(lo, hi, nv)
    ax.plot([xq, xn], [y, y], color=MUTE, lw=2.5, zorder=1)
    ax.plot([xq], [y], "o", ms=13, color=GOLD, mec=BG, mew=1.5, zorder=3)
    ax.plot([xn], [y], "D", ms=14, color=CRIMSON, mec=BG, mew=1.5, zorder=3)
    ax.annotate(fmt.format(q6), (xq, y), textcoords="offset points",
                xytext=(0, 15), ha="center", color=GOLD, fontsize=11.5)
    ax.annotate(fmt.format(nv), (xn, y), textcoords="offset points",
                xytext=(0, -22), ha="center", color=CRIMSON, fontsize=11.5)
    ax.text(0.02, y, label, ha="left", va="center", color=TEXT, fontsize=13)
    ax.text(0.985, y, take, ha="right", va="center", color=CRIMSON, fontsize=12.5)

ax.set_xlim(0, 1)
ax.set_ylim(-0.6, len(ROWS) - 0.4)
ax.axis("off")

# small legend
ax.plot([0.36], [-0.5], "o", ms=11, color=GOLD, mec=BG, mew=1.2)
ax.text(0.375, -0.5, "Q6_K", va="center", color=MUTE, fontsize=11)
ax.plot([0.47], [-0.5], "D", ms=12, color=CRIMSON, mec=BG, mew=1.2)
ax.text(0.485, -0.5, "NVFP4-A", va="center", color=MUTE, fontsize=11)

fig.suptitle("NVFP4-A vs Q6_K on one RTX 5090: same quality, smaller, faster",
             fontsize=14, color=TEXT, x=0.02, ha="left")
fig.text(0.02, 0.80,
         "Qwen3.6-27B, same t090 harness (MMLU/GSM8K/HumanEval, greedy, think-off) · "
         "single-stream decode, served VRAM peak",
         fontsize=9.5, color=MUTE)

fig.savefig("reports/chart_nvfp4_headtohead.png", dpi=170)
print("wrote reports/chart_nvfp4_headtohead.png")
