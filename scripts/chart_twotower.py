#!/usr/bin/env python3
"""Gold & Crimson chart for the Nemotron-TwoTower autopsy (runs on the Mac).

One panel: extra GPU memory to enable a ~2x AR-throughput speedup (log-x) vs the
throughput multiplier. TwoTower buys 2.42x with a second 30B backbone (~60GB,
2x80GB); the rig's measured spec-decode (t036, one 5090) buys the same ~2.1-2.2x
for a draft head of ~0-2GB. Same height, ~60x the memory.

Data: TwoTower from arXiv 2606.26493 + model card (2.42x, ~59GB/GPU x2).
Spec-decode from reports/specdecode-gemma-4-26b-a4b.md (one RTX 5090, vLLM, sm_120).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, GOLD, CRIMSON = "#0d0906", "#e8c44a", "#e06060"
TEXT, GRID, MUTE = "#f5e6d0", "#3a2f25", "#8a7a64"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": TEXT,
    "ytick.color": TEXT, "axes.edgecolor": GRID, "font.size": 12,
})

# (label, extra_mem_GB, multiplier, color, note, text_x, dy, ha)
pts = [
    ("MTP",      0.3, 2.13, GOLD, "ships with the model", 0.42, 0.02, "left"),
    ("DFlash",   1.0, 2.19, GOLD, "~1GB block drafter",   1.45, 0.03, "left"),
    ("EAGLE-3",  1.8, 1.69, GOLD, "0.9B drafter",         2.6, -0.02, "left"),
    ("Nemotron-TwoTower", 60.0, 2.42, CRIMSON, "a second 30B backbone", 40, 0.06, "right"),
]

fig, ax = plt.subplots(figsize=(12, 6.6))
ax.set_xscale("log")

# spec-decode "consumer, one 5090" band
ax.axvspan(0.15, 3.0, color=GOLD, alpha=0.06, zorder=0)
ax.text(0.62, 2.62, "speculative decoding\none RTX 5090 (32GB)", color=GOLD, fontsize=11,
        ha="center", va="bottom", zorder=3)
ax.text(60, 2.62, "Nemotron-TwoTower\n2x 80GB datacenter", color=CRIMSON, fontsize=11,
        ha="center", va="bottom", zorder=3)

for label, x, y, c, note, tx, dy, ha in pts:
    big = c == CRIMSON
    ax.scatter([x], [y], s=430 if big else 190, color=c, zorder=4,
               marker="D" if big else "o", edgecolor=BG, linewidth=1.5)
    shown = f"{y:.2f}x" if big else f"{label}  {y:.2f}x"
    ax.annotate(shown, (x, y), xytext=(tx, y + dy), ha=ha,
                color=TEXT if big else c, fontsize=12.5 if big else 11.5,
                fontweight="bold" if big else "normal", va="center")
    ax.annotate(note, (x, y), xytext=(tx, y + dy - 0.055), ha=ha,
                color=MUTE, fontsize=9.5, va="center")

# the gap annotation
ax.annotate("", xy=(48, 2.30), xytext=(2.4, 2.30),
            arrowprops=dict(arrowstyle="<->", color=TEXT, lw=1.3))
ax.text(11, 2.335, "same speedup, ~60x the memory", color=TEXT, fontsize=11.5,
        ha="center", va="bottom", style="italic")

ax.set_xlim(0.15, 130)
ax.set_ylim(1.55, 2.75)
ax.set_xlabel("extra GPU memory to buy the speedup  (GB, log scale)")
ax.set_ylabel("decode throughput vs autoregressive")
ax.set_title("The same 2x speedup: ~1GB on a gaming card, or 60GB in a datacenter",
             color=GOLD, fontsize=15.5, pad=14, loc="left")
ax.set_xticks([0.2, 0.5, 1, 2, 5, 10, 20, 50, 100])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid(color=GRID, lw=0.8, zorder=0)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

fig.tight_layout()
fig.savefig("reports/nemotron-twotower-autopsy.png", dpi=140)
print("wrote reports/nemotron-twotower-autopsy.png")
