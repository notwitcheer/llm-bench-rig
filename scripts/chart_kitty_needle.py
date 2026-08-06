#!/usr/bin/env python3
"""Gold & Crimson: does 2-bit KV cache (Kitty) degrade long-context retrieval?

Per-depth needle recall, f16 KV vs Kitty 2-bit KV, on the HARD multi-needle probe
(8 needles hidden at 8 depths per prompt, keyed retrieval, Qwen3-8B, 16k+32k, 20
prompts/leg = 160 recoveries each). GOLD line = f16 baseline, CRIMSON line = 2-bit
KV. The point of the chart is the overlap: the two lines coincide at 100% for every
depth from 18.75% on, and the only wobble is at the shallowest depth where BOTH legs
slip (f16 90, 2-bit 95) which is noise, not quantizer damage. Secondary encoding
(circle vs diamond markers, solid vs dashed line, direct labels) so identity never
rests on color alone. Single-needle A/B was a flat 100 vs 100 and is noted, not plotted.

Usage: python3 scripts/chart_kitty_needle.py  (writes reports/chart_kitty_needle.png)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, GOLD, CRIMSON = "#0d0906", "#e8c44a", "#e06060"
TEXT, GRID, MUTE = "#f5e6d0", "#3a2f25", "#8a7a64"

DEPTHS = [6.25, 18.75, 31.25, 43.75, 56.25, 68.75, 81.25, 93.75]
F16 = [90.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
Kitty = [95.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": TEXT,
    "ytick.color": TEXT, "axes.edgecolor": GRID, "font.size": 12,
})

fig, ax = plt.subplots(figsize=(11.5, 6.6))
fig.subplots_adjust(top=0.74, bottom=0.16, left=0.10, right=0.90)

# lines: f16 solid+circle, 2-bit dashed+diamond (drawn on top with a surface ring)
ax.plot(DEPTHS, F16, color=GOLD, lw=2.5, marker="o", markersize=10, mec=BG, mew=1.5,
        solid_capstyle="round", zorder=3, label="f16 KV (baseline)")
ax.plot(DEPTHS, Kitty, color=CRIMSON, lw=2.5, ls="--", marker="D", markersize=8,
        mec=BG, mew=1.5, zorder=4, label="Kitty 2-bit KV")

# the only separated point: annotate both values, name it as noise
ax.annotate("90", (DEPTHS[0], F16[0]), textcoords="offset points", xytext=(0, -18),
            ha="center", va="top", fontsize=11, color=GOLD, fontweight="bold")
ax.annotate("95", (DEPTHS[0], Kitty[0]), textcoords="offset points", xytext=(0, 12),
            ha="center", va="bottom", fontsize=11, color=CRIMSON, fontweight="bold")
# notes placed in the empty band (every depth from 18.75% on sits at 100%)
ax.text(21, 90.6, "shallowest depth: both legs slip, and 2-bit is if\n"
        "anything ahead. Across 160 recoveries that is noise.",
        ha="left", va="center", fontsize=10.5, color=MUTE)
ax.text(21, 87.6, "overall recall    f16  98.8%     2-bit KV  99.4%",
        ha="left", va="center", fontsize=12, color=TEXT)

# the coincident region: one honest label instead of overlapping markers everywhere
ax.annotate("f16 and 2-bit KV coincide at 100%\nfor every depth from 18% on",
            (56.25, 100), textcoords="offset points", xytext=(0, 16),
            ha="center", va="bottom", fontsize=11, color=TEXT)

ax.set_xlim(-2, 100)
ax.set_ylim(84, 101.6)
ax.set_xticks(DEPTHS)
ax.set_xticklabels([f"{d:g}" for d in DEPTHS])
ax.set_xlabel("needle depth in context  (%, 0 = start · 100 = just before the question)",
              fontsize=11.5, color=MUTE)
ax.set_ylabel("needles recovered  (%)", fontsize=11.5, color=MUTE)
ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(length=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)

leg = ax.legend(loc="lower right", frameon=False, fontsize=12, labelcolor=TEXT,
                handlelength=2.4, borderaxespad=1.2)

fig.text(0.10, 0.93,
         "2-bit KV cache holds long-context retrieval, the axis Kitty never measured",
         fontsize=17, color=TEXT, fontweight="bold", ha="left")
fig.text(0.10, 0.865,
         "Per-depth needle recall on the hard 8-needle keyed probe. "
         "Qwen3-8B, 16k+32k, 160 recoveries per leg.",
         fontsize=11.5, color=MUTE, ha="left")
fig.text(0.90, 0.035,
         "one RTX 5090 32GB · Kitty-Pro (2-bit K/V, 20% keys to 4-bit) · "
         "single-needle A/B was 100 vs 100 · WITCHEER 2026-08-06",
         fontsize=9, color=MUTE, ha="right")

fig.savefig("reports/chart_kitty_needle.png", dpi=150)
print("wrote reports/chart_kitty_needle.png")
