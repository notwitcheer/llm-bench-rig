#!/usr/bin/env python3
"""Gold & Crimson quant-tax frontier for the t090 Qwen3.6-27B GGUF ladder (runs on the Mac).

One panel, one trade-off: composite quality (y, mean of MMLU/GSM8K/HumanEval) against
single-stream decode tok/s (x). The ladder is one CRIMSON line Q8->Q3; the eye reads the
"tax" directly: flat along the top while speed climbs, then a step down at Q3. Q4_K_M,
the sweet spot, is highlighted GOLD. Size rides in each point's label, so all three
measures (quality, speed, size) are present without a second y-scale. Per-suite numbers
live in the report table, not here; a composite is the chart's job, precision is the
table's. y starts at 90% by design and the note says so; every rung is within 1.8 points.

Usage: python3 scripts/chart_quant_tax.py  (writes reports/chart_quant_tax_5090.png)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, GOLD, CRIMSON = "#0d0906", "#e8c44a", "#e06060"
TEXT, GRID, MUTE = "#f5e6d0", "#3a2f25", "#8a7a64"

# rung: (decode tok/s, composite quality %, size GB, label, offset pts, ha)
RUNGS = [
    (52.77, 92.32, 29.0, "Q8_0",   (0, 13),   "center"),
    (63.68, 92.32, 22.4, "Q6_K",   (0, -34),  "center"),
    (71.76, 92.24, 19.5, "Q5_K_M", (0, 14),   "center"),
    (80.38, 91.96, 16.8, "Q4_K_M", (16, 6),   "left"),
    (90.16, 90.54, 13.5, "Q3_K_M", (-16, 2),  "right"),
]

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": TEXT,
    "ytick.color": TEXT, "axes.edgecolor": GRID, "font.size": 12,
})

fig, ax = plt.subplots(figsize=(11.5, 6.8))
fig.subplots_adjust(top=0.80, bottom=0.13, left=0.085, right=0.965)

xs = [r[0] for r in RUNGS]
ys = [r[1] for r in RUNGS]

# the ladder line
ax.plot(xs, ys, color=CRIMSON, lw=2, zorder=2)
# points: all crimson, Q4_K_M highlighted gold
for x, y, gb, name, off, ha in RUNGS:
    hi = name == "Q4_K_M"
    ax.plot([x], [y], marker="o", markersize=13 if hi else 9,
            color=GOLD if hi else CRIMSON, zorder=4,
            mec=BG, mew=1.5)
    lab = f"{name}\n{gb:g} GB"
    ax.annotate(lab, (x, y), textcoords="offset points", xytext=off,
                ha=ha, fontsize=11, color=GOLD if hi else TEXT,
                zorder=5, linespacing=1.25)

ax.set_xlim(46, 96)
ax.set_ylim(90.0, 92.9)
ax.grid(True, axis="both", color=GRID, lw=0.8, zorder=0)
ax.tick_params(length=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

ax.set_xlabel("single-stream decode  (tok/s, higher = faster)", color=MUTE, fontsize=11.5)
ax.set_ylabel("composite quality  (%, mean of 3 suites)", color=MUTE, fontsize=11.5)

# the flat run: delta text sits in the empty lower-left, the line itself shows it
ax.text(50.5, 91.30, "Q8 to Q4\n+52% tok/s\n-12 GB\n-0.4 pt quality", ha="left",
        fontsize=11, color=TEXT, linespacing=1.4)
# the knee: Q4 to Q3
ax.annotate("", xy=(89.2, 90.72), xytext=(81.6, 91.82),
            arrowprops=dict(arrowstyle="-|>", color=CRIMSON, lw=1.6))
ax.text(84.0, 91.55, "Q4 to Q3\n+12% tok/s\n-1.4 pt quality", ha="left",
        fontsize=10.5, color=CRIMSON, linespacing=1.35)

fig.suptitle("Qwen3.6-27B K-quants on one RTX 5090: the quant tax is near-zero down to Q4",
             fontsize=13.5, color=TEXT, x=0.085, ha="left")
fig.text(0.085, 0.885,
         "t090 quant-tax ladder · llama.cpp CUDA sm_120 · greedy, think-off · "
         "quality = mean(MMLU, GSM8K, HumanEval), same harness · y-axis 90-93%",
         fontsize=9.5, color=MUTE)

fig.savefig("reports/chart_quant_tax_5090.png", dpi=170)
print("wrote reports/chart_quant_tax_5090.png")
