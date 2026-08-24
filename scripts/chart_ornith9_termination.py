#!/usr/bin/env python3
"""Single-finding chart: the 25 Ornith 1.5 9B GPQA think-on chains that never terminate.

Three legs, one bar each: converged count out of 25. House palette (deep navy).
Run via: /opt/data/venvs/charts/bin/python scripts/chart_ornith9_termination.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG   = "#0e1420"
TEXT = "#eaf0f7"
MUTE = "#7d8ba0"
GRID = "#20293a"
BAR_STUCK = "#e0645f"   # still truncated
BAR_CONV  = "#35d0a5"   # converged

OUT = "reports/chart_ornith9_termination.png"

legs = [
    ("Q6_K\n16k budget (as-run)", 0),
    ("Q6_K\n32k budget (retry)", 0),
    ("BF16\n32k budget (falsification leg)", 3),
]
TOTAL = 25

fig, ax = plt.subplots(figsize=(12, 6.8), dpi=110)
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
fig.subplots_adjust(left=0.075, right=0.965, top=0.76, bottom=0.13)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0, labelsize=13, colors=MUTE)
ax.grid(axis="y", color=GRID, linewidth=1, alpha=0.7); ax.set_axisbelow(True)

xs = range(len(legs))
stuck = [TOTAL - c for _, c in legs]
conv  = [c for _, c in legs]
ax.bar(xs, stuck, 0.52, color=BAR_STUCK, zorder=3, label="never terminated (full window burned)")
ax.bar(xs, conv, 0.52, bottom=stuck, color=BAR_CONV, zorder=3, label="converged")

for x, (label, c) in zip(xs, legs):
    ax.text(x, TOTAL - c - 1.6, f"{TOTAL - c}/25 stuck", ha="center", fontsize=14,
            fontweight="bold", color=TEXT)
    if c:
        ax.text(x + 0.33, TOTAL - c / 2, f"{c} converged", ha="left", va="center",
                fontsize=12.5, fontweight="bold", color=BAR_CONV)

ax.set_xticks(list(xs))
ax.set_xticklabels([l for l, _ in legs], fontsize=13, color=TEXT)
ax.set_ylim(0, 29)
ax.set_ylabel("items (of the 25 non-terminating think-on chains)", fontsize=13, color=TEXT)

fig.text(0.075, 0.93, "Ornith 1.5 9B: the GPQA think-chains that never finish",
         fontsize=25, fontweight="bold", color=TEXT)
fig.text(0.075, 0.845,
         "25 of 198 GPQA-diamond think-on items truncated at the 16k cap under greedy decoding.\n"
         "Doubling the budget recovers zero; the vendor's own BF16 GGUF at 32k recovers three.\n"
         "The wall is the model, not the quant.",
         fontsize=13, color=MUTE, va="top")
ax.legend(loc="upper right", frameon=False, fontsize=12, labelcolor=TEXT)
fig.text(0.965, 0.035, "WITCHEER  ·  RTX 5090 benchmarks", fontsize=11.5, color=MUTE, ha="right")

fig.savefig(OUT, facecolor=BG)
print("wrote", OUT)
