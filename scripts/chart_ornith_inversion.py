"""t082 Leg B — the harness-as-variable 2x2: the model ranking inverts (Gold & Crimson v2).

Same 12 SWE-bench Verified bugs, same quant (Q4_K_M), same think-OFF, same llama.cpp build
(b9653) — only the MODEL and the HARNESS vary. Under the strict native loop the base leads
(7 vs 5); under the lenient omp harness Ornith leads (8 vs 7). The lines CROSS: which model
"wins" flips with the harness alone.

left  = the slopegraph (strict -> lenient): base flat 7->7, Ornith 5->8 crossing above it.
right = the per-bug 2x2 grid (4 columns), the bugs that move highlighted.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = ("#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64")
AMBER = "#d8902f"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.6, 8.4), facecolor=BG,
                               gridspec_kw={"width_ratios": [1.0, 1.32]})
fig.suptitle("Leg A's regression is a harness artifact: base +2 under the strict loop, gone under lenient omp",
             color=GOLD, fontsize=14.0, fontweight="bold", y=0.975)
fig.text(0.5, 0.93, "Ornith-1.0-35B vs base Qwen3.5-35B-A3B · same 12 SWE-bench Verified bugs · same quant · think-OFF · llama.cpp b9653 · only model + harness vary",
         color=MUTE, fontsize=9.0, ha="center")

# ================= LEFT: the slopegraph =================
ax1.set_facecolor(BG)
X0, X1 = 0.0, 1.0
# base 7 -> 7 ; ornith 5 -> 8
ax1.plot([X0, X1], [7, 7], color=GOLD, lw=3.2, solid_capstyle="round", zorder=3)
ax1.plot([X0, X1], [5, 8], color=AMBER, lw=3.6, solid_capstyle="round", zorder=3)
# crossover point: ornith crosses base(=7) at x where 5+3x=7 -> x=2/3
xc = (7 - 5) / 3.0
ax1.scatter([xc], [7], s=120, color=CRIMSON, zorder=5, marker="X")
ax1.annotate("the lines cross\nhere", xy=(xc, 7), xytext=(xc - 0.04, 8.55),
             color=CRIMSON, fontsize=9.2, fontweight="bold", ha="center", va="bottom",
             arrowprops=dict(arrowstyle="-", color=CRIMSON, lw=1.2))
# endpoint markers + labels
for x, y, c, m in ((X0, 7, GOLD, "o"), (X1, 7, GOLD, "o"), (X0, 5, AMBER, "D"), (X1, 8, AMBER, "D")):
    ax1.scatter([x], [y], s=150, color=c, zorder=4, marker=m)
ax1.text(X0 - 0.04, 7.0, "base 7", color=GOLD, fontsize=11.5, va="center", ha="right", fontweight="bold")
ax1.text(X1 + 0.04, 7.0, "base 7", color=GOLD, fontsize=11.5, va="center", ha="left", fontweight="bold")
ax1.text(X0 - 0.04, 5.0, "Ornith 5", color=AMBER, fontsize=11.5, va="center", ha="right", fontweight="bold")
ax1.text(X1 + 0.04, 8.0, "Ornith 8", color=AMBER, fontsize=11.5, va="center", ha="left", fontweight="bold")
ax1.set_xticks([X0, X1])
ax1.set_xticklabels(["strict\nnative loop", "lenient\nomp harness"], color=TEXT, fontsize=11, fontweight="bold")
ax1.set_xlim(-0.42, 1.42)
ax1.set_ylim(4.3, 9.0)
ax1.set_ylabel("resolved / 12 SWE-bench Verified", color=TEXT, fontsize=10.5)
ax1.set_yticks([5, 6, 7, 8])
ax1.tick_params(colors=MUTE)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax1.spines[s].set_color(GRID)
ax1.text(0.0, 4.52, "base +2", color=GOLD, fontsize=9.0, ha="center", va="center")
ax1.text(1.0, 4.52, "Ornith +1*", color=AMBER, fontsize=9.0, ha="center", va="center")

# ================= RIGHT: per-bug 2x2 grid =================
ax2.set_facecolor(BG)
# columns: base-native, base-omp, ornith-native, ornith-omp
BUGS = [
    ("astropy-12907",      "R", "R", "E", "R"),
    ("xarray-3677",        "R", "R", "E", "R"),
    ("matplotlib-23314",   "E", "R", "E", "R"),
    ("pytest-6202",        "R", "W", "R", "R"),
    ("django-16082",       "R", "R", "R", "R"),
    ("flask-5014",         "R", "R", "R", "R"),
    ("scikit-learn-14141", "R", "R", "R", "R"),
    ("sympy-22914",        "R", "R", "R", "R"),
    ("seaborn-3187",       "W", "W", "W", "E"),
    ("requests-1921",      "W", "W", "W", "W"),
    ("pylint-7080",        "E", "E", "E", "E"),
    ("sphinx-8621",        "E", "W", "W", "E"),
]
FILL = {"R": GOLD, "W": "#3a1414", "E": "#161009"}
EDGE = {"R": GOLD, "W": CRIMSON, "E": MUTE}
GLYPH = {"R": "PASS", "W": "fail", "E": "gave up"}
TXTC = {"R": BG, "W": CRIMSON, "E": MUTE}
COLS = [("base\nstrict", GOLD), ("base\nomp", GOLD), ("Ornith\nstrict", AMBER), ("Ornith\nomp", AMBER)]
n = len(BUGS)
cw, ch = 0.86, 0.84
XS = [0.0, 1.0, 2.25, 3.25]  # gap between the base pair and the ornith pair
for cx, (clab, cc) in zip(XS, COLS):
    ax2.text(cx, n + 0.15, clab, ha="center", va="bottom", color=cc, fontsize=9.6, fontweight="bold")
for r, (lab, bn, bo, on, oo) in enumerate(BUGS):
    y = n - 1 - r
    for outcome, cx in ((bn, XS[0]), (bo, XS[1]), (on, XS[2]), (oo, XS[3])):
        ax2.add_patch(FancyBboxPatch((cx - cw / 2, y - ch / 2), cw, ch,
                                     boxstyle="round,pad=0.0,rounding_size=0.09",
                                     fc=FILL[outcome], ec=EDGE[outcome], lw=1.5))
        ax2.text(cx, y, GLYPH[outcome], ha="center", va="center", color=TXTC[outcome], fontsize=7.6, fontweight="bold")
    # highlight the rows that actually move (Ornith strict->omp flips to PASS)
    if on != oo and oo == "R":
        ax2.add_patch(Rectangle((XS[2] - cw / 2 - 0.07, y - ch / 2 - 0.06), (XS[3] - XS[2]) + cw + 0.14, ch + 0.12,
                                fill=False, ec=CRIMSON, lw=2.0))
    ax2.text(-0.72, y, lab, ha="right", va="center", color=TEXT, fontsize=8.8)
# divider between the two model pairs
ax2.axvline((XS[1] + XS[2]) / 2, color=GRID, lw=1.2, ls=":")
ax2.annotate("Ornith's 3 strict-loop\nlosses recover\nunder omp",
             xy=(XS[3] + cw / 2, n - 1 - 1), xytext=(XS[3] + 0.55, n - 2.4),
             color=CRIMSON, fontsize=8.6, va="center", ha="left", fontweight="bold",
             arrowprops=dict(arrowstyle="-", color=CRIMSON, lw=1.1))
ax2.text((XS[0] + XS[1]) / 2, -1.2, "base:  7 -> 7  (robust)", ha="center", va="center", color=GOLD, fontsize=10, fontweight="bold")
ax2.text((XS[2] + XS[3]) / 2, -1.2, "Ornith:  5 -> 8  (harness-sensitive)", ha="center", va="center", color=AMBER, fontsize=10, fontweight="bold")
ax2.set_xlim(-2.5, 5.2)
ax2.set_ylim(-1.9, n + 1.0)
ax2.axis("off")

fig.text(0.012, 0.012, "*Ornith's omp edge is pytest-6202 alone, which the base lost to omp's 450s wall-clock cap (it solved it in the native loop); both resolve the same other 7.",
         color=MUTE, fontsize=7.6, ha="left")
fig.text(0.99, 0.012, "WITCHEER", color=MUTE, fontsize=9, ha="right", fontweight="bold")
fig.subplots_adjust(left=0.085, right=0.99, top=0.9, bottom=0.075, wspace=0.18)
fig.savefig("reports/ornith-legb-inversion.png", dpi=150, facecolor=BG)
print("wrote reports/ornith-legb-inversion.png")
