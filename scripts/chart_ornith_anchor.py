"""t082 Ornith-1.0-35B vs base Qwen3.5-35B-A3B (Gold & Crimson v2), two panels.

Same 12 SWE-bench Verified bugs, same rig native harness, same quant (Q4_K_M), both
think-OFF — only the MODEL differs. The headline is an INVERSION: the synthetic Agentic
Score ranks Ornith ABOVE its base (98.06 vs 97.5), while the real-bug anchor ranks it
BELOW (5/12 vs 7/12). The board doesn't just miss the regression — it flips the ranking.

left  = the inversion as "% of base" on the two metrics (synthetic ~equal, real 71%).
right = the per-bug grid: a strict subset, the 2 losses to tool-call JSON fragility.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = ("#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64")
AMBER = "#d8902f"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 8.2), facecolor=BG,
                               gridspec_kw={"width_ratios": [1.0, 1.18]})
fig.suptitle("The synthetic board ranks Ornith above its base; real bugs rank it below",
             color=GOLD, fontsize=15, fontweight="bold", y=0.975)
fig.text(0.5, 0.93, "Ornith-1.0-35B vs base Qwen3.5-35B-A3B · same 12 SWE-bench Verified bugs · same harness · think-OFF · Q4_K_M",
         color=MUTE, fontsize=9.3, ha="center")

# ================= LEFT: the inversion, as % of base =================
ax1.set_facecolor(BG)
ROWS = ["Synthetic\nAgentic Score", "Real SWE-bench\nVerified resolve"]
# Ornith as % of base: synthetic 98.06/97.5 = 100.6% ; real 5/7 = 71.4%
PCT = [100.6, 71.4]
PLAB = ["100.6%", "71%"]
RAWS = ["98.06  vs  97.5", "5/12  vs  7/12"]
COLS = [AMBER, CRIMSON]
ys = np.array([1.0, 0.0])
ax1.axvline(100, color=MUTE, lw=1.5, ls="--")
ax1.text(100, 1.62, "base = 100%", color=MUTE, fontsize=9.5, ha="center", va="bottom")
for y, lab, p, plab, raw, c in zip(ys, ROWS, PCT, PLAB, RAWS, COLS):
    ax1.plot([100, p], [y, y], color=c, lw=3.4, solid_capstyle="round", zorder=2)
    ax1.scatter([100], [y], s=150, color=GOLD, zorder=3, marker="o")
    ax1.scatter([p], [y], s=185, color=c, zorder=3, marker="D")
    ax1.text(p + (2.5 if p > 100 else -2.5), y, plab, color=c, fontsize=13,
             va="center", ha="left" if p > 100 else "right", fontweight="bold")
    ax1.text(64, y + 0.17, lab, color=TEXT, fontsize=11, va="bottom", ha="left", fontweight="bold")
    ax1.text(64, y - 0.2, raw, color=MUTE, fontsize=9.5, va="top", ha="left")
ax1.set_xlim(58, 112)
ax1.set_ylim(-0.75, 1.9)
ax1.set_yticks([])
ax1.set_xlabel("Ornith performance as % of its base", color=TEXT, fontsize=10.5)
ax1.tick_params(colors=MUTE)
for s in ax1.spines.values():
    s.set_color(GRID)
ax1.annotate("the inversion: synthetic says\n≈ base (even a touch better),\nreal bugs say 71% of base",
             xy=(71.4, 0.0), xytext=(60, -0.55), color=CRIMSON, fontsize=9.3, fontweight="bold",
             va="center", ha="left")

# ================= RIGHT: per-bug grid =================
ax2.set_facecolor(BG)
BUGS = [
    ("django-16082", "R", "R"), ("flask-5014", "R", "R"), ("pytest-6202", "R", "R"),
    ("scikit-learn-14141", "R", "R"), ("sympy-22914", "R", "R"),
    ("astropy-12907", "R", "E"), ("xarray-3677", "R", "E"),
    ("sphinx-8621", "E", "U"), ("matplotlib-23314", "E", "E"), ("pylint-7080", "E", "E"),
    ("seaborn-3187", "U", "U"), ("requests-1921", "U", "U"),
]
FILL = {"R": GOLD, "U": "#3a1414", "E": "#161009"}
EDGE = {"R": GOLD, "U": CRIMSON, "E": MUTE}
GLYPH = {"R": "PASS", "U": "fail", "E": "gave up"}
TXTC = {"R": BG, "U": CRIMSON, "E": MUTE}
n = len(BUGS)
XB, XO = 0.0, 1.05
cw, ch = 0.92, 0.84
ax2.text(XB, n + 0.1, "base", ha="center", va="bottom", color=TEXT, fontsize=10, fontweight="bold")
ax2.text(XO, n + 0.1, "Ornith", ha="center", va="bottom", color=AMBER, fontsize=10, fontweight="bold")
for r, (lab, bv, ov) in enumerate(BUGS):
    y = n - 1 - r
    for outcome, cx in ((bv, XB), (ov, XO)):
        ax2.add_patch(FancyBboxPatch((cx - cw / 2, y - ch / 2), cw, ch,
                                     boxstyle="round,pad=0.0,rounding_size=0.08",
                                     fc=FILL[outcome], ec=EDGE[outcome], lw=1.6))
        ax2.text(cx, y, GLYPH[outcome], ha="center", va="center", color=TXTC[outcome], fontsize=8.2, fontweight="bold")
    ax2.text(-0.72, y, lab, ha="right", va="center", color=TEXT, fontsize=9.2)
# highlight the 2 regression rows (astropy, xarray)
top = n - 1 - 5
ax2.add_patch(Rectangle((XB - cw / 2 - 0.06, (top - 1) - 0.5), (XO - XB) + cw + 0.12, 2.0,
                        fill=False, ec=CRIMSON, lw=2.2))
ax2.annotate("2 bugs the base SOLVED,\nlost to tool-call JSON fragility",
             xy=(XO + cw / 2, top - 0.5), xytext=(XO + 0.7, top - 0.5),
             color=CRIMSON, fontsize=8.8, va="center", ha="left", fontweight="bold",
             arrowprops=dict(arrowstyle="-", color=CRIMSON, lw=1.3))
ax2.text((XB + XO) / 2, -1.15, "resolved:  base 7/12  →  Ornith 5/12  (a strict subset)",
         ha="center", va="center", color=TEXT, fontsize=10.5, fontweight="bold")
ax2.set_xlim(-2.4, 3.0)
ax2.set_ylim(-1.7, n + 0.8)
ax2.axis("off")

fig.text(0.99, 0.012, "WITCHEER", color=MUTE, fontsize=9, ha="right", fontweight="bold")
fig.subplots_adjust(left=0.04, right=0.985, top=0.9, bottom=0.08, wspace=0.12)
fig.savefig("reports/ornith-anchor.png", dpi=150, facecolor=BG)
print("wrote reports/ornith-anchor.png")
