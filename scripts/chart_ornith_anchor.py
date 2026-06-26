"""t082 Ornith-1.0-35B vs base Qwen3.5-35B-A3B: per-bug SWE-bench grid (Gold & Crimson v2).

Same 12 SWE-bench Verified bugs, same rig native harness (40-step loop, official grader),
same quant (Q4_K_M), both think-OFF — only the MODEL differs. Ornith is the RL "self-scaffold"
coder claiming 75.6 SWE-V (in OpenHands). Stripped to the rig's strict native loop it resolves
5/12 vs the base's 7/12 — a regression, and a strict subset. The 2 it loses (astropy, xarray)
are bugs the base solved cleanly; Ornith burned its budget re-emitting malformed tool-call JSON.

The grid is the EV+ payload: the subset structure + the two fragility losses. Not a bar chart.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = ("#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64")

# (label, base_outcome, ornith_outcome)  R=resolved, U=wrong patch, E=empty/gave up
BUGS = [
    ("django-16082", "R", "R"),
    ("flask-5014", "R", "R"),
    ("pytest-6202", "R", "R"),
    ("scikit-learn-14141", "R", "R"),
    ("sympy-22914", "R", "R"),
    ("astropy-12907", "R", "E"),   # regression — base solved, Ornith burned budget on bad JSON
    ("xarray-3677", "R", "E"),     # regression — same
    ("sphinx-8621", "E", "U"),     # both unresolved (Ornith at least patched)
    ("matplotlib-23314", "E", "E"),
    ("pylint-7080", "E", "E"),
    ("seaborn-3187", "U", "U"),
    ("requests-1921", "U", "U"),
]
FILL = {"R": GOLD, "U": "#3a1414", "E": "#161009"}
EDGE = {"R": GOLD, "U": CRIMSON, "E": MUTE}
GLYPH = {"R": "PASS", "U": "fail", "E": "gave up"}
TXTC = {"R": BG, "U": CRIMSON, "E": MUTE}

n = len(BUGS)
fig, ax = plt.subplots(figsize=(9.8, 8.8), facecolor=BG)
ax.set_facecolor(BG)

XB, XO = 0.0, 1.05
cw, ch = 0.92, 0.86

fig.suptitle("The RL “self-scaffold” coder regresses on real bugs vs its own base",
             color=GOLD, fontsize=14.5, fontweight="bold", y=0.975)
fig.text(0.5, 0.93, "same 12 SWE-bench Verified bugs · same rig native harness · think-OFF · Q4_K_M · only the model differs",
         color=MUTE, fontsize=9.2, ha="center")

ax.text(XB, n + 0.05, "base\nQwen3.5-35B-A3B", ha="center", va="bottom", color=TEXT, fontsize=10.3, fontweight="bold")
ax.text(XO, n + 0.05, "Ornith-1.0-35B\n(self-scaffold RL)", ha="center", va="bottom", color=TEXT, fontsize=10.3, fontweight="bold")

for r, (lab, bv, ov) in enumerate(BUGS):
    y = n - 1 - r
    for outcome, cx in ((bv, XB), (ov, XO)):
        ax.add_patch(FancyBboxPatch((cx - cw / 2, y - ch / 2), cw, ch,
                                    boxstyle="round,pad=0.0,rounding_size=0.08",
                                    fc=FILL[outcome], ec=EDGE[outcome], lw=1.7))
        ax.text(cx, y, GLYPH[outcome], ha="center", va="center", color=TXTC[outcome], fontsize=8.6, fontweight="bold")
    ax.text(-0.78, y, lab, ha="right", va="center", color=TEXT, fontsize=9.6)

# divider under the 5 shared resolves
dy = n - 5 - 0.5
ax.plot([-1.7, XO + 0.6], [dy, dy], color=GRID, lw=1.0, ls="--")
ax.text(-1.65, dy + 0.12, "5 shared resolves", ha="left", va="bottom", color=MUTE, fontsize=8.3, style="italic")

# highlight the 2 regression rows (astropy, xarray)
top = n - 1 - 5   # astropy row y
ax.add_patch(Rectangle((XB - cw / 2 - 0.07, (top - 1) - 0.5), (XO - XB) + cw + 0.14, 2.0,
                       fill=False, ec=CRIMSON, lw=2.4))
ax.annotate("the 2 losses: bugs the base SOLVED.\nOrnith burned its 40-step budget\nre-emitting malformed tool-call JSON",
            xy=(XO + cw / 2, top - 0.5), xytext=(XO + 0.78, top - 0.5),
            color=CRIMSON, fontsize=9.0, va="center", ha="left", fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=CRIMSON, lw=1.4))

# bottom tallies
ax.text(0.5, -1.15, "resolved:   base 7/12     →     Ornith 5/12",
        ha="center", va="center", color=TEXT, fontsize=11.0, fontweight="bold")
ax.text(0.5, -1.7, "a strict subset — no bug the base failed did Ornith recover; empty patches rose 3 → 4",
        ha="center", va="center", color=MUTE, fontsize=9.0, style="italic")

ax.set_xlim(-2.7, 3.6)
ax.set_ylim(-2.1, n + 0.7)
ax.axis("off")
fig.text(0.985, 0.012, "WITCHEER", color=MUTE, fontsize=9, ha="right", fontweight="bold")
fig.subplots_adjust(left=0.02, right=0.98, top=0.905, bottom=0.02)
fig.savefig("reports/ornith-anchor.png", dpi=150, facecolor=BG)
print("wrote reports/ornith-anchor.png")
