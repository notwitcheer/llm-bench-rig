"""t068 omp harness-as-variable: per-bug SWE-bench outcome grid (Gold & Crimson v2).

Same Qwen3.6-27B-Q6_K, same 12 SWE-bench Verified bugs, same llama-server (:8090).
Only the agent SCAFFOLD differs: the rig-native tool loop (40-step budget) vs
omp v16.1.14 (450s wall-clock). omp resolves a strict SUPERSET (9/12 vs 8/12) — the
lone delta is sphinx-8621. Mechanism: omp gives up less (1 empty patch on the 4 hard
bugs vs native's 3) — persistence, not reasoning. Both miss the same model-ceiling bugs.

The grid is the EV+ payload: per-bug outcomes the tweet can't list. Not a bar chart.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = ("#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64")

# (label, native_outcome, omp_outcome)  R=resolved, U=wrong patch (tried, failed), E=empty (gave up)
BUGS = [
    ("astropy-12907", "R", "R"),
    ("django-16082", "R", "R"),
    ("matplotlib-23314", "R", "R"),
    ("flask-5014", "R", "R"),
    ("xarray-3677", "R", "R"),
    ("pytest-6202", "R", "R"),
    ("scikit-learn-14141", "R", "R"),
    ("sympy-22914", "R", "R"),
    ("sphinx-8621", "E", "R"),   # the flip: native gave up, omp landed it
    ("pylint-7080", "E", "U"),   # native gave up, omp tried (wrong patch)
    ("requests-1921", "U", "U"),  # both tried, both wrong
    ("seaborn-3187", "E", "E"),   # both gave up
]
FILL = {"R": GOLD, "U": "#3a1414", "E": "#161009"}
EDGE = {"R": GOLD, "U": CRIMSON, "E": MUTE}
GLYPH = {"R": "PASS", "U": "fail", "E": "gave up"}
TXTC = {"R": BG, "U": CRIMSON, "E": MUTE}

n = len(BUGS)
fig, ax = plt.subplots(figsize=(9.6, 8.6), facecolor=BG)
ax.set_facecolor(BG)

XN, XO = 0.0, 1.05
cw, ch = 0.92, 0.86

fig.suptitle("Swap the agent harness, move one bug in twelve", color=GOLD,
             fontsize=15.5, fontweight="bold", y=0.975)
fig.text(0.5, 0.93, "same Qwen3.6-27B-Q6_K · same 12 SWE-bench Verified bugs · same llama-server :8090 · "
         "only the scaffold differs", color=MUTE, fontsize=9.3, ha="center")

# column headers
ax.text(XN, n + 0.05, "rig-native\n40-step loop", ha="center", va="bottom", color=TEXT,
        fontsize=10.5, fontweight="bold")
ax.text(XO, n + 0.05, "omp v16.1.14\n450s wall", ha="center", va="bottom", color=TEXT,
        fontsize=10.5, fontweight="bold")

for r, (lab, nv, op) in enumerate(BUGS):
    y = n - 1 - r
    for outcome, cx in ((nv, XN), (op, XO)):
        ax.add_patch(FancyBboxPatch((cx - cw / 2, y - ch / 2), cw, ch,
                                    boxstyle="round,pad=0.0,rounding_size=0.08",
                                    fc=FILL[outcome], ec=EDGE[outcome], lw=1.7))
        ax.text(cx, y, GLYPH[outcome], ha="center", va="center",
                color=TXTC[outcome], fontsize=8.6, fontweight="bold")
    ax.text(-0.78, y, lab, ha="right", va="center", color=TEXT, fontsize=9.7)

# divider between the 8 shared resolves and the 4 hard bugs
dy = n - 8 - 0.5
ax.plot([-1.55, XO + 0.6], [dy, dy], color=GRID, lw=1.0, ls="--")
ax.text(-1.5, dy + 0.12, "8 shared resolves  (model does these unaided)", ha="left",
        va="bottom", color=MUTE, fontsize=8.4, style="italic")
ax.text(-1.5, dy - 0.12, "the 4 hard bugs", ha="left", va="top", color=MUTE,
        fontsize=8.4, style="italic")

# highlight the flip row (sphinx-8621)
flip_y = n - 1 - 8
ax.add_patch(Rectangle((XN - cw / 2 - 0.07, flip_y - 0.5), (XO - XN) + cw + 0.14, 1.0,
                       fill=False, ec=CRIMSON, lw=2.4))
ax.annotate("the only delta — omp gives up\nless, so it lands the patch native\nwouldn't even attempt",
            xy=(XO + cw / 2, flip_y), xytext=(XO + 0.78, flip_y),
            color=CRIMSON, fontsize=9.2, va="center", ha="left", fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=CRIMSON, lw=1.4))

# model-ceiling bracket on the bottom 3
ax.annotate("model ceiling:\nneither resolves",
            xy=(XO + cw / 2, 1.0), xytext=(XO + 0.78, 1.0),
            color=MUTE, fontsize=9.0, va="center", ha="left")

# empty-patch tally — the persistence mechanism in one line
ax.text(0.5, -1.15, "empty patches on the 4 hard bugs:   native 3   →   omp 1",
        transform=ax.get_xaxis_transform() if False else ax.transData,
        ha="center", va="center", color=TEXT, fontsize=10.2, fontweight="bold")
ax.text(0.5, -1.7, "persistence, not reasoning — the scaffold is a +1/12 lever; the model is the ceiling",
        ha="center", va="center", color=MUTE, fontsize=9.0, style="italic")

ax.set_xlim(-2.6, 3.5)
ax.set_ylim(-2.1, n + 0.7)
ax.axis("off")
fig.text(0.985, 0.012, "WITCHEER", color=MUTE, fontsize=9, ha="right", fontweight="bold")
fig.subplots_adjust(left=0.02, right=0.98, top=0.905, bottom=0.02)
fig.savefig("reports/omp-harness-as-variable.png", dpi=150, facecolor=BG)
print("wrote reports/omp-harness-as-variable.png")
