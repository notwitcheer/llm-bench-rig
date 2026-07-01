"""t081 -- AgentWorld/base think-OFF vs think-ON, bounded-12 SWE-bench Verified (Gold & Crimson v2).

Same 12 bugs, same quant, only the reasoning toggle varies. AgentWorld is bug-for-bug
IDENTICAL both ways (8/12, same 8, same 3 give-ups, same 1 wrong-patch) -- flat line, flat grid.
Base moves by exactly one swap (astropy resolved->give-up, matplotlib give-up->resolved), net zero.

left  = the slopegraph (think-OFF -> think-ON): both lines flat, nothing to unlock.
right = the per-bug grid (4 columns), the base's one swap highlighted.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = ("#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64")
AMBER = "#d8902f"

d = json.load(open("results/agentworld/think_on_grid.json"))
ids = d["ids"]
grid = d["grid"]
SHORT = {i: i.split("__")[1].replace("-", "-") for i in ids}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.0, 8.4), facecolor=BG,
                               gridspec_kw={"width_ratios": [1.0, 1.32]})
fig.suptitle("Thinking mode changes nothing: AgentWorld is bug-for-bug identical think-OFF vs think-ON",
             color=GOLD, fontsize=13.5, fontweight="bold", y=0.975)
fig.text(0.5, 0.93, "Qwen-AgentWorld-35B-A3B vs base Qwen3.5-35B-A3B · same 12 SWE-bench Verified bugs · same quant · llama.cpp b9653 · only the reasoning toggle varies",
         color=MUTE, fontsize=9.0, ha="center")

# ================= LEFT: the slopegraph =================
ax1.set_facecolor(BG)
X0, X1 = 0.0, 1.0
# base 7 -> 7 ; agentworld 8 -> 8 (both perfectly flat)
ax1.plot([X0, X1], [7, 7], color=GOLD, lw=3.2, solid_capstyle="round", zorder=3)
ax1.plot([X0, X1], [8, 8], color=AMBER, lw=3.6, solid_capstyle="round", zorder=3)
for x, y, c, m in ((X0, 7, GOLD, "o"), (X1, 7, GOLD, "o"), (X0, 8, AMBER, "D"), (X1, 8, AMBER, "D")):
    ax1.scatter([x], [y], s=150, color=c, zorder=4, marker=m)
ax1.text(X0 - 0.04, 7.0, "base 7", color=GOLD, fontsize=11.5, va="center", ha="right", fontweight="bold")
ax1.text(X1 + 0.04, 7.0, "base 7", color=GOLD, fontsize=11.5, va="center", ha="left", fontweight="bold")
ax1.text(X0 - 0.04, 8.0, "AgentWorld 8", color=AMBER, fontsize=11.5, va="center", ha="right", fontweight="bold")
ax1.text(X1 + 0.04, 8.0, "AgentWorld 8", color=AMBER, fontsize=11.5, va="center", ha="left", fontweight="bold")
ax1.annotate("nothing to unlock", xy=(0.5, 8.35), color=CRIMSON, fontsize=10, fontweight="bold", ha="center")
ax1.set_xticks([X0, X1])
ax1.set_xticklabels(["think-OFF\n(banked)", "think-ON"], color=TEXT, fontsize=11, fontweight="bold")
ax1.set_xlim(-0.42, 1.42)
ax1.set_ylim(6.0, 9.0)
ax1.set_ylabel("resolved / 12 SWE-bench Verified", color=TEXT, fontsize=10.5)
ax1.set_yticks([6, 7, 8, 9])
ax1.tick_params(colors=MUTE)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax1.spines[s].set_color(GRID)

# ================= RIGHT: per-bug grid =================
ax2.set_facecolor(BG)
FILL = {"R": GOLD, "W": "#3a1414", "E": "#161009"}
EDGE = {"R": GOLD, "W": CRIMSON, "E": MUTE}
GLYPH = {"R": "PASS", "W": "fail", "E": "gave up"}
TXTC = {"R": BG, "W": CRIMSON, "E": MUTE}
COLS = [("AW\nOFF", AMBER), ("AW\nON", AMBER), ("base\nOFF", GOLD), ("base\nON", GOLD)]
n = len(ids)
cw, ch = 0.86, 0.84
XS = [0.0, 1.15, 2.6, 3.75]
for cx, (clab, cc) in zip(XS, COLS):
    ax2.text(cx, n + 0.15, clab, ha="center", va="bottom", color=cc, fontsize=10.5, fontweight="bold")
for r, iid in enumerate(ids):
    y = n - 1 - r
    aw_off, aw_on = grid["agentworld"][iid]["off"], grid["agentworld"][iid]["on"]
    b_off, b_on = grid["base"][iid]["off"], grid["base"][iid]["on"]
    for outcome, cx in ((aw_off, XS[0]), (aw_on, XS[1]), (b_off, XS[2]), (b_on, XS[3])):
        ax2.add_patch(FancyBboxPatch((cx - cw / 2, y - ch / 2), cw, ch,
                                     boxstyle="round,pad=0.0,rounding_size=0.09",
                                     fc=FILL[outcome], ec=EDGE[outcome], lw=1.5))
        ax2.text(cx, y, GLYPH[outcome], ha="center", va="center", color=TXTC[outcome], fontsize=7.6, fontweight="bold")
    if b_off != b_on:
        ax2.add_patch(Rectangle((XS[2] - cw / 2 - 0.07, y - ch / 2 - 0.06), (XS[3] - XS[2]) + cw + 0.14, ch + 0.12,
                                fill=False, ec=CRIMSON, lw=2.0))
    ax2.text(-0.85, y, iid.split("__")[1], ha="right", va="center", color=TEXT, fontsize=8.8)
ax2.axvline((XS[1] + XS[2]) / 2, color=GRID, lw=1.2, ls=":")
ax2.annotate("base's only move:\nastropy R->E,\nmatplotlib E->R\n(net zero)",
             xy=(XS[3] + cw / 2, n - 1 - 0), xytext=(XS[3] + 0.65, n - 1.9),
             color=CRIMSON, fontsize=8.2, va="center", ha="left", fontweight="bold",
             arrowprops=dict(arrowstyle="-", color=CRIMSON, lw=1.1))
ax2.text(XS[0] + (XS[1] - XS[0]) / 2, -1.35, "AW:  8 -> 8\n(12/12 identical)", ha="center", va="top", color=AMBER, fontsize=9.5, fontweight="bold")
ax2.text(XS[2] + (XS[3] - XS[2]) / 2, -1.35, "base:  7 -> 7\n(1 swap, net zero)", ha="center", va="top", color=GOLD, fontsize=9.5, fontweight="bold")
ax2.set_xlim(-3.0, 6.8)
ax2.set_ylim(-2.6, n + 1.0)
ax2.axis("off")

fig.text(0.012, 0.012, "n=12, bounded subset shared with the t082 Ornith harness study; the 30-bug think-OFF anchor (14/30 vs 16/30) remains the authoritative resolve-rate number.",
         color=MUTE, fontsize=7.6, ha="left")
fig.text(0.99, 0.012, "WITCHEER", color=MUTE, fontsize=9, ha="right", fontweight="bold")
fig.subplots_adjust(left=0.085, right=0.99, top=0.9, bottom=0.075, wspace=0.18)
fig.savefig("reports/agentworld-think-on-null.png", dpi=150, facecolor=BG)
print("wrote reports/agentworld-think-on-null.png")
