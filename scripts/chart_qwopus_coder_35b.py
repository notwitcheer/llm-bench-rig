"""t083 -- Qwopus-3.6-35B-A3B-Coder-MTP vs base, native-strict vs Hermes-as-harness (Gold & Crimson v2).

Both models gain under Hermes and lose nothing (real recoveries, no reshuffles), but Qwopus
remains a STRICT SUBSET of its base under either harness -- the -2 deficit narrows to -1,
it doesn't close or invert like Ornith's did under omp.

left  = the slopegraph (native-strict -> Hermes): base 7->8, Qwopus 5->7, gap narrows not closes.
right = the per-bug grid (4 columns), the harness-recovered bugs highlighted.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = ("#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64")
AMBER = "#d8902f"

d = json.load(open("results/agentworld/t083_grid.json"))
ids = d["ids"]
grid = d["grid"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.0, 8.4), facecolor=BG,
                               gridspec_kw={"width_ratios": [1.0, 1.32]})
fig.suptitle("Hermes narrows Qwopus-Coder's deficit to its base, but doesn't close it",
             color=GOLD, fontsize=13.2, fontweight="bold", y=0.975)
fig.text(0.5, 0.93, "Qwopus-3.6-35B-A3B-Coder-MTP vs base Qwen3.6-35B-A3B · same 12 SWE-bench Verified bugs · Q4/Q5_K_M · llama.cpp b9653 · think-off · native-strict vs Hermes-as-harness",
         color=MUTE, fontsize=8.8, ha="center")

# ================= LEFT: the slopegraph =================
ax1.set_facecolor(BG)
X0, X1 = 0.0, 1.0
ax1.plot([X0, X1], [7, 8], color=GOLD, lw=3.2, solid_capstyle="round", zorder=3)
ax1.plot([X0, X1], [5, 7], color=AMBER, lw=3.6, solid_capstyle="round", zorder=3)
for x, y, c, m in ((X0, 7, GOLD, "o"), (X1, 8, GOLD, "o"), (X0, 5, AMBER, "D"), (X1, 7, AMBER, "D")):
    ax1.scatter([x], [y], s=150, color=c, zorder=4, marker=m)
ax1.text(X0 - 0.04, 7.0, "base 7", color=GOLD, fontsize=11, va="center", ha="right", fontweight="bold")
ax1.text(X1 + 0.04, 8.0, "base 8", color=GOLD, fontsize=11, va="center", ha="left", fontweight="bold")
ax1.text(X0 - 0.04, 5.0, "Qwopus 5", color=AMBER, fontsize=11, va="center", ha="right", fontweight="bold")
ax1.text(X1 + 0.04, 7.0, "Qwopus 7", color=AMBER, fontsize=11, va="center", ha="left", fontweight="bold")
ax1.annotate("gap: -2", xy=(0.0, 6.0), color=CRIMSON, fontsize=9.5, fontweight="bold", ha="center")
ax1.annotate("gap: -1", xy=(1.0, 7.5), color=CRIMSON, fontsize=9.5, fontweight="bold", ha="center")
ax1.annotate("narrows, doesn't close", xy=(0.5, 4.4), color=CRIMSON, fontsize=10, fontweight="bold", ha="center")
ax1.set_xticks([X0, X1])
ax1.set_xticklabels(["native-strict", "Hermes-as-harness"], color=TEXT, fontsize=11, fontweight="bold")
ax1.set_xlim(-0.42, 1.42)
ax1.set_ylim(3.8, 9.0)
ax1.set_ylabel("resolved / 12 SWE-bench Verified", color=TEXT, fontsize=10.5)
ax1.set_yticks([4, 5, 6, 7, 8, 9])
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
COLS = [("base\nnative", GOLD), ("base\nHermes", GOLD), ("Qwopus\nnative", AMBER), ("Qwopus\nHermes", AMBER)]
n = len(ids)
cw, ch = 0.86, 0.84
XS = [0.0, 1.15, 2.6, 3.75]
for cx, (clab, cc) in zip(XS, COLS):
    ax2.text(cx, n + 0.15, clab, ha="center", va="bottom", color=cc, fontsize=9.8, fontweight="bold")
for r, iid in enumerate(ids):
    y = n - 1 - r
    bn, bh = grid["base_native"][iid], grid["base_hermes"][iid]
    qn, qh = grid["qwopus_native"][iid], grid["qwopus_hermes"][iid]
    for outcome, cx in ((bn, XS[0]), (bh, XS[1]), (qn, XS[2]), (qh, XS[3])):
        ax2.add_patch(FancyBboxPatch((cx - cw / 2, y - ch / 2), cw, ch,
                                     boxstyle="round,pad=0.0,rounding_size=0.09",
                                     fc=FILL[outcome], ec=EDGE[outcome], lw=1.5))
        ax2.text(cx, y, GLYPH[outcome], ha="center", va="center", color=TXTC[outcome], fontsize=7.5, fontweight="bold")
    if qn != qh and qh == "R":
        ax2.add_patch(Rectangle((XS[2] - cw / 2 - 0.07, y - ch / 2 - 0.06), (XS[3] - XS[2]) + cw + 0.14, ch + 0.12,
                                fill=False, ec=CRIMSON, lw=2.0))
    if bn != bh and bh == "R":
        ax2.add_patch(Rectangle((XS[0] - cw / 2 - 0.07, y - ch / 2 - 0.06), (XS[1] - XS[0]) + cw + 0.14, ch + 0.12,
                                fill=False, ec=GOLD, lw=1.6))
    ax2.text(-0.85, y, iid.split("__")[1], ha="right", va="center", color=TEXT, fontsize=8.6)
ax2.axvline((XS[1] + XS[2]) / 2, color=GRID, lw=1.2, ls=":")
ax2.annotate("Qwopus recovers\ndjango + matplotlib\nunder Hermes",
             xy=(XS[3] + cw / 2, n - 1 - 1), xytext=(XS[3] + 0.65, n - 2.3),
             color=CRIMSON, fontsize=8.4, va="center", ha="left", fontweight="bold",
             arrowprops=dict(arrowstyle="-", color=CRIMSON, lw=1.1))
ax2.text(XS[0] + (XS[1] - XS[0]) / 2, -1.35, "base:  7 -> 8", ha="center", va="top", color=GOLD, fontsize=9.5, fontweight="bold")
ax2.text(XS[2] + (XS[3] - XS[2]) / 2, -1.35, "Qwopus:  5 -> 7\n(still a strict subset of base)", ha="center", va="top", color=AMBER, fontsize=9.5, fontweight="bold")
ax2.set_xlim(-3.0, 6.8)
ax2.set_ylim(-2.6, n + 1.0)
ax2.axis("off")

fig.text(0.012, 0.012, "n=12, bounded subset shared with the t082 Ornith harness study; Hermes-as-harness is new infra (first run this session), not a replication of omp's specific leniency profile.",
         color=MUTE, fontsize=7.5, ha="left")
fig.text(0.99, 0.012, "WITCHEER", color=MUTE, fontsize=9, ha="right", fontweight="bold")
fig.subplots_adjust(left=0.085, right=0.99, top=0.9, bottom=0.075, wspace=0.18)
fig.savefig("reports/qwopus-coder-35b-a3b.png", dpi=150, facecolor=BG)
print("wrote reports/qwopus-coder-35b-a3b.png")
