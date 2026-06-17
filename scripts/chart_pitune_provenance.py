"""Gold & Crimson chart for Qwen3.6-27B pi-tune: the synthetic agentic score is a narrow band while
real SWE-bench resolve spreads wide — and training-data provenance, not the 'agentic' label, tracks
real capability. Scatter: agentic (x) vs SWE-bench resolved/30 (y), colored by provenance. pi-tune
(real traces) is the only tune that improves. Runs on the Mac (system python3 matplotlib)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = ("#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64")

# (label, agentic, resolved/30, color, marker_size)
pts = [
    ("Qwen3.6-27B base",                 98.61, 19, MUTE,    260),
    ("pi-tune  (real agent traces)",     98.01, 20, GOLD,    540),
    ("Qwopus-Coder  (Hermes distill)",   100.0, 17, CRIMSON, 300),
    ("Qwable-3.6-27b  (Fable-5 distill)", 97.64, 11, CRIMSON, 300),
]

fig, ax = plt.subplots(figsize=(11, 7), facecolor=BG)
ax.set_facecolor(BG)

# the synthetic-agentic cluster: all four within a ~2.4pt band
ax.axvspan(97.4, 100.2, color=GOLD, alpha=0.06, zorder=0)
ax.axhline(19, color=MUTE, lw=1, ls="--", alpha=0.5, zorder=1)
ax.annotate("base = 19/30", (96.95, 19), color=MUTE, fontsize=9, va="bottom", ha="left",
            xytext=(0, 3), textcoords="offset points")

for label, ag, sw, color, sz in pts:
    ax.scatter([ag], [sw], s=sz, color=color, edgecolor=TEXT, linewidth=1.3, zorder=4)
    big = color == GOLD
    ax.annotate(label, (ag, sw), color=(GOLD if big else TEXT), fontsize=(11.5 if big else 9.5),
                fontweight=("bold" if big else "normal"), ha="center", va="bottom",
                xytext=(0, 15 if big else 12), textcoords="offset points")

ax.set_xlabel("synthetic Agentic Score  (native tool-calling, 40 tasks)", color=TEXT)
ax.set_ylabel("real SWE-bench Verified resolved  (of 30)", color=TEXT)
ax.set_xlim(96.8, 101.0)
ax.set_ylim(8, 22.5)
ax.set_title("the synthetic score is a narrow band. real bugs are not.", color=GOLD, fontsize=14, pad=18)
ax.text(0.5, 1.015,
        "Qwen3.6-27B + 3 coding tunes · agentic clusters 97.6-100 (a 2.4pt band) while real SWE spans 11-20/30 · color = training-data provenance",
        transform=ax.transAxes, color=MUTE, fontsize=9, ha="center")
ax.tick_params(colors=TEXT)
for sp in ax.spines.values():
    sp.set_color(GRID)
ax.grid(True, color=GRID, alpha=0.35)
leg = [Line2D([0], [0], marker="o", color="none", markerfacecolor=GOLD, markeredgecolor=TEXT, markersize=12, label="real agent traces (improves)"),
       Line2D([0], [0], marker="o", color="none", markerfacecolor=CRIMSON, markeredgecolor=TEXT, markersize=11, label="synthetic distill (regresses)"),
       Line2D([0], [0], marker="o", color="none", markerfacecolor=MUTE, markeredgecolor=TEXT, markersize=11, label="untuned base")]
ax.legend(handles=leg, loc="lower right", facecolor=BG, edgecolor=GRID, fontsize=9, labelcolor=TEXT)
fig.tight_layout()
fig.savefig("reports/pi-tune-provenance.png", dpi=150, facecolor=BG)
print("wrote reports/pi-tune-provenance.png")
