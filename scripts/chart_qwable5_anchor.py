"""Qwable-5 EV+ chart: the same-base Qwen3.6-27B coding-tune cohort on two axes —
synthetic Agentic Score (x, a razor-thin 97.6-100 band) vs real SWE-bench Verified
resolve (y, spread 11-20). Qwable-5 sits at the base's EXACT agentic score (98.61)
yet two bugs lower: a vertical drop that is the whole finding.
matplotlib lives on system python3 (not .venv): `python3 scripts/chart_qwable5_anchor.py`."""
import matplotlib.pyplot as plt

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = ("#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64")

# slug -> (display, agentic_score, resolved/30, color, label-offset (dx,dy))
PTS = [
    ("pi-tune\n(real terminal/repo traces)", 98.01, 20, GOLD,    (0,  14)),
    ("Qwen3.6-27B base",                     98.61, 19, "#c9a86a", (-10, 14)),
    ("Qwable-5\n(Fable-5 + Kimi traces)",    98.61, 17, CRIMSON, (0, -34)),
    ("Qwopus-Coder\n(Hermes traces)",       100.00, 17, MUTE,    (0,  14)),
    ("Qwable-3.6\n(Fable-5 distill)",        97.64, 11, MUTE,    (0, -26)),
]

fig, ax = plt.subplots(figsize=(9.6, 6.6), facecolor=BG)
ax.set_facecolor(BG)

# the finding: base -> Qwable-5 vertical drop at the SAME x (98.61)
ax.annotate("", xy=(98.61, 17.12), xytext=(98.61, 18.88),
            arrowprops=dict(arrowstyle="-|>", color=CRIMSON, lw=2.0))
ax.text(98.72, 18.0, "same agentic score,\n2 fewer real bugs", color=CRIMSON,
        fontsize=9.5, va="center", ha="left")

for disp, x, y, color, (dx, dy) in PTS:
    big = color in (CRIMSON, "#c9a86a", GOLD)
    ax.scatter([x], [y], s=360 if color == CRIMSON else 300, color=color,
               edgecolor=TEXT, zorder=4, linewidth=1.3)
    ax.annotate(disp, (x, y), color=TEXT if big else MUTE, fontsize=9.3,
                ha="center", va="center", xytext=dx and (dx, dy) or (0, dy),
                textcoords="offset points")

# the band the cheap axis lives in
ax.axvspan(97.64, 100.0, color=GOLD, alpha=0.05, zorder=0)
ax.text(98.82, 11.6, "synthetic band: 97.6 to 100\n(2.4 pts) — blind to the spread",
        color=MUTE, fontsize=9, ha="center")

ax.set_xlabel("synthetic Agentic Score (40-task tool-calling harness)", color=TEXT)
ax.set_ylabel("real SWE-bench Verified resolved (of 30)", color=TEXT)
ax.set_xlim(97.2, 100.7)
ax.set_ylim(9.5, 21.5)
ax.set_title("Same base, same cheap scores — different real bugs\n"
             "5 Qwen3.6-27B coding tunes · RTX 5090 · matched quant · think-OFF",
             color=GOLD, pad=14, fontsize=12.5)
for s in ax.spines.values():
    s.set_color(GRID)
ax.tick_params(colors=MUTE)
ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
fig.text(0.985, 0.015, "WITCHEER", color=MUTE, fontsize=9, ha="right", va="bottom")
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig("reports/qwable-5-anchor.png", dpi=150, facecolor=BG)
print("wrote reports/qwable-5-anchor.png")
