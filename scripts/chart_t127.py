#!/usr/bin/env python3
"""Gold & Crimson: what turning reasoning OFF costs LFM2.5-2.6B on agentic tool-use.

Dumbbell, one question: how far does each task tier fall when the model's thinking is
genuinely disabled. GOLD dot = reasoning ON, CRIMSON dot = reasoning OFF; the bar
between is the drop. Overall row is emphasized. Data = t127: 30 tasks/leg (10 per tier)
* 3 reps = 90 runs/leg, greedy, one RTX 5090. Rates are over all runs; the McNemar p in
the subtitle is over the 30 tasks (majority vote per task). The on/off A/B is only valid
because the off leg was served with a patched chat template that pre-closes the <think>
block (the shipped template hard-forced thinking and ignored the disable flag).

Usage: python3 scripts/chart_t127.py  (writes reports/chart_t127.png)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, GOLD, CRIMSON = "#0d0906", "#e8c44a", "#e06060"
TEXT, GRID, MUTE = "#f5e6d0", "#3a2f25", "#8a7a64"

# (label, reasoning_on %, reasoning_off %, emphasize)
ROWS = [
    ("overall", 96.7, 70.0, True),
    ("single-tool", 100.0, 60.0, False),
    ("multi-step", 100.0, 90.0, False),
    ("multi-turn instr.", 90.0, 60.0, False),
]

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": TEXT,
    "ytick.color": TEXT, "axes.edgecolor": GRID, "font.size": 12,
})

fig, ax = plt.subplots(figsize=(11.5, 6.4))
fig.subplots_adjust(top=0.78, bottom=0.13, left=0.16, right=0.955)

ys = list(range(len(ROWS)))[::-1]  # first row on top
for y, (label, on, off, emph) in zip(ys, ROWS):
    lw = 6 if emph else 3
    ax.plot([off, on], [y, y], color=MUTE, lw=lw, solid_capstyle="round", zorder=2)
    ax.plot([on], [y], marker="o", markersize=15 if emph else 12, color=GOLD,
            mec=BG, mew=1.5, zorder=4)
    ax.plot([off], [y], marker="o", markersize=15 if emph else 12, color=CRIMSON,
            mec=BG, mew=1.5, zorder=4)
    # value labels
    ax.annotate(f"{on:.0f}" if on == int(on) else f"{on:.1f}", (on, y),
                textcoords="offset points", xytext=(10, 0), va="center", ha="left",
                fontsize=11, color=GOLD, fontweight="bold" if emph else "normal")
    ax.annotate(f"{off:.0f}" if off == int(off) else f"{off:.1f}", (off, y),
                textcoords="offset points", xytext=(-10, 0), va="center", ha="right",
                fontsize=11, color=CRIMSON, fontweight="bold" if emph else "normal")
    # delta on the bar
    ax.annotate(f"-{on - off:.1f}".rstrip("0").rstrip("."), ((on + off) / 2, y),
                textcoords="offset points", xytext=(0, 11 if emph else 9), va="bottom",
                ha="center", fontsize=10, color=MUTE)

ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in ROWS], fontsize=12,
                   color=TEXT)
# bold the "overall" tick label
for lbl, (_, _, _, emph) in zip(ax.get_yticklabels(), ROWS):
    if emph:
        lbl.set_fontweight("bold")

ax.set_xlim(50, 108)
ax.set_ylim(-0.6, len(ROWS) - 0.4)
ax.grid(True, axis="x", color=GRID, lw=0.8, zorder=0)
ax.tick_params(length=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.set_xlabel("task success rate  (%, over 90 runs/leg · 30 per tier)", color=MUTE, fontsize=11.5)

# legend as colored words in the subtitle
fig.suptitle("Turn reasoning off and this 2.6B model falls from best to worst",
             fontsize=15, color=TEXT, x=0.16, ha="left")
fig.text(0.16, 0.885,
         "LFM2.5-2.6B agentic tool-use, one RTX 5090 · overall 96.7 to 70.0, "
         "McNemar p=0.008 (8/30 tasks flipped, 0 the other way)",
         fontsize=9.5, color=MUTE)
fig.text(0.16, 0.85, "reasoning ON", fontsize=10.5, color=GOLD, fontweight="bold")
fig.text(0.285, 0.85, "vs", fontsize=10.5, color=MUTE)
fig.text(0.315, 0.85, "reasoning OFF", fontsize=10.5, color=CRIMSON, fontweight="bold")
fig.text(0.44, 0.85, "· off only genuine after patching a template that ignored the toggle",
         fontsize=9.5, color=MUTE)

fig.savefig("reports/chart_t127.png", dpi=170)
print("wrote reports/chart_t127.png")
