#!/usr/bin/env python3
"""Gold & Crimson scatter: generation throughput vs 5-task quality average.
Think-off field on one RTX 5090. Numbers from results/<slug>/{speed,quality}.json.
Story: Nemotron-3-Nano (Mamba hybrid) is fastest by ~2 t/s over gpt-oss-20B but
trails it by ~5 quality points -> linear-attention speed, not a value win."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BG="#0d0906"; GOLD="#e8c44a"; TEXT="#f5e6d0"; CRIMSON="#e06060"; GRID="#3a2f25"; MUTE="#8a7a64"

# (label, gen_tok_s, q_avg, kind, (dx,dy,ha))  kind: head|side|mute
DATA = [
    ("Nemotron-3-Nano-30B-A3B  (Mamba hybrid)", 369.6, 82.2, "head", (-12, 10, "right")),
    ("gpt-oss-20B",                             367.4, 87.4, "side", (-14, 4, "right")),
    ("Nemotron-Cascade-2-30B-A3B",              350.8, 81.6, "mute", (-12, -18, "right")),
    ("Qwen3.6-35B-A3B",                         270.6, 93.3, "mute", (10, 6, "left")),
    ("Qwen3.6-27B (dense)",                      61.9, 94.1, "mute", (10, 6, "left")),
    ("Gemma-4-31B (dense)",                      53.0, 94.2, "mute", (10, -18, "left")),
]

fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
fig.subplots_adjust(left=0.085, right=0.96, top=0.76, bottom=0.13)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0, labelsize=14, colors=TEXT)
ax.grid(color=GRID, linewidth=1, alpha=0.6); ax.set_axisbelow(True)

# the vertical gap at ~same speed = the whole story
ax.plot([368.5, 368.5], [82.2, 87.4], color=CRIMSON, lw=1.4, ls="--", alpha=0.7, zorder=2)
ax.annotate("+5.2 q_avg\nsame speed", (368.5, 84.8), xytext=(-14, 0),
            textcoords="offset points", ha="right", va="center",
            color=CRIMSON, fontsize=12, fontweight="bold")

for label, ts, q, kind, off in DATA:
    if kind == "head":   c, sz, z = CRIMSON, 380, 5
    elif kind == "side": c, sz, z = GOLD, 360, 5
    else:                c, sz, z = MUTE, 200, 3
    ax.scatter(ts, q, s=sz, color=c, zorder=z, edgecolors=BG, linewidths=1.5)
    big = kind in ("head", "side")
    dx, dy, ha = off
    ax.annotate(label, (ts, q), xytext=(dx, dy), textcoords="offset points",
                ha=ha, va="bottom", color=(TEXT if big else MUTE),
                fontsize=(13 if big else 11), fontweight=("bold" if big else "normal"))

ax.set_xlim(30, 405); ax.set_ylim(79, 97)
ax.set_xlabel("Generation throughput  (tokens/s, tg128, -ngl 99)", fontsize=14, color=TEXT)
ax.set_ylabel("Quality  (5-task average, %)", fontsize=14, color=TEXT)
ax.text(180, 79.6, "faster ->", color=MUTE, fontsize=11)

fig.add_artist(Rectangle((0.085, 0.91), 0.07, 0.012, color=CRIMSON, transform=fig.transFigure))
fig.text(0.085, 0.85, "Fastest by 2 t/s — and that is all the Mamba hybrid wins", fontsize=21, fontweight="bold", color=TEXT)
fig.text(0.085, 0.805, "Generation throughput vs 5-task quality average  ·  one RTX 5090  ·  reasoning off", fontsize=12.5, color=GOLD)
fig.text(0.96, 0.045, "WITCHEER · RTX 5090", fontsize=12, color=MUTE, ha="right")
fig.savefig("/home/witcheer/benchmark-rig/reports/chart-nemotron-speed-vs-quality.png", facecolor=BG)
print("wrote reports/chart-nemotron-speed-vs-quality.png")
