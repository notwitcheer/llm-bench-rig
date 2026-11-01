#!/usr/bin/env python3
"""chart_pheromone_replay.py: dot plot of every pheromone-replay cell (solve rate + 95% Wilson interval),
with the paper's number beside the cells where arXiv 2601.08129 reports one.
usage: chart_pheromone_replay.py data/pheromone-replay/summary.json reports/pheromone-replay.png"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

S = {r["cell"]: r for r in json.load(open(sys.argv[1]))}
# (label, cell key, paper value in % or None, accent)
ROWS = [
    ("small models, easy", None, None, False),
    ("fading marks, 4 agents", "pe-pf4", 83.3, False),
    ("fading marks, 1 agent", "pe-pf1", 83.3, False),
    ("agents that chat", "pe-cv4", 33.3, False),
    ("fading ON (ablation)", "pe-decay:full", 96.7, True),
    ("fading OFF (ablation)", "pe-decay:no_decay", 86.7, True),
    ("small models, medium", None, None, False),
    ("fading marks, 4 agents ", "pm-pf4", 43.3, False),
    ("agents that chat ", "pm-cv4", 0.0, False),
    ("small models, hard", None, None, False),
    ("fading marks, 4 agents  ", "ph-pf4", 15.6, False),
    ("agents that chat  ", "ph-cv4", 0.0, False),
    ("Ornith 1.5 35B-A3B, easy", None, None, False),
    ("fading marks, 4 agents   ", "me-pf4", None, False),
    ("fading marks, 1 agent ", "me-pf1", None, False),
    ("agents that chat   ", "me-cv4", None, False),
    ("chat, same token budget", "me-cvM", None, False),
    ("fading ON (ablation) ", "me-decay:full", None, False),
    ("fading OFF (ablation) ", "me-decay:no_decay", None, False),
    ("Ornith 1.5 35B-A3B, medium", None, None, False),
    ("fading marks, 4 agents    ", "mm-pf4", None, False),
    ("agents that chat    ", "mm-cv4", None, False),
    ("chat, same token budget ", "mm-cvM", None, False),
]
ACC, INK, MUTED, PAPER = "#d9480f", "#222222", "#888888", "#1c7ed6"
fig, ax = plt.subplots(figsize=(10, 9.6), dpi=160)
y = 0
yt, yl, ystyle = [], [], []
for label, key, paper, acc in ROWS:
    if key is None:
        if yt:
            y -= 0.4
        yt.append(y); yl.append(label); ystyle.append("head")
        y -= 1
        continue
    r = S[key]
    p, lo, hi = r["rate"] * 100, r["ci95"][0] * 100, r["ci95"][1] * 100
    col = ACC if acc else INK
    ax.plot([lo, hi], [y, y], color=col, lw=2.2, solid_capstyle="round", alpha=0.85)
    ax.plot([p], [y], "o", color=col, ms=7.5, zorder=3)
    ax.text(101.5, y, f"{r['solved']}/{r['n']}", fontsize=9, color=col, va="center", ha="left")
    if paper is not None:
        ax.plot([paper], [y], "D", mfc="white", mec=PAPER, mew=1.8, ms=7, zorder=4)
    yt.append(y); yl.append("  " + label.strip()); ystyle.append("acc" if acc else "row")
    y -= 1
ax.set_yticks(yt)
ax.set_yticklabels(yl, fontsize=9.5)
for t, st in zip(ax.get_yticklabels(), ystyle):
    if st == "head":
        t.set_fontweight("bold"); t.set_color(INK)
    elif st == "acc":
        t.set_color(ACC)
ax.set_xlim(-2, 100)
ax.set_ylim(y + 0.3, 0.6)
ax.set_xlabel("problems solved (%), dot = our rate, bar = 95% interval", fontsize=9.5, color=INK)
ax.grid(axis="x", color="#e6e6e6", lw=0.8)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.plot([], [], "o", color=INK, label="ours, one RTX 5090")
ax.plot([], [], "D", mfc="white", mec=PAPER, mew=1.8, label="the paper (arXiv 2601.08129)")
ax.legend(loc="lower right", fontsize=9, frameon=False)
fig.suptitle("agents that coordinate only through fading marks, re-run on one RTX 5090", fontsize=12.5, fontweight="bold", x=0.02, ha="left", y=0.985)
fig.text(0.02, 0.952, "meeting-room scheduling, the paper's own code (Govcraft/pressure-field-experiment @ d3b0d5b). small models = Qwen2.5 0.5b > 1.5b > 3b on Ollama.",
         fontsize=8.6, color=MUTED, ha="left")
fig.text(0.02, 0.934, "paper diamonds: its tables 4, 6 and 8. table 8 (medium, hard, and the chat rows) pools 1, 2 and 4 agents; the ablation does not state its agent count.",
         fontsize=8.6, color=MUTED, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.925))
fig.savefig(sys.argv[2], facecolor="white")
print("wrote", sys.argv[2])
