#!/usr/bin/env python3
"""Gold & Crimson card: "The Agent Knew" (arXiv 2610.17301) reproduced for gpt-oss-20b on one RTX 5090.

Dot plot, AUROC for flagging an unauthorized action at decision time (after the reasoning, before the tool call),
fresh confirmation units only. Rows: all actions, the obfuscate variant, the subset the paper's CoT monitor scored.
Markers: our activation readout, the supervised text classifier, the paper's CoT monitor.

Usage: python3 scripts/chart_agent_knew_gptoss.py   (reads data/agent-knew/compare.json + ours_v57_summary.json)
"""
import json
from decimal import Decimal, ROUND_HALF_UP

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def f3(v):
    return str(Decimal(str(v)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


BG, GOLD, CRIMSON = "#0d0906", "#e8c44a", "#e06060"
TEXT, GRID, MUTE = "#f5e6d0", "#3a2f25", "#8a7a64"

C = json.load(open("data/agent-knew/compare.json"))
S = json.load(open("data/agent-knew/ours_v57_summary.json"))
ours = C["conf_per_model_gptoss20b"]["ours"]["all"]
obf = C["ours_obfuscate_gptoss"]
mon = S["monitor_subset_auroc"]

# label, n line, readout, text, cot (None = not scored on that row)
rows = [
    ("all fresh actions", "8,416 actions, 710 rule breaks", ours["I"], ours["T_sup"], None),
    ("agent told to keep\npermissions out of its reasoning", "2,470 actions, 219 rule breaks", obf["I"], obf["T_sup"], None),
    ("subset the CoT monitor read", "894 actions, 298 rule breaks", mon["I"], mon["T_sup"], mon["M_cot"]),
]

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": TEXT,
    "ytick.color": TEXT, "axes.edgecolor": GRID, "font.size": 13,
})
fig, ax = plt.subplots(figsize=(12, 6.75))
fig.subplots_adjust(left=0.30, right=0.96, top=0.74, bottom=0.17)

ys = [2, 1, 0]
for y, (lab, nline, i, t, m) in zip(ys, rows):
    vals = [v for v in (i, t, m) if v is not None]
    ax.plot([min(vals), max(vals)], [y, y], color=GRID, lw=3, zorder=1)
    ax.scatter([i], [y], s=260, color=CRIMSON, zorder=3)
    ax.scatter([t], [y], s=260, color=GOLD, zorder=3)
    if m is not None:
        ax.scatter([m], [y], s=260, color=TEXT, zorder=3)
    # value labels above each dot; nudge apart when two values sit close
    pts = sorted([(i, CRIMSON), (t, GOLD)] + ([(m, TEXT)] if m is not None else []))
    last = -1
    for v, col in pts:
        x = v if v - last > 0.017 else last + 0.017
        ax.text(x, y + 0.30, f3(v), ha="center", va="bottom", color=col, fontsize=13, fontweight="bold")
        last = x
    ax.text(0.742, y - 0.30, nline, ha="left", va="center", color=MUTE, fontsize=11)

ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in rows], fontsize=13)
ax.set_xlim(0.74, 0.93)
ax.set_ylim(-0.6, 2.6)
ax.set_xlabel("AUROC (1.0 = flags every rule break first, 0.5 = coin flip)", fontsize=12, color=TEXT)
ax.grid(axis="x", color=GRID, lw=0.8)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)

fig.text(0.03, 0.93, "Reading gpt-oss-20b's activations vs reading its words", fontsize=21, fontweight="bold", color=TEXT, va="top")
fig.text(0.03, 0.855, "spotting an unauthorized action before the tool call is written, on held-out episodes", fontsize=13, color=TEXT, va="top")
fig.text(0.03, 0.81, f"\"The Agent Knew\" (arXiv 2610.17301) replayed on one RTX 5090: readout {f3(ours['I'])} overall, the paper's {f3(C['conf_per_model_gptoss20b']['paper']['all']['I'])}",
         fontsize=13, color=MUTE, va="top")
fig.add_artist(plt.Rectangle((0.0, 0.905), 0.012, 0.03, transform=fig.transFigure, color=CRIMSON))

# legend below the axes
for k, (col, lab) in enumerate([(CRIMSON, "activation readout (ours)"), (GOLD, "text classifier, same moment"), (TEXT, "the paper's CoT monitor")]):
    x0 = 0.30 + k * 0.23
    fig.add_artist(plt.Circle((x0, 0.065), 0.008, transform=fig.transFigure, color=col))
    fig.text(x0 + 0.015, 0.065, lab, fontsize=12, color=TEXT, va="center")
fig.text(0.96, 0.02, "WITCHEER · RTX 5090", fontsize=10, color=MUTE, ha="right")

fig.savefig("reports/agent-knew-gptoss20b.png", dpi=150)
print("wrote reports/agent-knew-gptoss20b.png")
