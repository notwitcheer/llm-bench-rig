#!/usr/bin/env python3
"""Nanbeige4.3-33B card (2026-11-02): Nanbeige Q4_K_M against the pre-registered Qwen3.8-27B comparator, one row per test.

Every number is copied from scripts/capsule-queue/nb_readout.py output (readout of 2026-11-02 06:08 CET; reports/nanbeige4-3-33b.md
carries the same table). Verdicts are the pre-registered rules: toolcall S5 / GPQA / AIME item-level or margin ties, agenttest lead under 3 of
30 tasks is a tie (here the lead is 4), board q_avg margin 1.2.

    python3 scripts/chart_nanbeige43.py [out.png]   # matplotlib; the rig's charts venv
"""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "reports", "nanbeige4-3-33b.png")
BG, PANEL, TEXT, MUTE, GRID = "#0e1420", "#151d2d", "#eaf0f7", "#7d8ba0", "#20293a"
NB, QW, TIE, LOSE = "#e8b64c", "#6fd6a8", "#7aa7e8", "#ff6b6b"

# (test, unit line, nanbeige, qwen, verdict, verdict colour, comparator + detail)
ROWS = [
    ("multi-turn tool calls", "tasks solved of 100", "82", "81", "TIE", TIE, "Qwen Q6_K file · paired p = 1.0"),
    ("one-step tool calls", "correct of 906", "800", "799", "TIE", TIE, "Qwen Q6_K file · paired p = 0.91"),
    ("30 Hermes Agent tasks", "passed at 64k context", "25", "29", "QWEN AHEAD", LOSE, "Qwen UD-Q4_K_M · paired p = 0.125 · Nanbeige needs a compressed cache"),
    ("GPQA Diamond", "thinking on, 16k cap, %", "81.3", "79.3", "TIE", TIE, "Qwen Q6_K file · paired p = 0.54 · Nanbeige capped on 35 of 198"),
    ("AIME 2026", "thinking on, 32k cap, of 30", "26", "25", "TIE", TIE, "Qwen Q6_K file · paired p = 1.0"),
    ("five-task board", "thinking off, q_avg", "87.9", "93.2", "QWEN AHEAD", LOSE, "Qwen Q4_K_M file · outside the 1.2-point margin"),
    ("decode speed", "tg128, tokens per second", "71.7", "78.9", "QWEN FASTER", LOSE, "Qwen Q4_K_M file · llama-bench"),
]

fig = plt.figure(figsize=(16, 9.4), dpi=110)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 160)
ax.set_ylim(0, 94)
ax.axis("off")

fig.text(0.02, 0.945, "Nanbeige4.3-33B ties Qwen3.8-27B on tool calls and finishes fewer Hermes Agent tasks",
         color=TEXT, fontsize=19.5, weight="bold")
fig.text(0.02, 0.905, "Nanbeige Q4_K_M (20.8 GB) against Qwen3.8-27B rows on the same tests · one RTX 5090 · llama.cpp b10371 · greedy, except the Hermes tasks "
         "(sampled, one run each)", color=MUTE, fontsize=11.2)

# column heads
HEAD_Y = 80.2
ax.text(4, HEAD_Y, "test", color=MUTE, fontsize=11.5, va="center")
ax.text(66, HEAD_Y, "Nanbeige4.3-33B", color=NB, fontsize=13.5, weight="bold", ha="center", va="center")
ax.text(88, HEAD_Y, "Qwen3.8-27B", color=QW, fontsize=13.5, weight="bold", ha="center", va="center")
ax.text(102, HEAD_Y, "verdict (pre-registered rule)", color=MUTE, fontsize=11.5, va="center")

top, rh = 76.0, 8.05
for i, (test, unit, nb, qw, verdict, vcol, detail) in enumerate(ROWS):
    yc = top - i * rh - rh / 2 + 0.6
    ax.add_patch(FancyBboxPatch((1.2, yc - rh / 2 + 0.55), 157.6, rh - 1.1, boxstyle="round,pad=0,rounding_size=1.0",
                                facecolor=PANEL, edgecolor="none", zorder=1))
    ax.text(4, yc + 1.3, test, color=TEXT, fontsize=15, weight="bold", va="center", zorder=3)
    ax.text(4, yc - 1.9, unit, color=MUTE, fontsize=10.8, va="center", zorder=3)
    ax.text(66, yc, nb, color=NB, fontsize=25, weight="bold", ha="center", va="center", zorder=3)
    ax.text(88, yc, qw, color=QW, fontsize=25, weight="bold", ha="center", va="center", zorder=3)
    ax.add_patch(FancyBboxPatch((102, yc + 0.6), 17.5 if verdict != "QWEN AHEAD" and verdict != "QWEN FASTER" else 21, 3.1,
                                boxstyle="round,pad=0,rounding_size=1.2", facecolor=vcol, edgecolor="none", zorder=2))
    ax.text(102 + (17.5 if verdict not in ("QWEN AHEAD", "QWEN FASTER") else 21) / 2, yc + 2.15, verdict, color=BG, fontsize=11.5, weight="bold",
            ha="center", va="center", zorder=3)
    ax.text(102, yc - 1.6, detail, color=MUTE, fontsize=9.6, va="center", zorder=3)

# the fact the rows cannot show
ax.text(4, 11.6, "The 6-bit Nanbeige file (29.0 GB) ties its 4-bit file on multi-turn tool calls, 80 vs 82, and its context tops out at 41k on 32 GB.",
        color=TEXT, fontsize=13.2, weight="bold", va="center")
ax.text(4, 6.4, "ties need exact p ≥ 0.05 on the same items (McNemar) or a gap inside the pre-registered margin\n"
        "the Hermes task gap clears the 3-task margin: 4 of 4 discordant tasks go Qwen's way\n"
        "GPQA and AIME use our caps, greedy; the card's scores use other benchmarks at the highest thinking effort with no cap",
        color=MUTE, fontsize=10.2, va="center", linespacing=1.6)
fig.text(0.02, 0.02, "WITCHEER · RTX 5090 · llm-bench-rig reports/nanbeige4-3-33b.md", color=MUTE, fontsize=9.5)
ax.add_patch(plt.Rectangle((0, 91.5), 2.2, 2.5, color=NB, zorder=5))  # house accent tab

fig.savefig(OUT, facecolor=BG)
print(OUT)
