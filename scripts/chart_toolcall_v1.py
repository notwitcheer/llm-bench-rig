#!/usr/bin/env python3
"""toolcall-v1 card (2026-10-25): left, S5 multi-turn agent score (BFCL v4 multi_turn_base, 100 entries, think-on), the metric
that separates the six Qwen3.8-27B-family files; right, single-turn S1..S4 score (906 scored items), thinking on (filled dot) and
thinking off (ring). Reads dataset/toolcall_v1.csv (scripts/toolcall_v1_table.py). The paired-McNemar labels beside the bars are the
S5 discordant counts against Q6_K from the same script's output (S5_PAIRS below).

    python3 scripts/chart_toolcall_v1.py [out.png]   # matplotlib; the rig's charts venv
"""
import csv
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "dataset", "toolcall_v1.csv")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "reports", "toolcall-v1.png")
BG, TEXT, MUTE, GRID = "#0e1420", "#eaf0f7", "#7d8ba0", "#20293a"
SUBJ, BASE, TERN, GSQC, WARN = "#e8b64c", "#6fd6a8", "#7aa7e8", "#ff8c42", "#ff6b6b"
ROWS = [  # slug, label, colour
    ("qwen3-8-27b-q6-k", "Qwen3.8-27B · Q6_K · {gb} GB", BASE),
    ("qwen3-8-27b-ud-iq3-xxs", "Qwen3.8-27B · UD-IQ3_XXS · {gb} GB", BASE),
    ("qwen3-8-27b-gsq-rco-iq2-xs", "Qwen3.8-27B · GSQ-RCO IQ2_XS · {gb} GB", GSQC),
    ("ternary-bonsai-2-27b-pq2-0", "Ternary-Bonsai-2-27B · PQ2_0 · {gb} GB †", TERN),
    ("ternary-bonsai-2-27b-ptq1-0", "Ternary-Bonsai-2-27B · PTQ1_0 · {gb} GB †", TERN),
    ("underdog-saluki-27b-iq2-mix", "Underdog-Saluki-27B · IQ2-mix · {gb} GB", SUBJ),
]
# S5 discordant items against Q6_K (Q6_K right only / row right only), exact McNemar p
S5_PAIRS = {
    "qwen3-8-27b-ud-iq3-xxs": (5, 4, 1.0), "qwen3-8-27b-gsq-rco-iq2-xs": (9, 4, 0.267), "ternary-bonsai-2-27b-pq2-0": (10, 2, 0.039),
    "ternary-bonsai-2-27b-ptq1-0": (11, 3, 0.057), "underdog-saluki-27b-iq2-mix": (15, 5, 0.041),
}
d = {}
for r in csv.DictReader(open(CSV, newline="")):
    d[(r["slug"], r["leg"])] = r
s5 = {s: int(d[(s, "s5-on")]["s5_correct"]) for s, _, _ in ROWS}
on = {s: float(d[(s, "st-on")]["single_turn_pct"]) for s, _, _ in ROWS}
off = {s: float(d[(s, "st-off")]["single_turn_pct"]) for s, _, _ in ROWS}
gb = {s: f"{int(d[(s, 'st-on')]['file_bytes']) / 1e9:.1f}" for s, _, _ in ROWS}
rows = sorted(ROWS, key=lambda r: -s5[r[0]])  # stable: PQ2_0 before PTQ1_0 on the 73 tie
assert [s5[r[0]] for r in rows] == [81, 80, 76, 73, 73, 71], [s5[r[0]] for r in rows]

fig = plt.figure(figsize=(16, 7.6), dpi=110)
fig.patch.set_facecolor(BG)
gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 0.88], wspace=0.17, left=0.245, right=0.975, top=0.70, bottom=0.14)
ax = fig.add_subplot(gs[0])
bx = fig.add_subplot(gs[1], sharey=ax)
for a in (ax, bx):
    a.set_facecolor(BG)
    for sp in a.spines.values():
        sp.set_visible(False)
    a.tick_params(length=0, colors=MUTE)
    a.grid(axis="x", color=GRID, lw=1, zorder=0)
n = len(rows)
for i, (s, lab, col) in enumerate(rows):
    y = n - 1 - i
    ax.barh(y, s5[s], color=col, height=0.62, zorder=3)
    ax.text(s5[s] - 1.2, y, f"{s5[s]}", ha="right", va="center", color=BG, fontsize=17, weight="bold", zorder=4)
    if s == "qwen3-8-27b-q6-k":
        note, nc = "reference", MUTE
    else:
        a_, b_, p_ = S5_PAIRS[s]
        note, nc = f"{a_} / {b_} vs Q6_K, p = {p_:.3f}".replace("1.000", "1.0"), (TEXT if p_ >= 0.5 else MUTE)
    if s == "qwen3-8-27b-ud-iq3-xxs":
        ax.text(s5[s] + 1.0, y + 0.17, note, ha="left", va="center", color=nc, fontsize=10.5)
        ax.text(s5[s] + 1.0, y - 0.19, "≈ Q6_K at half the size", ha="left", va="center", color=BASE, fontsize=11.5, weight="bold")
    else:
        ax.text(s5[s] + 1.0, y, note, ha="left", va="center", color=nc, fontsize=10.5)
    ax.text(-2.0, y, lab.format(gb=gb[s]), ha="right", va="center", color=TEXT, fontsize=11.8)
ax.set_xlim(0, 118)
ax.set_ylim(-0.6, n - 0.4)
ax.set_yticks([])
ax.set_xticks([0, 25, 50, 75, 100])
ax.tick_params(axis="x", labelsize=10)
ax.text(0, n - 0.1, "multi-turn agent test: tasks solved of 100\n(BFCL v4 multi_turn_base, thinking on)", ha="left", va="bottom",
        color=TEXT, fontsize=12.5)

for i, (s, lab, col) in enumerate(rows):
    y = n - 1 - i
    bx.plot([min(on[s], off[s]), max(on[s], off[s])], [y, y], color=GRID if abs(on[s] - off[s]) < 5 else WARN, lw=3, zorder=2,
            solid_capstyle="round")
    bx.scatter([on[s]], [y], s=190, color=col, zorder=4, edgecolor=BG, lw=1.5)
    bx.scatter([off[s]], [y], s=190, facecolor=BG, edgecolor=(WARN if off[s] < 80 else col), lw=2.6, zorder=4)
    hi = max(on[s], off[s])
    bx.text(hi + 1.4, y, f"{on[s]:.1f} on · {off[s]:.1f} off", ha="left", va="center", color=TEXT, fontsize=10.5)
    if off[s] < 80:
        bx.text(off[s], y + 0.36, "thinking off", ha="center", va="bottom", color=WARN, fontsize=10.5, weight="bold")
bx.set_xlim(70, 104)
bx.set_ylim(-0.7, n - 0.4)
bx.set_xticks([75, 80, 85, 90, 95])
bx.tick_params(axis="x", labelsize=10)
plt.setp(bx.get_yticklabels(), visible=False)
bx.text(70, n - 0.1, "single-turn tool calls: % correct of 906\n● thinking on   ○ thinking off", ha="left", va="bottom", color=TEXT, fontsize=12.5)

fig.text(0.02, 0.935, "Single-turn tool calls score 86 to 90 on every file. The multi-turn agent test spreads them 71 to 81",
         color=TEXT, fontsize=18.5, weight="bold")
fig.text(0.02, 0.895, "same base model (Qwen3.8-27B), one RTX 5090, greedy, one run per leg. UD-IQ3_XXS is level with Q6_K on the agent test (5 / 4 discordant tasks); "
         "Saluki and PQ2_0 sit 10 and 8 points under it\n(p = 0.04 each, unadjusted, five comparisons). GSQ-RCO is the only file that breaks with thinking off "
         "(76.6; set S1 271 of 439 against 387 with thinking on). BFCL v4 frozen slice, not the leaderboard.\n"
         "llama.cpp b10371 for the Qwen3.8 rows · † Bonsai rows on the PrismML llama.cpp fork (its packings are not in mainline) · Saluki: 1 of 100 agent tasks force-terminated, scored wrong.",
         color=MUTE, fontsize=10.6, va="top")
fig.text(0.02, 0.025, "witcheer · llm-bench-rig dataset/toolcall_v1.csv + reports/toolcall-v1.md · sizes are the GGUF on disk · "
         "all 18 legs run 2026-10-24 · pairs: exact McNemar on the same 100 tasks", color=MUTE, fontsize=9.5)
fig.savefig(OUT, facecolor=BG)
print(OUT, s5)
