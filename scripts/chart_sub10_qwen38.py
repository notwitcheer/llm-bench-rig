#!/usr/bin/env python3
"""sub-10 GB Qwen3.8-27B card: five files of one base model on the second-tier think-on 16k board.
left: scores as a heatmap (rows sorted by the four-task mean). right: four-task mean vs GGUF size, with the
propagated Wilson 95% half-width as the error bar. data: dataset/second_tier.csv (this commit)."""
import csv
import math
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BG, TEXT, MUTE, GRID = "#0e1420", "#eaf0f7", "#7d8ba0", "#20293a"
SUBJ, BASE, TERN = "#e8b64c", "#6fd6a8", "#7aa7e8"
CSV = "/opt/data/repos/llm-bench-rig/dataset/second_tier.csv"
TASKS = [("ifeval", "IFEval\nprompt-strict"), ("math500", "MATH-500"), ("humaneval_plus", "HumanEval+"), ("mbpp_plus", "MBPP+")]
ROWS = {  # slug: (label, short label, colour)
    "qwen3-8-27b-q6-k": ("Qwen3.8-27B · Q6_K · 22.9 GB", "Q6_K", BASE),
    "qwen3-8-27b-ud-iq3-xxs": ("Qwen3.8-27B · UD-IQ3_XXS · 11.9 GB", "UD-IQ3_XXS", BASE),
    "ternary-bonsai-2-27b-pq2-0": ("Ternary-Bonsai-2-27B · PQ2_0 · 7.2 GB †", "Bonsai PQ2_0 †", TERN),
    "ternary-bonsai-2-27b-ptq1-0": ("Ternary-Bonsai-2-27B · PTQ1_0 · 5.9 GB †", "Bonsai PTQ1_0 †", TERN),
    "underdog-saluki-27b-iq2-mix": ("Underdog-Saluki-27B-1.0 · IQ2-mix · 7.9 GB", "Saluki IQ2-mix", SUBJ),
}
z = 1.96
def r1(v):  # half-up to one decimal, so 87.25 prints 87.3 like the report table (float repr would give 87.2)
    from decimal import Decimal, ROUND_HALF_UP
    return str(Decimal(str(round(v, 4))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
def whw(k, n):
    p = k / n
    return 100 * z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)

rows = {}
for r in csv.DictReader(open(CSV)):
    if r["slug"] in ROWS:
        rows.setdefault(r["slug"], {})[r["task"]] = r
assert len(rows) == 5 and all(len(v) == 4 for v in rows.values())
means = {s: statistics.mean(float(rows[s][t]["score"]) for t, _ in TASKS) for s in rows}
hws = {s: math.sqrt(sum(whw(int(rows[s][t]["correct"]), int(rows[s][t]["total"])) ** 2 for t, _ in TASKS)) / 4 for s in rows}
gb = {s: int(rows[s]["ifeval"]["file_bytes"]) / 1e9 for s in rows}
order = sorted(rows, key=lambda s: -means[s])

fig = plt.figure(figsize=(16, 7.4), dpi=110)
fig.patch.set_facecolor(BG)
gs = fig.add_gridspec(1, 2, width_ratios=[2.35, 1.45], wspace=0.13, left=0.255, right=0.975, top=0.79, bottom=0.15)
ax = fig.add_subplot(gs[0]); bx = fig.add_subplot(gs[1])
for a in (ax, bx):
    a.set_facecolor(BG)
    for sp in a.spines.values():
        sp.set_visible(False)
    a.tick_params(length=0, colors=MUTE)

# left: heatmap on a fixed 74-96 scale
cmap = LinearSegmentedColormap.from_list("board", ["#1b2436", "#25506a", "#3d9d8f", "#6fd6a8"])
n = len(order); ncol = len(TASKS) + 1
for i, s in enumerate(order):
    y = n - 1 - i
    for j, (t, _) in enumerate(TASKS + [("mean", "")]):
        v = means[s] if t == "mean" else float(rows[s][t]["score"])
        ax.add_patch(plt.Rectangle((j, y), 1, 1, color=cmap(max(0.0, min(1.0, (v - 74) / 22))), ec=BG, lw=3))
        ax.text(j + 0.5, y + 0.5, r1(v), ha="center", va="center", color=TEXT, fontsize=14,
                weight="bold" if t == "mean" else "normal")
    lab, _, col = ROWS[s]
    ax.text(-0.08, y + 0.5, lab, ha="right", va="center", color=SUBJ if col == SUBJ else TEXT, fontsize=11.4,
            weight="bold" if col == SUBJ else "normal")
for j, (t, name) in enumerate(TASKS + [("mean", "four-task\nmean")]):
    ax.text(j + 0.5, n + 0.1, name, ha="center", va="bottom", color=TEXT if t == "mean" else MUTE, fontsize=11.2,
            weight="bold" if t == "mean" else "normal")
ax.set_xlim(0, ncol); ax.set_ylim(0, n + 0.85); ax.set_xticks([]); ax.set_yticks([])

# right: mean vs file size
for s in rows:
    _, short, col = ROWS[s]
    bx.errorbar(gb[s], means[s], yerr=hws[s], fmt="o", color=col, ms=11 if col == SUBJ else 9, mec="white" if col == SUBJ else col,
                mew=1.4, elinewidth=1.6, capsize=4, zorder=4 if col == SUBJ else 3)
OFF = {  # label placement in data coords (x, y), ha; checked by a zoomed vision pass
    "qwen3-8-27b-q6-k": (21.0, 91.85, "right"),
    "qwen3-8-27b-ud-iq3-xxs": (12.7, 90.6, "left"),
    "ternary-bonsai-2-27b-pq2-0": (9.0, 86.95, "left"),
    "ternary-bonsai-2-27b-ptq1-0": (3.3, 91.85, "left"),
    "underdog-saluki-27b-iq2-mix": (8.7, 84.9, "left"),
}
for s, (x, y, ha) in OFF.items():
    _, short, col = ROWS[s]
    bx.annotate(f"{short}  {r1(means[s])}", xy=(gb[s], means[s]), xytext=(x, y), textcoords="data", ha=ha, va="center",
                color=col, fontsize=10.8, weight="bold" if col == SUBJ else "normal",
                arrowprops=dict(arrowstyle="-", color=col, lw=0.8, alpha=0.7, shrinkA=2, shrinkB=7,
                                relpos=(0.0, 0.5) if ha == "left" else (1.0, 0.5)))
bx.set_xlim(3, 25); bx.set_ylim(82, 93)
bx.set_xticks([5, 10, 15, 20, 25]); bx.set_xticklabels(["5", "10", "15", "20", "25 GB"], fontsize=10)
bx.set_yticks([82, 84, 86, 88, 90, 92]); bx.tick_params(axis="y", labelsize=10)
bx.grid(color=GRID, lw=1, alpha=0.8, zorder=0)
bx.text(3, 93.25, "four-task mean vs GGUF size\n(bars: Wilson 95% interval of the mean)", ha="left", va="bottom", color=MUTE, fontsize=11.2)

fig.text(0.02, 0.94, "Qwen3.8-27B under 10 GB, thinking on: Saluki keeps 95.6% of the Q6_K mean, both Bonsai files 98.9%", color=TEXT, fontsize=19, weight="bold")
fig.text(0.02, 0.905, "same base model, five GGUF files, one RTX 5090. greedy, zero-shot, max 16,384 completion tokens, ctx 24,576, 1,583 items per row.\n"
         "llama.cpp b10371 for the Qwen3.8 and Saluki rows. † Ternary-Bonsai rows run on the PrismML llama.cpp fork (its packings are not in mainline).",
         color=MUTE, fontsize=10.8, va="top")
fig.text(0.02, 0.025, "witcheer · llm-bench-rig dataset/second_tier.csv · IFEval prompt-strict (541), MATH-500 (500), HumanEval+ (164), MBPP+ (378) · "
         "file sizes are the GGUF on disk · Saluki measured 2026-10-11 to 2026-10-13", color=MUTE, fontsize=9.5)
out = "/opt/data/tmp/sub10/sub10-qwen38-27b.png"
fig.savefig(out, facecolor=BG)
print(out, {s: (round(gb[s], 2), round(means[s], 2), round(hws[s], 2)) for s in order})
