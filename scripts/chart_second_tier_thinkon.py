#!/usr/bin/env python3
"""second-tier think-on board card: 10 models x 4 tasks at a 16k thinking budget on one RTX 5090.
left: score table as a heatmap (rows sorted by the four-task mean), cells that hit the budget cap on
10% or more of their items carry a marker, the one format-miss cell (answers without \\boxed{}) too.
right: completion tokens per correct MATH-500 answer, the cost behind the maths column.
data: dataset/second_tier.csv (this commit), built from results/<slug>-thinkon/<task>_detail.json."""
import csv
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BG, TEXT, MUTE, GRID = "#0e1420", "#eaf0f7", "#7d8ba0", "#20293a"
FLAG, FORMAT = "#e8b64c", "#e88fb8"
CSV = "/opt/data/repos/llm-bench-rig/dataset/second_tier.csv"
TASKS = [("ifeval", "IFEval\nprompt-strict"), ("math500", "MATH-500"), ("humaneval_plus", "HumanEval+"), ("mbpp_plus", "MBPP+")]
LABEL = {
    "gemma-4-31b-q4-0-it": "Gemma 4 31B · QAT Q4_0 · 17.7 GB",
    "qwen3-8-27b-q6-k": "Qwen3.8-27B · Q6_K · 22.9 GB",
    "qwen3-8-27b-ud-iq3-xxs": "Qwen3.8-27B · UD-IQ3_XXS · 11.9 GB",
    "nvidia-nemotron-3-5-lightning-30b-a3b-q4-k-m": "Nemotron 3.5 Lightning 30B-A3B · Q4_K_M · 24.5 GB",
    "ornith-1-5-35b-q4-k-m": "Ornith 1.5 35B-A3B · Q4_K_M · 21.7 GB",
    "qwen3-6-35b-a3b-ud-q5-k-m": "Qwen3.6-35B-A3B · UD-Q5_K_M · 26.5 GB",
    "qwen3-6-27b-q6-k": "Qwen3.6-27B · Q6_K · 22.9 GB",
    "qwable-27b-q4-k-m": "Qwable-27B · Q4_K_M · 16.5 GB",
    "ornith-9b": "Ornith 1.5 9B · Q6_K · 7.4 GB",
    "qwopus3-8-27b-flash-q6-k": "Qwopus3.8-27B-Flash · Q6_K · 22.4 GB",
}

rows = {}
for r in csv.DictReader(open(CSV)):
    rows.setdefault(r["slug"], {})[r["task"]] = r
order = sorted(rows, key=lambda s: -statistics.mean(float(rows[s][t]["score"]) for t, _ in TASKS))
means = {s: statistics.mean(float(rows[s][t]["score"]) for t, _ in TASKS) for s in order}

fig = plt.figure(figsize=(16, 8.4), dpi=110)
fig.patch.set_facecolor(BG)
gs = fig.add_gridspec(1, 2, width_ratios=[3.05, 1.15], wspace=0.06, left=0.305, right=0.985, top=0.80, bottom=0.145)
ax = fig.add_subplot(gs[0]); bx = fig.add_subplot(gs[1])
for a in (ax, bx):
    a.set_facecolor(BG)
    for s in a.spines.values():
        s.set_visible(False)
    a.tick_params(length=0, colors=MUTE)

cmap = LinearSegmentedColormap.from_list("board", ["#1b2436", "#25506a", "#3d9d8f", "#6fd6a8"])
n = len(order)
ncol = len(TASKS) + 1
for i, s in enumerate(order):
    y = n - 1 - i
    for j, (t, _) in enumerate(TASKS + [("mean", "")]):
        if t == "mean":
            v = means[s]; capped = 0.0; pf = 0; total = 1
        else:
            r = rows[s][t]; v = float(r["score"]); capped = float(r["capped_rate"]); pf = int(r["parse_failures"] or 0); total = int(r["total"])
        # colour on a 60-96 scale, so the 33.6 cell reads as the floor it is
        c = cmap(max(0.0, min(1.0, (v - 60) / 36)))
        ax.add_patch(plt.Rectangle((j, y), 1, 1, color=c, ec=BG, lw=3))
        weight = "bold" if t == "mean" else "normal"
        ax.text(j + 0.5, y + 0.56, f"{v:.1f}", ha="center", va="center", color=TEXT, fontsize=13.5, weight=weight)
        if t != "mean" and capped >= 0.10:
            ax.text(j + 0.5, y + 0.22, f"▲ {capped*100:.0f}% hit the cap", ha="center", va="center", color=FLAG, fontsize=8.6)
        elif t == "math500" and pf > 0.25 * total:
            ax.text(j + 0.5, y + 0.22, f"◆ {pf}/{total} unboxed", ha="center", va="center", color=FORMAT, fontsize=8.6)
    ax.text(-0.08, y + 0.5, LABEL[s], ha="right", va="center", color=TEXT, fontsize=11.2)
for j, (t, name) in enumerate(TASKS + [("mean", "four-task\nmean")]):
    ax.text(j + 0.5, n + 0.12, name, ha="center", va="bottom", color=MUTE if t != "mean" else TEXT, fontsize=11.2, weight="bold" if t == "mean" else "normal")
ax.set_xlim(0, ncol); ax.set_ylim(0, n + 0.9); ax.set_xticks([]); ax.set_yticks([])

# right panel: tokens per correct MATH-500 answer, same row order
tpc = [float(rows[s]["math500"]["tokens_per_correct"]) for s in order]
ys = [n - 1 - i + 0.5 for i in range(n)]
colors = [FLAG if float(rows[s]["math500"]["capped_rate"]) >= 0.10 else "#5b6b85" for s in order]
bx.barh(ys, tpc, height=0.62, color=colors, zorder=3)
for y, v in zip(ys, tpc):
    bx.text(v + 250, y, f"{v/1000:.1f}k", va="center", ha="left", color=TEXT, fontsize=10.5)
bx.set_ylim(0, n + 0.9); bx.set_yticks([]); bx.set_xlim(0, 16500)
bx.set_xticks([0, 4000, 8000, 12000, 16000]); bx.set_xticklabels(["0", "4k", "8k", "12k", "16k"], fontsize=10)
bx.grid(axis="x", color=GRID, lw=1, alpha=0.8, zorder=0)
bx.text(0, n + 0.12, "tokens generated per\ncorrect MATH-500 answer", ha="left", va="bottom", color=MUTE, fontsize=11.2)

fig.text(0.02, 0.945, "thinking on, 16k budget: ten models, four tasks, one RTX 5090", color=TEXT, fontsize=19, weight="bold")
fig.text(0.02, 0.905, "greedy, zero-shot, max 16,384 completion tokens, ctx 24,576, llama.cpp b9653, 1,583 items per model. rows sorted by the four-task mean.\n"
         "▲ budget-limited: the model hit the 16k cap on 10% or more of the items, so that score is a floor, not a ceiling. ◆ format miss: answers given without \\boxed{}, the grader could not read them.",
         color=MUTE, fontsize=10.8, va="top")
fig.text(0.02, 0.02, "witcheer · llm-bench-rig dataset/second_tier.csv · IFEval prompt-strict (541), MATH-500 (500), HumanEval+ (164), MBPP+ (378) · file sizes are the GGUF on disk · measured 2026-09-10 to 2026-09-20",
         color=MUTE, fontsize=9.5)
out = "/opt/data/tmp/second-tier-thinkon-16k.png"
fig.savefig(out, facecolor=BG); print(out, n, "rows plotted")
