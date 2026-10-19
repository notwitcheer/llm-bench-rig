"""Qwen3.8-27B-Coder390 (nerkyor fine-tune) vs Qwen3.8-27B, both Q6_K, thinking on, 16k cap, one RTX 5090.

numbers re-derived 2026-10-19 from capsule results/qwen3-8-27b-coder390-q6-k-mtp-thinkon/ and
results/qwen3-8-27b-q6-k-thinkon/ (<task>_detail.json / gpqa.json scores, <task>_tokens.jsonl
completion tokens deduped by idx, paired sign test on <task>_progress.json correctness).
render: python3 scripts/chart_coder390_vs_base.py [out.png]  (matplotlib)
base gpqa tokens: results/qwen3-8-27b-q6-k-thinkon/budget-16384/gpqa_tokens.jsonl (198/198 predictions identical to the board row).
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

BG, FG, MUT, GRID = "#0f1117", "#e8e8ea", "#9a9caa", "#262938"
TUNE, BASE, RED = "#f2a541", "#5b8def", "#e5484d"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": FG, "axes.labelcolor": FG,
                     "xtick.color": MUT, "ytick.color": FG})

# task label, tune score, base score, delta note
SCORES = [
    ("GPQA Diamond", 83.84, 79.29, "+4.6  (p 0.08, 15% of answers hit the cap)"),
    ("MATH-500", 94.8, 95.2, "-0.4  (noise)"),
    ("HumanEval+", 92.07, 92.07, "0.0"),
    ("MBPP+", 82.54, 81.75, "+0.8  (noise)"),
    ("IFEval (strict)", 86.32, 89.65, "-3.3  (p 0.008)"),
]
# mean completion tokens per item: tune, base
TOKENS = [
    ("GPQA Diamond", 4622, 6138),
    ("MATH-500", 1412, 1694),
    ("HumanEval+", 1141, 1577),
    ("MBPP+", 983, 1722),
    ("IFEval (strict)", 2025, 2004),
]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 6.8), dpi=150, gridspec_kw={"width_ratios": [1.25, 1]})
fig.patch.set_facecolor(BG)
for a in (a1, a2):
    a.set_facecolor(BG)
    for s in ("top", "right", "left"):
        a.spines[s].set_visible(False)
    a.spines["bottom"].set_color("#333645")

# panel 1: dumbbell of scores
ys = list(range(len(SCORES)))[::-1]
for y, (name, t, b, note) in zip(ys, SCORES):
    a1.plot([min(t, b), max(t, b)], [y, y], color="#4a4d5e", lw=3, zorder=1)
    a1.scatter([b], [y], s=150, color=BASE, zorder=3, edgecolor=BG, lw=1.5)
    a1.scatter([t], [y], s=150, color=TUNE, zorder=4, edgecolor=BG, lw=1.5)
    lo, hi = (t, b) if t < b else (b, t)
    if abs(t - b) < 0.01:
        a1.scatter([t], [y], s=150, color=BASE, zorder=5, edgecolor=BG, lw=1.5, marker=MarkerStyle("o", fillstyle="right"))
        a1.text(t + 0.6, y, f"{t:.2f} both", ha="left", va="center", fontsize=11, color=FG, fontweight="bold")
    else:
        left_c, right_c = (TUNE, BASE) if t < b else (BASE, TUNE)
        a1.text(lo - 0.6, y, f"{lo:.2f}", ha="right", va="center", fontsize=11, color=left_c, fontweight="bold")
        a1.text(hi + 0.6, y, f"{hi:.2f}", ha="left", va="center", fontsize=11, color=right_c, fontweight="bold")
    col = RED if name.startswith("IFEval") else MUT
    a1.text(74.2, y - 0.36, note, ha="left", va="center", fontsize=10, color=col,
            bbox=dict(boxstyle="square,pad=0.15", fc=BG, ec="none"), zorder=6)
a1.set_yticks(ys)
a1.set_yticklabels([s[0] for s in SCORES], fontsize=12)
a1.set_xlim(74, 99.5)
a1.set_ylim(-0.75, len(SCORES) - 0.35)
a1.set_xticks([75, 80, 85, 90, 95])
a1.tick_params(axis="y", length=0)
a1.grid(axis="x", color=GRID, lw=0.8)
a1.set_axisbelow(True)
a1.set_title("score (%), same items, coder tune minus base", fontsize=13.5, color=FG, loc="left", pad=12)

# panel 2: mean completion tokens
ys2 = list(range(len(TOKENS)))[::-1]
h = 0.36
for y, (name, t, b) in zip(ys2, TOKENS):
    a2.barh(y + h / 2, b, h, color=BASE)
    a2.barh(y - h / 2, t, h, color=TUNE)
    a2.text(b + 110, y + h / 2, f"{b:,}", ha="left", va="center", fontsize=10.5, color=FG)
    a2.text(t + 110, y - h / 2, f"{t:,}", ha="left", va="center", fontsize=10.5, color=FG)
    pct = round(100 * (t - b) / b)
    lab = f"{pct:+d}%"
    a2.text(8550, y, lab, ha="right", va="center", fontsize=13, fontweight="bold",
            color=TUNE if pct < -5 else MUT)
a2.set_yticks(ys2)
a2.set_yticklabels([s[0] for s in TOKENS], fontsize=12)
a2.tick_params(axis="y", length=0)
a2.set_xlim(0, 8600)
a2.set_xticks([])
a2.set_title("tokens spent per answer (mean)", fontsize=13.5, color=FG, loc="left", pad=12)

# shared legend
from matplotlib.lines import Line2D
handles = [Line2D([], [], marker="o", ls="", ms=11, color=TUNE, label="Qwen3.8-27B-Coder390 (nerkyor fine-tune), Q6_K"),
           Line2D([], [], marker="o", ls="", ms=11, color=BASE, label="Qwen3.8-27B base, Q6_K")]
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.03, 0.075), ncol=2, frameon=False, fontsize=11.5)

fig.suptitle("A popular coder fine-tune of Qwen3.8-27B, against its base", fontsize=19, fontweight="bold",
             x=0.03, ha="left", y=0.975)
fig.text(0.03, 0.895, "one RTX 5090, llama.cpp, thinking on, 16,384-token cap, zero-shot, greedy, 1,781 items per model",
         fontsize=11.5, color=MUT, ha="left")
fig.text(0.03, 0.03, "p = paired sign test on the items only one model got right", fontsize=10, color=MUT, ha="left")
fig.text(0.97, 0.03, "witcheer · RTX 5090 bench", fontsize=10, color=MUT, ha="right")
fig.subplots_adjust(left=0.13, right=0.975, top=0.80, bottom=0.17, wspace=0.36)
out = sys.argv[1] if len(sys.argv) > 1 else "/opt/data/cache/scratch/coder390-vs-qwen38-27b.png"
fig.savefig(out, facecolor=BG)
print(out)
