#!/usr/bin/env python3
"""this year's contests (AIME 2026 + HMMT Feb 2026, think-on 32k) vs MATH-500, nine local models on one RTX 5090.
left: stacked bar per model out of 63 problems: correct / finished but wrong / hit the 32k cap with no credited answer.
right: the same row's MATH-500 score and its rank among the nine. a pending leg is drawn from progress and labelled.
usage: python scripts/contest_board_data.py > contest-board.json (on the rig), then
       python scripts/chart_contest_board.py contest-board.json reports/matharena-contests-32k.png"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, TEXT, MUTE, GRID = "#0e1420", "#eaf0f7", "#7d8ba0", "#20293a"
OK, WRONG, CAP, PEND = "#1f9e8a", "#3a4558", "#e0645f", "#2a3346"
DATA = sys.argv[1] if len(sys.argv) > 1 else "contest-board.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "reports/matharena-contests-32k.png"
rows = json.load(open(DATA))

def tot(r, k):
    return sum(l[k] for l in r["legs"].values())

for r in rows:
    r["c"] = tot(r, "correct"); r["final"] = all(l["final"] for l in r["legs"].values())
    r["cw"] = tot(r, "capped") - tot(r, "capped_correct")
    r["fw"] = tot(r, "finished") - tot(r, "finished_correct")
    r["pend"] = 63 - tot(r, "done")
rows.sort(key=lambda r: (-r["c"], r["label"]))
mrank = {r["slug"]: i + 1 for i, r in enumerate(sorted(rows, key=lambda r: -r["math500"]["score"]))}

fig = plt.figure(figsize=(13.5, 7.6), dpi=110); fig.patch.set_facecolor(BG)
ax = fig.add_axes([0.27, 0.17, 0.5, 0.6]); ax.set_facecolor(BG)
for sp in ax.spines.values(): sp.set_visible(False)
ax.tick_params(length=0, labelsize=10.5, colors=MUTE)
ax.grid(axis="x", color=GRID, linewidth=1, alpha=0.7); ax.set_axisbelow(True)
pending = []
for y, r in enumerate(rows):
    ax.barh(y, r["c"], color=OK, height=0.64)
    ax.barh(y, r["fw"], left=r["c"], color=WRONG, height=0.64)
    ax.barh(y, r["cw"], left=r["c"] + r["fw"], color=CAP, height=0.64)
    if r["pend"]:
        ax.barh(y, r["pend"], left=r["c"] + r["fw"] + r["cw"], color=PEND, height=0.64, hatch="///", edgecolor=MUTE, linewidth=0)
        pending.append(r["label"])
    lab = f"{r['c']}" if r["final"] else f"{r['c']}+ ({r['pend']} left)"
    ax.text(r["c"] - 0.6, y, lab, ha="right", va="center", color=TEXT, fontsize=11, fontweight="bold")
    m = r["math500"]
    ax.text(66.5, y, f"{m['score']:.1f}", ha="left", va="center", color=TEXT, fontsize=11, fontweight="bold")
    ax.text(71.5, y, f"#{mrank[r['slug']]}  ({m['budget']})", ha="left", va="center", color=MUTE, fontsize=10)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r["label"] for r in rows], color=TEXT, fontsize=10.6)
ax.invert_yaxis(); ax.set_xlim(0, 63); ax.set_xticks([0, 10, 20, 30, 40, 50, 63])
ax.set_xlabel("problems out of 63 (AIME 2026: 30, HMMT Feb 2026: 33)", color=MUTE, fontsize=10.8)
ax.text(66.5, -0.95, "MATH-500", ha="left", va="center", color=MUTE, fontsize=10.2, fontweight="bold")
ax.text(66.5, len(rows) - 0.1, "", ha="left")

fig.text(0.04, 0.93, "this year's AIME and HMMT reorder the MATH-500 board", color=TEXT, fontsize=17, fontweight="bold")
fig.text(0.04, 0.885, "nine local models, one RTX 5090, thinking on, 32k-token cap, greedy, zero-shot. sorted by problems solved;",
         color=MUTE, fontsize=10.6)
fig.text(0.04, 0.855, "right column: the same model's MATH-500 score and rank among the nine (16k or 32k thinking cap, as run).",
         color=MUTE, fontsize=10.6)
hx = 0.04
for col, name in ((OK, "correct"), (WRONG, "finished, wrong"), (CAP, "hit the 32k cap, no credited answer")):
    fig.patches.append(matplotlib.patches.Rectangle((hx, 0.805), 0.014, 0.022, color=col, transform=fig.transFigure, figure=fig))
    fig.text(hx + 0.019, 0.816, name, color=TEXT, fontsize=10.2, va="center"); hx += 0.019 + 0.0072 * len(name) + 0.03
foot = ("llama.cpp, max_tokens 32,768, ctx 40,960, MathArena problem sets, strict sympy grader plus an exact-number check (0 vetoes fired).\n"
        "a few capped answers had already written a final answer and count as correct. MATH-500 cells are 16k for six rows and 32k for three.")
fig.text(0.04, 0.062, foot, color=MUTE, fontsize=9.1, linespacing=1.5)
fig.text(0.04, 0.035, "github.com/notwitcheer/llm-bench-rig  @witcheer", color=MUTE, fontsize=9.6)
if pending:
    fig.text(0.62, 0.035, f"PREVIEW: {', '.join(pending)} still running", color=CAP, fontsize=9.6)
fig.savefig(OUT, facecolor=BG); print(OUT, "pending:", pending)
