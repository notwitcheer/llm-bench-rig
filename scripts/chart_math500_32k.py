#!/usr/bin/env python3
"""MATH-500 think-on at 16k vs 32k completion budget, the three rows that capped at 16k on the second-tier board.
one stacked bar per model per budget: correct / finished but wrong / ran out of budget (capped and not credited).
data: a json built on capsule from results/<slug>-thinkon{,-32k}/math500_progress.json + math500_detail.json
(mercury: /opt/data/tmp/math500-32k/data32k.json). a leg without a detail json is drawn and labelled as in progress."""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, TEXT, MUTE, GRID = "#0e1420", "#eaf0f7", "#7d8ba0", "#20293a"
OK, WRONG, CAP = "#1f9e8a", "#3a4558", "#e0645f"
DATA = sys.argv[1] if len(sys.argv) > 1 else "/opt/data/tmp/math500-32k/data32k.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/opt/data/tmp/math500-32k/math500-32k.png"
rows = json.load(open(DATA))

fig = plt.figure(figsize=(13, 7.4), dpi=110); fig.patch.set_facecolor(BG)
ax = fig.add_axes([0.25, 0.2, 0.62, 0.56]); ax.set_facecolor(BG)
for sp in ax.spines.values(): sp.set_visible(False)
ax.tick_params(length=0, labelsize=10.5, colors=MUTE)
ax.grid(axis="x", color=GRID, linewidth=1, alpha=0.7); ax.set_axisbelow(True)

y, ticks, labels, partial = 0.0, [], [], []
for r in rows:
    for key in ("16k", "32k"):
        s = r[key]; n = s["n"]
        c, fw, cw = s["correct"] / n * 100, s["finished_wrong"] / n * 100, s["capped_wrong"] / n * 100
        ax.barh(y, c, color=OK, height=0.62)
        ax.barh(y, fw, left=c, color=WRONG, height=0.62)
        ax.barh(y, cw, left=c + fw, color=CAP, height=0.62)
        if s.get("final"):
            lab = f"{s['score']:.1f}"
        else:
            lab = f"{c:.1f} running ({n}/500)"; partial.append(r["label"])
        ax.text(c - 1.2, y, lab, ha="right", va="center", color=TEXT, fontsize=11, fontweight="bold")
        ax.text(101.2, y, f"{s['capped'] / n * 100:.0f}% hit the cap", ha="left", va="center", color=CAP if key == "16k" else MUTE, fontsize=10)
        ticks.append(y); labels.append(f"{r['label']}  ·  {key}")
        y += 0.78
    y += 0.55
ax.set_yticks(ticks); ax.set_yticklabels(labels, color=TEXT, fontsize=10.8); ax.invert_yaxis()
ax.set_xlim(0, 100); ax.set_xlabel("share of the 500 MATH-500 questions, %", color=MUTE, fontsize=11)

d = sorted(r["32k"]["score"] - r["16k"]["score"] for r in rows if r["32k"].get("final"))
rng = f"+{d[0]:.1f}" if abs(d[-1] - d[0]) < 0.05 else f"+{d[0]:.1f} to +{d[-1]:.1f}"
fig.text(0.04, 0.93, f"double the thinking room, {rng} points on MATH-500", color=TEXT, fontsize=16, fontweight="bold")
fig.text(0.04, 0.885, "same 500 questions, one RTX 5090; only the budget changed (max_tokens 16,384 to 32,768, context raised to fit). every answer that finished",
         color=MUTE, fontsize=10.6)
fig.text(0.04, 0.855, "inside 16k got the same verdict at 32k: the extra points are all questions that had run out of room.", color=MUTE, fontsize=10.6)
hx = 0.04
for col, name in ((OK, "correct"), (WRONG, "finished, wrong"), (CAP, "hit the cap, no credited answer")):
    fig.patches.append(matplotlib.patches.Rectangle((hx, 0.805), 0.014, 0.022, color=col, transform=fig.transFigure, figure=fig))
    fig.text(hx + 0.019, 0.816, name, color=TEXT, fontsize=10.2, va="center"); hx += 0.019 + 0.0075 * len(name) + 0.03
foot = ("think-on, greedy, zero-shot, llama.cpp; ctx 24,576 at 16k, 40,960 at 32k. one question = 0.2 pts. some capped answers still carry a gradable "
        "answer and count as correct, so the red bar is shorter than the cap share. the 32k pass is its own table, not merged into the 16k board.")
fig.text(0.04, 0.085, foot, color=MUTE, fontsize=9.2, wrap=True)
fig.text(0.04, 0.055, "github.com/notwitcheer/llm-bench-rig  @witcheer", color=MUTE, fontsize=9.6)
if partial:
    fig.text(0.04, 0.025, f"PREVIEW: {', '.join(partial)} 32k run still in progress", color=CAP, fontsize=9.6)
fig.savefig(OUT, facecolor=BG); print(OUT, "partial:", partial)
