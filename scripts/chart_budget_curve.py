#!/usr/bin/env python3
"""reasoning-budget curve: GPQA-diamond think-on accuracy and tokens per correct at completion budgets 4k / 8k / 16k,
two models. left: accuracy vs budget, point labels carry the capped share. right: tokens per correct vs budget.
16k token cells are read when results/<slug>/budget-16384/gpqa.json exists on the mirror; otherwise the 16k point
uses the August accuracy and the right panel stops at 8k for that model."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, TEXT, MUTE, GRID = "#0e1420", "#eaf0f7", "#7d8ba0", "#20293a"
BASE = "/opt/data/tmp/budget"
MODELS = [("qwen3-8-27b-q6-k-thinkon", "Qwen3.8-27B Q6_K", "#4c9be8"),
          ("ornith-35b-thinkon", "Ornith 1.5 35B-A3B Q4_K_M", "#e8b64c")]
BUDGETS = [4096, 8192, 16384]

def leg(slug, b):
    p = f"{BASE}/{slug}/budget-{b}/gpqa.json"
    if os.path.exists(p):
        g = json.load(open(p))
        return dict(score=g["score"], tpc=g.get("tokens_per_correct"), capped=g.get("capped_rate"), tokens=True)
    if b == 16384:
        g = json.load(open(f"{BASE}/{slug}/gpqa-16k.json"))
        return dict(score=g["score"], tpc=None, capped=None, tokens=False)
    return None

fig = plt.figure(figsize=(14, 7.2), dpi=110); fig.patch.set_facecolor(BG)
gs = fig.add_gridspec(1, 2, left=0.07, right=0.98, top=0.74, bottom=0.2, wspace=0.26)
axL, axR = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
for ax in (axL, axR):
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0, labelsize=10.5, colors=MUTE)
    ax.grid(axis="y", color=GRID, linewidth=1, alpha=0.7); ax.set_axisbelow(True)
    ax.set_xticks(range(3)); ax.set_xticklabels(["4,096", "8,192", "16,384"], color=TEXT, fontsize=11)
    ax.set_xlabel("completion budget (max_tokens), think-on", color=MUTE, fontsize=11)
    ax.set_xlim(-0.35, 2.45)

partial = []
for slug, name, col in MODELS:
    legs = [leg(slug, b) for b in BUDGETS]
    ys = [l["score"] for l in legs]
    axL.plot(range(3), ys, color=col, lw=2.4, marker="o", ms=8, zorder=3, label=name)
    for i, l in enumerate(legs):
        cap = f"  {l['capped']*100:.0f}% capped" if l["capped"] is not None else "  (August run)"
        # qwen (lower line) labels below the point, ornith (upper line) above; the 16k pair is close so push harder
        dy = (-2.6 if slug.startswith("qwen") else 2.0) * (1.6 if i == 2 else 1.0)
        axL.text(i, l["score"] + dy, f"{l['score']:.1f}{cap}", ha="center", va="top" if dy < 0 else "bottom",
                 color=TEXT, fontsize=9.6)
    tp = [(i, l["tpc"]) for i, l in enumerate(legs) if l["tpc"]]
    axR.plot([i for i, _ in tp], [v for _, v in tp], color=col, lw=2.4, marker="o", ms=8, zorder=3, label=name)
    for i, v in tp:
        axR.text(i, v + 130, f"{v:,.0f}", ha="center", va="bottom", color=TEXT, fontsize=10, fontweight="bold")
    if not legs[2]["tokens"]:
        partial.append(name)

axL.set_ylim(52, 90); axL.set_ylabel("GPQA-diamond accuracy, % (198 items, greedy)", color=MUTE, fontsize=11)
axL.set_title("accuracy tracks how many items the budget cuts off", color=TEXT, fontsize=12.5, loc="left", pad=10)
axL.legend(loc="lower right", frameon=False, labelcolor=TEXT, fontsize=10.5)
axR.set_ylim(2500, 8000); axR.set_ylabel("completion tokens per correct answer", color=MUTE, fontsize=11)
axR.set_title("what each extra point costs", color=TEXT, fontsize=12.5, loc="left", pad=10)
axR.legend(loc="upper left", frameon=False, labelcolor=TEXT, fontsize=10.5)

fig.text(0.07, 0.93, "how much thinking is worth paying for: GPQA-diamond think-on at three completion budgets",
         color=TEXT, fontsize=15.5, fontweight="bold")
fig.text(0.07, 0.885, "one RTX 5090, llama.cpp, ctx 24,576, temperature 0, 198 items. every point lost to a smaller budget is an item that hit the cap;",
         color=MUTE, fontsize=10.6)
fig.text(0.07, 0.855, "items that finished were ~95% correct at every budget, for both models.", color=MUTE, fontsize=10.6)
foot = "one item = 0.5 pts; gaps under ~3 pts are noise. 4k and 8k legs 2026-09-09, 16k legs 2026-09-10.  github.com/notwitcheer/llm-bench-rig  @witcheer"
fig.text(0.07, 0.075, foot, color=MUTE, fontsize=9.6)
if partial:
    fig.text(0.07, 0.045, f"16k token cells pending for {', '.join(partial)}: 16k accuracy from the August run, tokens not recorded there.", color=MUTE, fontsize=9.6)
out = "/opt/data/tmp/budget/reasoning-budget-curve.png"
fig.savefig(out, facecolor=BG); print(out, "partial:", partial)
