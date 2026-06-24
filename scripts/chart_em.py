"""One-Shot EM: claimed vs measured (Gold & Crimson v2). The paper claims +24.7 avg on
Qwen2.5-Math-7B from one example. On a 32GB consumer GPU (the full-param recipe OOMs, so LoRA),
we reproduce the BASE exactly but the gain evaporates: MATH500 +2.0 then collapse, AMC23 -2.5.
Entropy was minimized (0.098 -> 0.035) with ~no accuracy gain — distribution-sharpening, not
learning. The bars show our base (reproduces the paper's base) and our measured EM against the
paper's claimed EM."""
import glob
import json

import matplotlib.pyplot as plt
import numpy as np

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = ("#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64")

R = {}
for f in glob.glob("results/em/*.json"):
    if f.endswith(".gens.json"):
        continue
    d = json.load(open(f))
    R[(d["label"], d["variant"], d["benchmark"])] = d


def acc(lab, var, bm):
    return R.get((lab, var, bm), {}).get("acc", float("nan"))


CLAIM = {"math500": (53.0, 78.8), "amc23": (44.1, 70.3)}   # paper Table 2: (base, EM@10)
benches = ["math500", "amc23"]
labels = {"math500": "MATH500 (500q)", "amc23": "AMC23 (40q)"}

fig, ax = plt.subplots(figsize=(10, 6.2), facecolor=BG)
ax.set_facecolor(BG)
fig.suptitle("One-Shot Entropy Minimization on Qwen2.5-Math-7B: claimed vs measured",
             color=GOLD, fontsize=15, fontweight="bold", y=0.97)

x = np.arange(len(benches))
w = 0.26
base_ours = [acc("base", "paper", b) for b in benches]
em_ours = [acc("em-step10", "paper", b) for b in benches]
em_claim = [CLAIM[b][1] for b in benches]

b1 = ax.bar(x - w, base_ours, w, color=MUTE, edgecolor=TEXT, zorder=3,
            label="base (ours — reproduces the paper's base)")
b2 = ax.bar(x, em_ours, w, color=CRIMSON, edgecolor=TEXT, zorder=3,
            label="EM @ step 10 (ours, measured)")
b3 = ax.bar(x + w, em_claim, w, color=BG, edgecolor=GOLD, hatch="//", linewidth=1.6, zorder=3,
            label="EM (paper's claim)")

for bars in (b1, b2, b3):
    for r in bars:
        ax.annotate(f"{r.get_height():.1f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                    color=TEXT, ha="center", va="bottom", fontsize=10.5, fontweight="bold",
                    xytext=(0, 3), textcoords="offset points")
# our measured delta over base
for i, b in enumerate(benches):
    d = em_ours[i] - base_ours[i]
    ax.annotate(f"ours: {d:+.1f}", (i, em_ours[i] / 2), color=TEXT, ha="center",
                fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([labels[b] for b in benches], color=TEXT, fontsize=12)
ax.set_ylabel("accuracy (%)", color=TEXT, fontsize=11)
ax.set_ylim(0, 98)
ax.set_title("greedy pass@1, authors' grader · base reproduced (53.4 vs paper 53.0) · "
             "entropy minimized 0.098→0.035, accuracy flat",
             color=MUTE, fontsize=9.5, pad=8)
ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=10, loc="upper center", ncol=3)
for s in ax.spines.values():
    s.set_color(GRID)
ax.tick_params(colors=MUTE)
ax.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)

fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("reports/one-shot-em.png", dpi=150, facecolor=BG)
print("wrote reports/one-shot-em.png")
