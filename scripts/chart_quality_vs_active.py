#!/usr/bin/env python3
"""Gold & Crimson scatter: MMLU vs active params (log x). Sparse MoE vs dense field.
Highlights the two new 5090-treatment models. MMLU used (HumanEval is harness-valid
only for the two fixed models; not re-asserting old field HumanEval)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BG="#0d0906"; GOLD="#e8c44a"; TEXT="#f5e6d0"; CRIMSON="#e06060"; GRID="#3a2f25"; MUTE="#8a7a64"

# (label, active_params_B, mmlu, kind, (dx,dy,ha))  kind: head|side|dense|moe
DATA = [
    ("gpt-oss-120B  (MXFP4, offloaded)", 5.1, 89.5, "head", (9, 12, "left")),
    ("Qwen3.6-28B-REAP-A3B",             3.0, 87.7, "moe",  (9, 9, "left")),
    ("Qwen3.6-27B (dense)",             27.0, 87.9, "dense",(0, 13, "center")),
    ("Gemma-4-31B (dense)",             31.0, 87.8, "dense",(10, -20, "left")),
    ("Qwen3.6-35B-A3B",                  3.0, 85.0, "moe",  (9, 8, "left")),
    ("gpt-oss-20B",                      3.6, 78.6, "moe",  (9, 11, "left")),
    ("Nemotron-Cascade-2-30B-A3B",       3.0, 74.4, "moe",  (9, 11, "left")),
]

fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
fig.subplots_adjust(left=0.085, right=0.96, top=0.76, bottom=0.13)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0, labelsize=14, colors=TEXT)
ax.grid(color=GRID, linewidth=1, alpha=0.6); ax.set_axisbelow(True)
ax.set_xscale("log")

for label, ap, mmlu, kind, off in DATA:
    if kind == "head":   c, sz, z = CRIMSON, 360, 5
    elif kind == "side": c, sz, z = GOLD, 360, 5
    else:                c, sz, z = MUTE, 200, 3
    ax.scatter(ap, mmlu, s=sz, color=c, zorder=z, edgecolors=BG, linewidths=1.5)
    big = kind in ("head", "side")
    dx, dy, ha = off
    ax.annotate(label, (ap, mmlu),
                xytext=(dx, dy), textcoords="offset points",
                ha=ha, va="bottom",
                color=(TEXT if big else MUTE), fontsize=(13 if big else 11),
                fontweight=("bold" if big else "normal"))

ax.set_xlim(2.4, 46); ax.set_ylim(72, 92)
ax.set_xticks([3, 5, 10, 20, 30]); ax.set_xticklabels(["3B", "5B", "10B", "20B", "30B"])
ax.set_xlabel("Active parameters per token  (log scale)", fontsize=14, color=TEXT)
ax.set_ylabel("MMLU accuracy  (%)", fontsize=14, color=TEXT)

# shaded "sparse" region note
ax.axvspan(2.4, 6, color=GOLD, alpha=0.05)
ax.text(3.0, 73.2, "sparse-active MoE", color=GOLD, fontsize=12, alpha=0.8)
ax.text(28, 73.2, "dense", color=MUTE, fontsize=12, ha="center")

fig.add_artist(Rectangle((0.085, 0.91), 0.07, 0.012, color=CRIMSON, transform=fig.transFigure))
fig.text(0.085, 0.85, "gpt-oss-120B: frontier MMLU at 5.1B active params", fontsize=22, fontweight="bold", color=TEXT)
fig.text(0.085, 0.805, "MMLU vs active params on one RTX 5090 — sparse MoEs match dense 27-31B (reasoning on, n≈114)", fontsize=13, color=GOLD)
fig.text(0.96, 0.045, "WITCHEER · RTX 5090", fontsize=12, color=MUTE, ha="right")
fig.savefig("reports/chart-quality-vs-active.png", facecolor=BG)
print("wrote reports/chart-quality-vs-active.png")
