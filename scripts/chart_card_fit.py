#!/usr/bin/env python3
"""card_fit chart: five-task board quality vs measured 16k VRAM peak, one dot per resident row (think-off),
vertical card lines at 8/12/16/24/32 GB minus the 768 MiB headroom. dots coloured by 16 GB fit class.
data: dataset/card_fit.csv (llm-bench-rig 86c8bc6), the same rows as the post."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, TEXT, MUTE, GRID = "#0e1420", "#eaf0f7", "#7d8ba0", "#20293a"
FIT, TIGHT, BIG, OFF = "#4cc98a", "#e8b64c", "#5b6b85", "#e88fb8"
CSV = "/opt/data/repos/llm-bench-rig/dataset/card_fit.csv"
HEADROOM = 768

rows = []
for r in csv.DictReader(open(CSV)):
    if not r["vram_peak_mib"] or not r["q_avg"]:
        continue
    peak = int(r["vram_peak_mib"]) / 1024
    rows.append(dict(name=f'{r["model"]} {r["quant"]}', peak=peak, q=float(r["q_avg"]),
                     off=bool(r["offload_n_cpu_moe"]), gpqa=r["gpqa"]))

def cls(r):
    if r["off"]:
        return OFF
    if r["peak"] + HEADROOM / 1024 <= 16:
        return FIT
    return BIG

fig, ax = plt.subplots(figsize=(14, 7.6), dpi=110)
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0, labelsize=11.5, colors=MUTE)
ax.grid(color=GRID, linewidth=1, alpha=0.7); ax.set_axisbelow(True)

for gb in (8, 12, 16, 24, 32):
    x = gb - HEADROOM / 1024
    ax.axvline(x, color=MUTE if gb != 16 else FIT, lw=1.2 if gb != 16 else 2.0, ls=(0, (3, 3)), zorder=2)
    ax.text(x - 0.15, 82.7, f"{gb} GB card", color=MUTE if gb != 16 else FIT, fontsize=10.5, ha="right", va="bottom", rotation=90)

for r in rows:
    ax.scatter(r["peak"], r["q"], s=110, color=cls(r), edgecolor=BG, linewidth=1.2, zorder=4)

# labels for the rows the post names: explicit text positions in data coords + leader lines, spread so nothing stacks
LABELS = {
    # short labels: the Qwen3.8-27B cuts are named by quant only (the subtitle says so). positions chosen so no label
    # text crosses a card line (lines at 7.25, 11.25, 15.25, 23.25, 31.25) or sits on a dot.
    "Qwen3.8-27B UD-IQ2_XXS": (7.6, 89.2, "IQ2_XXS  90.8"),
    "Qwen3.8-27B UD-IQ2_M": (7.6, 92.9, "IQ2_M  91.5"),
    "Qwen3.8-27B UD-IQ3_XXS": (11.9, 94.6, "IQ3_XXS  92.7"),
    "Gemma 4 12B-it Q6_K": (2.3, 86.2, "Gemma 4 12B Q6_K  87.6"),
    "gpt-oss-20b Q4_K_M": (11.6, 85.6, "gpt-oss-20b  87.4"),
    "Qwen3.8-27B Q4_K_M": (16.0, 88.6, "Q4_K_M  93.2: 15.9 GB file,\n17.3 GB at 16k, misses 16 GB"),
    "Gemma 4 31B-it (unsloth cut) Q4_0": (16.0, 95.6, "Gemma 4 31B Q4_0  94.3"),
    "Qwen3.6-27B Q6_K": (24.2, 95.6, "Qwen3.6-27B Q6_K  94.2"),
}
for r in rows:
    if r["name"] in LABELS:
        tx, ty, txt = LABELS[r["name"]]
        ax.annotate(txt, (r["peak"], r["q"]), xytext=(tx, ty), color=TEXT, fontsize=10.5, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.9, shrinkA=0, shrinkB=5), zorder=5)

# legend
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=9, color=FIT, label="fits a 16 GB card with a 16k context"),
                   Line2D([], [], marker="o", ls="", ms=9, color=BIG, label="needs 24 or 32 GB at 16k"),
                   Line2D([], [], marker="o", ls="", ms=9, color=OFF, label="MoE with experts in system RAM (offload rows)")],
          loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3, frameon=False, fontsize=10.5, labelcolor=TEXT)

ax.set_xlim(2, 33.5); ax.set_ylim(82.5, 96.2)
ax.set_xlabel("measured peak VRAM at a 16k prompt, GB (RTX 5090, llama-server)", color=MUTE, fontsize=11.5)
ax.set_ylabel("five-task board, q_avg (thinking off)", color=MUTE, fontsize=11.5)
fig.text(0.07, 0.955, "what your card can run with a 16k context: quality vs VRAM, 38 resident rows", color=TEXT, fontsize=17, weight="bold")
fig.text(0.07, 0.915, "card lines sit at capacity minus 768 MiB headroom. quant-only labels are Qwen3.8-27B cuts.\npink = experts in system RAM (Ling-3 127B: 3 GB VRAM + 59 GB RAM; Flash-Next 125B: 28 GB + RAM).",
         color=MUTE, fontsize=11, va="top")
fig.text(0.07, 0.015, "witcheer · llm-bench-rig dataset/card_fit.md · 5090 measurements applied as a budget to other card sizes; speed does not transfer, memory and quality do", color=MUTE, fontsize=9.5)
fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.19)
out = "/opt/data/tmp/card-fit-16gb.png"
fig.savefig(out, facecolor=BG); print(out, len(rows), "rows plotted")
