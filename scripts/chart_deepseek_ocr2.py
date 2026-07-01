"""DeepSeek-OCR-2: compression ratio vs decode accuracy, measured vs the paper's claimed curve (Gold & Crimson v2).

Paper: <10x compression -> 97% precision, 20x -> ~60% (arXiv 2510.18234, exact test set unspecified).
Measured: 15 real OmniDocBench pages, edit-distance decode accuracy vs ground truth.
The measured points sit BELOW the paper's curve at comparable compression -- real, diverse
documents (dense multi-column newspapers especially) degrade faster than the paper's own reference.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = ("#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64")
AMBER = "#d8902f"

results = json.load(open("results/deepseek-ocr-2/results_v2.json"))
results = [r for r in results if "error" not in r]

fig, ax = plt.subplots(figsize=(10.5, 7.6), facecolor=BG)
ax.set_facecolor(BG)

xs = [r["compression_vs_gt"] for r in results]
ys = [r["decode_accuracy_v2"] * 100 for r in results]
ax.scatter(xs, ys, s=130, color=CRIMSON, edgecolor=TEXT, linewidth=1.2, zorder=4, label="measured (this rig, n=15)")

# paper's claimed reference curve: <10x -> 97%, 20x -> ~60%
px, py = [10, 20], [97, 60]
ax.plot(px, py, color=GOLD, lw=2.6, ls="--", marker="D", markersize=9, zorder=3, label="paper's claimed curve (arXiv 2510.18234)")
ax.annotate("<10x -> 97%", xy=(10, 97), xytext=(10.6, 99.5), color=GOLD, fontsize=9.5, fontweight="bold")
ax.annotate("20x -> ~60%", xy=(20, 60), xytext=(20.6, 62.5), color=GOLD, fontsize=9.5, fontweight="bold")

# annotate the repetition-failure outlier and the best page
econ = next(r for r in results if "TheEconomist.2024.02.24" in r["image_path"])
ax.annotate("degenerate repetition\n(Economist masthead loop)",
            xy=(econ["compression_vs_gt"], econ["decode_accuracy_v2"] * 100),
            xytext=(econ["compression_vs_gt"] + 2.2, econ["decode_accuracy_v2"] * 100 + 14),
            color=CRIMSON, fontsize=8.6, fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="-", color=CRIMSON, lw=1.1))
best = max(results, key=lambda r: r["decode_accuracy_v2"])
ax.annotate("best page\n(low density, 3.0x)",
            xy=(best["compression_vs_gt"], best["decode_accuracy_v2"] * 100),
            xytext=(best["compression_vs_gt"] + 1.6, best["decode_accuracy_v2"] * 100 - 12),
            color=MUTE, fontsize=8.6, fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="-", color=MUTE, lw=1.0))

ax.set_xlabel("compression ratio (ground-truth text tokens / vision tokens)", color=TEXT, fontsize=10.5)
ax.set_ylabel("decode accuracy (%, edit-distance vs ground truth)", color=TEXT, fontsize=10.5)
ax.set_title("DeepSeek-OCR-2: measured accuracy sits below the paper's own curve on real, diverse pages",
             color=GOLD, fontsize=13, fontweight="bold", pad=14)
ax.set_xlim(0, 29)
ax.set_ylim(-5, 105)
ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=9.5, loc="upper right")
ax.tick_params(colors=MUTE)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.grid(True, color=GRID, linewidth=0.6, zorder=0)

fig.text(0.012, 0.012, "n=15, OmniDocBench sample, English text-dominant pages; decode accuracy is an edit-distance proxy, not the official OmniDocBench scorer.",
         color=MUTE, fontsize=7.6, ha="left")
fig.text(0.99, 0.012, "WITCHEER", color=MUTE, fontsize=9, ha="right", fontweight="bold")
fig.tight_layout(rect=[0, 0.02, 1, 1])
fig.savefig("reports/deepseek-ocr-2.png", dpi=150, facecolor=BG)
print("wrote reports/deepseek-ocr-2.png")
