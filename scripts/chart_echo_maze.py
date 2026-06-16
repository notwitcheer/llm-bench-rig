"""Gold & Crimson v2 chart: solve-rate vs maze size, action-only vs +env-token, with seed
spread, gap annotated. A line chart fits a trend-over-difficulty finding (not a bar). Reads
results/echo_maze/results.json. Runs on the Mac (system python3 matplotlib)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = ("#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64")

d = json.load(open("results/echo_maze/results.json"))
sizes = d["sizes"]
agg = d["agg"]
ks = [str(s) for s in sizes]   # json dict keys are strings


def col(arm, key):
    return [agg[arm][k][key] for k in ks]


fig, ax = plt.subplots(figsize=(9.5, 6.5), facecolor=BG)
ax.set_facecolor(BG)
ax.errorbar(sizes, col("action_only", "mean"), yerr=col("action_only", "std"),
            color=MUTE, marker="o", lw=2, capsize=4, label="action-only (λ=0)")
ax.errorbar(sizes, col("echo", "mean"), yerr=col("echo", "std"),
            color=CRIMSON, marker="o", lw=2, capsize=4, label="+ env-token loss (λ=1)")
for s, a, e in zip(sizes, col("action_only", "mean"), col("echo", "mean")):
    ax.annotate(f"{(e - a) * 100:+.0f} pt", (s, e), color=GOLD, fontsize=10, ha="center", va="bottom")
ax.set_xticks(sizes)
ax.set_xticklabels([f"{s}x{s}" for s in sizes])
ax.set_xlabel("maze size", color=TEXT)
ax.set_ylabel("solve rate", color=TEXT)
ax.set_title("ECHO maze microcosm: does a free env-token loss buy a better policy?",
             color=GOLD, pad=16, fontsize=12.5)
ax.tick_params(colors=TEXT)
for sp in ax.spines.values():
    sp.set_color(GRID)
ax.grid(True, color=GRID, alpha=0.4)
ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT)
fig.tight_layout()
fig.savefig("reports/echo-maze-microcosm.png", dpi=150, facecolor=BG)
print("wrote reports/echo-maze-microcosm.png")
