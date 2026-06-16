"""Gold & Crimson v2 chart for the ECHO maze microcosm. Two-panel small-multiple: full obs
(walls+bearing) vs walls-only (map required). In each panel, action-only (lambda=0) vs
+env-token loss (lambda=1) with seed-spread error bars. The visual itself carries the finding:
the free env-token loss is inert in behavior cloning under BOTH observation regimes. Line form
fits a trend-over-difficulty result (not a bar). Runs on the Mac (system python3 matplotlib)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = ("#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64")

PANELS = [
    ("results/echo_maze/results.json", "full obs (walls + goal bearing)"),
    ("results/echo_maze/results_walls_only.json", "walls-only (agent must map)"),
]

fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), facecolor=BG, sharey=True)
for ax, (path, title) in zip(axes, PANELS):
    d = json.load(open(path))
    sizes = d["sizes"]
    agg = d["agg"]
    ks = [str(s) for s in sizes]

    def col(arm, key):
        return [agg[arm][k][key] for k in ks]

    ax.set_facecolor(BG)
    ax.errorbar(sizes, col("action_only", "mean"), yerr=col("action_only", "std"),
                color=MUTE, marker="o", lw=2, capsize=4, label="action-only (λ=0)")
    ax.errorbar(sizes, col("echo", "mean"), yerr=col("echo", "std"),
                color=CRIMSON, marker="o", lw=2, capsize=4, label="+ env-token loss (λ=1)")
    for s, a, e in zip(sizes, col("action_only", "mean"), col("echo", "mean")):
        ax.annotate(f"{(e - a) * 100:+.1f}", (s, max(a, e)), color=GOLD, fontsize=9,
                    ha="center", va="bottom")
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s}x{s}" for s in sizes])
    ax.set_xlabel("maze size", color=TEXT)
    ax.set_title(title, color=TEXT, fontsize=11, pad=8)
    ax.tick_params(colors=TEXT)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(True, color=GRID, alpha=0.4)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

axes[0].set_ylabel("solve rate (3-seed mean ± spread)", color=TEXT)
fig.suptitle("The free env-token loss is inert in behavior cloning — under both obs regimes",
             color=GOLD, fontsize=13.5, y=0.98)
fig.text(0.5, 0.005, "ECHO maze microcosm · 10M transformer from scratch · gold labels = (λ1 − λ0) pt gap",
         color=MUTE, fontsize=8.5, ha="center")
fig.tight_layout(rect=[0, 0.02, 1, 0.95])
fig.savefig("reports/echo-maze-microcosm.png", dpi=150, facecolor=BG)
print("wrote reports/echo-maze-microcosm.png")
