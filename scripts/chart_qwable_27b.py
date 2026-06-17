"""Gold & Crimson chart for Qwable-3.6-27b: the distill effect is invisible on every cheap eval and
only shows up on real bugs. Parallel-coordinates 'flat-flat-cliff' — base vs distill across three axes
(quality q_avg, agentic score, SWE-bench resolve %), all matched Q4_K_M. The two lines hug on quality
and agentic, then split hard on SWE. Per-axis deltas annotated so the contrast (0.7/0.5 vs 23) is the
story. Runs on the Mac (system python3 matplotlib)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = ("#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64")

axes_labels = ["quality\nq_avg", "agentic\nscore", "SWE-bench\nresolve %"]
x = [0, 1, 2]
base   = [94.05, 98.19, 60.0]   # base quality is Q6_K (quant moves q_avg <1pt); agentic+SWE are Q4
qwable = [93.40, 97.64, 36.7]
deltas = ["Δ0.7", "Δ0.5", "Δ23  (−7 bugs)"]

fig, ax = plt.subplots(figsize=(10, 6.5), facecolor=BG)
ax.set_facecolor(BG)
ax.plot(x, base,   color=GOLD,    lw=2.5, marker="o", markersize=9, zorder=3, label="Qwen3.6-27B base (Q4)")
ax.plot(x, qwable, color=CRIMSON, lw=2.5, marker="o", markersize=9, zorder=3, label="Qwable-3.6-27b (Q4)")

for xi, b, q, d in zip(x, base, qwable, deltas):
    ax.annotate(f"{b:.1f}", (xi, b), color=GOLD, fontsize=12, fontweight="bold",
                ha="center", va="bottom", xytext=(0, 8), textcoords="offset points")
    ax.annotate(f"{q:.1f}", (xi, q), color=CRIMSON, fontsize=12, fontweight="bold",
                ha="center", va="top", xytext=(0, -10), textcoords="offset points")
    # delta callout only on the axis that matters (SWE) — the flat ones speak for themselves
    if xi == 2:
        ymid = (b + q) / 2
        ax.annotate(d, (xi, ymid), color=CRIMSON, fontsize=13, fontweight="bold",
                    ha="right", va="center", xytext=(-14, 0), textcoords="offset points")

ax.set_xticks(x)
ax.set_xticklabels(axes_labels, color=TEXT, fontsize=11)
ax.set_xlim(-0.35, 2.55)
ax.set_ylim(32, 103)
ax.set_ylabel("score (higher = better, all 0-100)", color=TEXT)
ax.set_title("every cheap eval says the distill is fine. real bugs say it gave up.",
             color=GOLD, fontsize=14, pad=18)
ax.text(0.5, 1.015, "Qwable-3.6-27b vs its Qwen3.6-27B base · matched Q4_K_M · give-ups 7→13 · quant (Q6→Q4) cost the base only 1 bug",
        transform=ax.transAxes, color=MUTE, fontsize=9.5, ha="center")
ax.tick_params(colors=TEXT)
for sp in ax.spines.values():
    sp.set_color(GRID)
ax.grid(True, axis="y", color=GRID, alpha=0.4)
leg = ax.legend(loc="lower left", facecolor=BG, edgecolor=GRID, fontsize=10, labelcolor=TEXT)
fig.tight_layout()
fig.savefig("reports/qwable-27b-flat-cliff.png", dpi=150, facecolor=BG)
print("wrote reports/qwable-27b-flat-cliff.png")
