"""Gold & Crimson chart for Qwable-v1: the distillation pipeline regressed the model. A
descending slope of Agentic Score across the 3 stages (base -> +reasoning-distill -> +agentic-SFT),
with the real SWE-bench resolve annotated at the base and Qwable endpoints. Line form fits a
monotonic-decline finding. Runs on the Mac (system python3 matplotlib)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = ("#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64")

stages = ["Qwen3.6-35B-A3B\n(vanilla base)", "+ Opus-4.7\nreasoning distill", "+ Fable-5 agentic SFT\n= Qwable-v1"]
agentic = [99.58, 97.92, 96.25]
swe = {0: "SWE-bench\n19/30 (63%)\n9 give-ups", 2: "SWE-bench\n11/30 (37%)\n16 give-ups"}
x = [0, 1, 2]

fig, ax = plt.subplots(figsize=(10, 6.5), facecolor=BG)
ax.set_facecolor(BG)
ax.plot(x, agentic, color=CRIMSON, lw=2.5, marker="o", markersize=9, zorder=3)
for xi, a in zip(x, agentic):
    ax.annotate(f"{a:.2f}", (xi, a), color=CRIMSON, fontsize=12, fontweight="bold",
                ha="center", va="bottom", xytext=(0, 8), textcoords="offset points")
# real SWE-bench callouts at the endpoints, in gold
for xi, txt in swe.items():
    ax.annotate(txt, (xi, agentic[xi]), color=GOLD, fontsize=10, ha="center", va="top",
                xytext=(0, -22), textcoords="offset points")
ax.set_xticks(x)
ax.set_xticklabels(stages, color=TEXT, fontsize=10)
ax.set_ylabel("Agentic Score (native tool-calling bench)", color=TEXT)
ax.set_ylim(94.5, 100.3)
ax.set_title("Each distillation step made Qwable a worse agentic coder",
             color=GOLD, fontsize=14, pad=18)
ax.text(0.5, 1.015, "Qwen3.6-35B-A3B · Q5_K_M · controlled 3-way, same bench — synthetic AND real decline",
        transform=ax.transAxes, color=MUTE, fontsize=9.5, ha="center")
ax.tick_params(colors=TEXT)
for sp in ax.spines.values():
    sp.set_color(GRID)
ax.grid(True, axis="y", color=GRID, alpha=0.4)
fig.tight_layout()
fig.savefig("reports/qwable-decline.png", dpi=150, facecolor=BG)
print("wrote reports/qwable-decline.png")
