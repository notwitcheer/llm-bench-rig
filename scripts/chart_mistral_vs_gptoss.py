"""Mistral Small 4 (MXFP4) vs gpt-oss-120B (MXFP4) — two ~120B reasoning MoEs offloaded
onto a single 32GB RTX 5090. Decode + prefill throughput. Gold & Crimson v2."""
import matplotlib.pyplot as plt
import numpy as np

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = ("#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64")

# from speed.json (both runs, llama-bench, --n-cpu-moe offload on a 32GB card)
MISTRAL = {"decode": 35.95, "prefill": 313.0, "size": 66.9, "active": "6B",  "ncm": 24, "params": "119B"}
GPTOSS  = {"decode": 46.48, "prefill": 588.0, "size": 59.0, "active": "5.1B", "ncm": 20, "params": "117B"}

groups = ["decode (tok/s)", "prefill peak (tok/s)"]
x = np.arange(len(groups))
w = 0.36

fig, ax = plt.subplots(figsize=(9.5, 6), facecolor=BG)
ax.set_facecolor(BG)
mb = ax.bar(x - w/2, [MISTRAL["decode"], MISTRAL["prefill"]], w, color=CRIMSON, edgecolor=TEXT, label="Mistral-Small-4 119B (6B act, 66.9GB, ncm24)", zorder=3)
gb = ax.bar(x + w/2, [GPTOSS["decode"], GPTOSS["prefill"]], w, color=GOLD, edgecolor=TEXT, label="gpt-oss-120B 117B (5.1B act, 59GB, ncm20)", zorder=3)
for bars in (mb, gb):
    for b in bars:
        ax.annotate(f"{b.get_height():.0f}", (b.get_x() + b.get_width()/2, b.get_height()),
                    color=TEXT, fontsize=11, ha="center", xytext=(0, 4), textcoords="offset points")

ax.set_xticks(x); ax.set_xticklabels(groups, color=TEXT, fontsize=12)
ax.set_ylabel("tokens / sec", color=TEXT)
ax.set_title("Two ~120B reasoning MoEs on ONE 32GB card (MXFP4 + RAM offload, RTX 5090)", color=GOLD, pad=16)
leg = ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=9, loc="upper left")
for s in ax.spines.values():
    s.set_color(GRID)
ax.tick_params(colors=MUTE)
ax.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)
ax.margins(y=0.12)
fig.tight_layout()
fig.savefig("reports/mistral-vs-gptoss-speed.png", dpi=150, facecolor=BG)
print("wrote reports/mistral-vs-gptoss-speed.png")
