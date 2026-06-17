"""Gold & Crimson chart for the GLM-5.2 autopsy: memory to serve GLM-5.2 (smallest-quant weights +
MLA KV) vs context length, against the rig's 96GB addressable and a single H200 (141GB). Even with
MLA's ~57x KV compression, the weights floor (147GB) already exceeds both lines, and 1M context adds
~94GB of KV to ~241GB total. Runs on the Mac (system python3 matplotlib)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = ("#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64")

ctx = np.array([4096, 16384, 65536, 262144, 1048576])
labels = ["4K", "16K", "64K", "256K", "1M"]
KV_PER_TOK = (512 + 64) * 78 * 2          # bytes: (kv_lora_rank + qk_rope) x layers x bf16
kv_gb = ctx * KV_PER_TOK / 1e9            # MLA KV cache, GB
WEIGHTS = 147                            # smallest unsloth dynamic quant (~1.58-bit) of 743B
total = WEIGHTS + kv_gb

fig, ax = plt.subplots(figsize=(11, 7), facecolor=BG)
ax.set_facecolor(BG)

# weights band (flat floor) + KV band (grows with context)
ax.fill_between(ctx, 0, WEIGHTS, color=MUTE, alpha=0.30, zorder=2)
ax.fill_between(ctx, WEIGHTS, total, color=CRIMSON, alpha=0.30, zorder=2)
ax.plot(ctx, total, color=CRIMSON, lw=2.6, marker="o", markersize=7, zorder=4)

# hardware ceilings
ax.axhline(96, color=GOLD, lw=1.6, ls="--", zorder=3)
ax.axhline(141, color="#d8902f", lw=1.6, ls="--", zorder=3)

ax.annotate("weights — smallest 1.58-bit quant of 743B = 147 GB", (4300, WEIGHTS / 2),
            color=TEXT, fontsize=10, va="center", ha="left")
ax.annotate("MLA KV cache\n(1M = 94 GB)", (300000, WEIGHTS + kv_gb[-1] * 0.55),
            color=CRIMSON, fontsize=10, va="center", ha="center", fontweight="bold")
ax.annotate("RTX 5090 + 64 GB RAM = 96 GB addressable", (4300, 96), color=GOLD, fontsize=9.5,
            va="bottom", ha="left", xytext=(0, 3), textcoords="offset points")
ax.annotate("1x H200 = 141 GB  (can't even hold the weights)", (4300, 141), color="#d8902f",
            fontsize=9.5, va="top", ha="left", xytext=(0, -3), textcoords="offset points")
ax.annotate("1M context\n241 GB total", (1048576, total[-1]), color=TEXT, fontsize=11, fontweight="bold",
            va="bottom", ha="right", xytext=(-6, 8), textcoords="offset points")

ax.set_xscale("log")
ax.set_xticks(ctx)
ax.set_xticklabels(labels, color=TEXT)
ax.set_xlabel("context length (tokens)", color=TEXT)
ax.set_ylabel("memory to serve GLM-5.2  (GB)", color=TEXT)
ax.set_ylim(0, 260)
ax.set_xlim(3500, 1300000)
ax.set_title("GLM-5.2: MLA's 57x cheaper KV is real — and it still doesn't fit",
             color=GOLD, fontsize=14, pad=18)
ax.text(0.5, 1.015, "743B MoE · weights floor 147 GB + 1M MLA-KV 94 GB = 241 GB · the whole curve sits above the rig AND a single H200",
        transform=ax.transAxes, color=MUTE, fontsize=9, ha="center")
ax.tick_params(colors=TEXT)
for sp in ax.spines.values():
    sp.set_color(GRID)
ax.grid(True, which="both", color=GRID, alpha=0.30)
fig.tight_layout()
fig.savefig("reports/glm-5-2-walls.png", dpi=150, facecolor=BG)
print("wrote reports/glm-5-2-walls.png")
