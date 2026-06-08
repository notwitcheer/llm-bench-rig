"""Gemma 4 31B QAT + MTP treatment charts (Gold & Crimson v2).
(1) quality retention: naive-Q4 vs QAT-Q4 vs Q6_K q_avg.
(2) the hero: decode-speed ladder Q6_K -> QAT-Q4 -> +MTP, with VRAM + max context."""
import matplotlib.pyplot as plt
import numpy as np

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = ("#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64")

# --- measured (think-off, 50% MMLU/HellaSwag, full others; llama-bench tg128) ---
QUAL = [("naive Q4_0", 94.07, MUTE), ("QAT Q4_0", 94.26, CRIMSON), ("Q6_K", 94.24, GOLD)]
SPEED = [  # label, decode t/s, color, sublabel
    ("Q6_K\n(the old king)", 55.4, MUTE, "23.5GB · OOM at 32K"),
    ("QAT-Q4", 76.4, GOLD, "16.4GB · 128K ✓"),
    ("QAT-Q4 + MTP", 125.3, CRIMSON, "16.4GB · 128K ✓ · 1.67x"),
]

# Chart 1 — quality retention
fig, ax = plt.subplots(figsize=(8.5, 6), facecolor=BG); ax.set_facecolor(BG)
xs = np.arange(len(QUAL))
ax.bar(xs, [q[1] for q in QUAL], 0.55, color=[q[2] for q in QUAL], edgecolor=TEXT, zorder=3)
for i, q in enumerate(QUAL):
    ax.annotate(f"{q[1]}", (i, q[1]), color=TEXT, ha="center", fontsize=12, xytext=(0, 5), textcoords="offset points")
ax.set_xticks(xs); ax.set_xticklabels([q[0] for q in QUAL], color=TEXT, fontsize=11)
ax.set_ylabel("q_avg (5 tasks, think-off)", color=TEXT)
ax.set_ylim(92, 95.2)
ax.set_title("Gemma 4 31B: Q4 barely loses quality, and QAT ≈ naive Q4", color=GOLD, pad=16)
for s in ax.spines.values(): s.set_color(GRID)
ax.tick_params(colors=MUTE); ax.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)
fig.tight_layout(); fig.savefig("reports/gemma4-qat-quality.png", dpi=150, facecolor=BG)
print("wrote reports/gemma4-qat-quality.png")

# Chart 2 — the hero: speed ladder + context
fig, ax = plt.subplots(figsize=(9.5, 6), facecolor=BG); ax.set_facecolor(BG)
xs = np.arange(len(SPEED))
bars = ax.bar(xs, [s[1] for s in SPEED], 0.6, color=[s[2] for s in SPEED], edgecolor=TEXT, zorder=3)
for i, s in enumerate(SPEED):
    ax.annotate(f"{s[1]:.0f} tok/s", (i, s[1]), color=TEXT, ha="center", fontsize=13, fontweight="bold",
                xytext=(0, 16), textcoords="offset points")
    ax.annotate(s[3], (i, s[1]), color=MUTE, ha="center", fontsize=9, xytext=(0, 3), textcoords="offset points")
ax.set_xticks(xs); ax.set_xticklabels([s[0] for s in SPEED], color=TEXT, fontsize=11)
ax.set_ylabel("decode throughput (tok/s)", color=TEXT)
ax.set_ylim(0, 145)
ax.set_title("Gemma 4 31B: lighter (QAT) + faster (MTP) + longer context, one 32GB card", color=GOLD, pad=16)
for s in ax.spines.values(): s.set_color(GRID)
ax.tick_params(colors=MUTE); ax.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)
fig.tight_layout(); fig.savefig("reports/gemma4-qat-speed.png", dpi=150, facecolor=BG)
print("wrote reports/gemma4-qat-speed.png")
