"""t052 Qwopus-Coder chart (Gold & Crimson v2), two panels, no bars:
left = dumbbell: synthetic Agentic Score vs real SWE-bench resolve per model —
the gap is the story (synthetic compresses into 90-100, reality spreads 13-63);
right = line: MTP speedup vs workload — the original Qwen3.6 head climbs with
output predictability, the coder's "natively finetuned" head is flat and lower."""
import matplotlib.pyplot as plt
import numpy as np

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = ("#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64")

# (label, synthetic agentic score, real SWE resolve %), sorted by synthetic desc
MODELS = [
    ("Qwopus3.6-27B-Coder", 100.0, 57, True),
    ("Qwen3.6-27B", 98.6, 63, False),
    ("Qwen3.5-35B-A3B base", 97.5, 53, False),
    ("Qwopus-GLM-18B", 97.1, 40, False),
    ("Nemotron-Cascade-2-30B", 96.9, 13, False),
    ("Kimi-Linear-48B-A3B", 92.9, 27, False),
    ("Granite-4.1-30B", 92.0, 20, False),
    ("Nex-N2-mini", 90.4, 37, False),
]
WORKLOADS = ["prose", "Q&A", "code", "JSON", "repetitive"]
QWEN_HEAD = [1.8, 1.9, 2.0, 2.2, 2.2]      # Qwen3.6-27B-MTP Q6_K (2026-06-02)
CODER_HEAD = [1.39, 1.36, 1.39, 1.60, 1.63]  # Qwopus-Coder Q5_K_M (today)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.2), facecolor=BG,
                               gridspec_kw={"width_ratios": [1.45, 1]})
fig.suptitle("Qwopus3.6-27B-Coder, fully measured — RTX 5090",
             color=GOLD, fontsize=15, fontweight="bold", y=0.98)

# left: dumbbell — synthetic vs real, per model
ax1.set_facecolor(BG)
ys = np.arange(len(MODELS))[::-1]
for y, (name, synth, real, is_new) in zip(ys, MODELS):
    lw = 2.6 if is_new else 1.4
    ax1.plot([real, synth], [y, y], color=CRIMSON if is_new else GRID,
             linewidth=lw, zorder=2)
    ax1.scatter([synth], [y], s=120 if is_new else 70, color=CRIMSON, zorder=3,
                marker="D" if is_new else "o")
    ax1.scatter([real], [y], s=120 if is_new else 70, color=GOLD, zorder=3,
                marker="D" if is_new else "o")
    ax1.annotate(name, ((real + synth) / 2, y + 0.22), color=TEXT if is_new else MUTE,
                 fontsize=9.5, fontweight="bold" if is_new else "normal",
                 ha="center", va="bottom")
    ax1.annotate(f"{real}", (real, y), color=GOLD, fontsize=9, va="center",
                 ha="right", xytext=(-10, 0), textcoords="offset points")
    ax1.annotate(f"{synth:.0f}", (synth, y), color=CRIMSON, fontsize=9,
                 va="center", ha="left", xytext=(8, 0), textcoords="offset points")
ax1.set_xlim(0, 112)
ax1.set_ylim(-0.8, len(MODELS))
ax1.set_yticks([])
ax1.set_xlabel("score (0 to 100 scale)", color=TEXT)
ax1.scatter([], [], color=CRIMSON, label="Agentic Score (synthetic)")
ax1.scatter([], [], color=GOLD, label="SWE-bench Verified resolve % (real, n=30)")
ax1.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=9,
           loc="lower left")
ax1.set_title("a perfect synthetic score, and a real-world loss to its own base",
              color=TEXT, fontsize=11.5, pad=10)

# right: line — MTP speedup by workload, two heads
ax2.set_facecolor(BG)
xs = np.arange(len(WORKLOADS))
ax2.plot(xs, QWEN_HEAD, color=GOLD, linewidth=2.5, marker="o", markersize=7,
         label="Qwen3.6-27B original head (Q6_K)")
ax2.plot(xs, CODER_HEAD, color=CRIMSON, linewidth=2.5, marker="D", markersize=7,
         label="Qwopus-Coder finetuned head (Q5_K_M)")
for x, (q, c) in enumerate(zip(QWEN_HEAD, CODER_HEAD)):
    ax2.annotate(f"{q:.1f}x", (x, q), color=GOLD, fontsize=9.5, ha="center",
                 xytext=(0, 9), textcoords="offset points")
    ax2.annotate(f"{c:.1f}x", (x, c), color=CRIMSON, fontsize=9.5, ha="center",
                 xytext=(0, -16), textcoords="offset points")
ax2.set_xticks(xs)
ax2.set_xticklabels(WORKLOADS, color=TEXT, fontsize=10.5)
ax2.set_ylim(1.0, 2.5)
ax2.set_ylabel("MTP speedup over base decode", color=TEXT)
ax2.set_title("the finetuned MTP head is flat where the original climbs",
              color=TEXT, fontsize=11.5, pad=10)
ax2.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=9,
           loc="upper left")

for ax in (ax1, ax2):
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTE)
ax2.grid(True, axis="y", color=GRID, linewidth=0.5, zorder=0)

fig.text(0.5, 0.015,
         "q_avg 94.1 (#2 thinking-off, beats its base at a smaller quant) · MTP 96-114 tok/s (100-tps claim holds) · "
         "67% SWE claim: 57% measured on an easier subset",
         color=MUTE, ha="center", fontsize=8.5)
fig.tight_layout(rect=[0, 0.05, 1, 0.93])
fig.savefig("reports/qwopus-coder-chart.png", dpi=150, facecolor=BG)
print("wrote reports/qwopus-coder-chart.png")
