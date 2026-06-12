"""t051 MTP reaction chart (Gold & Crimson v2), two panels:
left = the news anchor: Gemma 4 31B decode ladder Q6_K -> QAT-Q4 -> +MTP (t039, 2026-06-08);
right = the mechanism: MTP gain tracks output predictability, Qwen3.6-27B 5-workload sweep
(2026-06-02, bench_mtp_workload.sh). Both measured on the RTX 5090."""
import matplotlib.pyplot as plt
import numpy as np

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = ("#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64")

GEMMA = [  # label, decode t/s, color, sublabel
    ("Q6_K", 55.4, MUTE, "23.5GB\nOOM at 32K"),
    ("QAT-Q4", 76.4, GOLD, "16.4GB\n128K ok"),
    ("QAT-Q4\n+ MTP", 125.3, CRIMSON, "16.4GB\n128K ok"),
]
QWEN = [  # label, base t/s, MTP t/s  (n_predict 256, temp 0, llama-server timings)
    ("free\nprose", 62, 113),
    ("Q&A", 62, 120),
    ("code", 62, 123),
    ("JSON", 62, 137),
    ("repetitive", 62, 137),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6), facecolor=BG,
                               gridspec_kw={"width_ratios": [1, 1.4]})
fig.suptitle("MTP speculative decode on the RTX 5090 — measured", color=GOLD,
             fontsize=16, fontweight="bold", y=0.98)

# left: gemma ladder (the news)
ax1.set_facecolor(BG)
xs = np.arange(len(GEMMA))
ax1.bar(xs, [g[1] for g in GEMMA], 0.6, color=[g[2] for g in GEMMA], edgecolor=TEXT, zorder=3)
for i, g in enumerate(GEMMA):
    ax1.annotate(f"{g[1]:.0f} tok/s", (i, g[1]), color=TEXT, ha="center", fontsize=12,
                 fontweight="bold", xytext=(0, 6), textcoords="offset points")
    ax1.annotate(g[3], (i, 4), color=BG if g[2] == MUTE else BG, ha="center", fontsize=8.5,
                 va="bottom", fontweight="bold")
ax1.annotate("1.67x", (2, 95), color=BG, ha="center", fontsize=14, fontweight="bold")
ax1.set_xticks(xs)
ax1.set_xticklabels([g[0] for g in GEMMA], color=TEXT, fontsize=11)
ax1.set_ylim(0, 160)
ax1.set_ylabel("decode throughput (tok/s)", color=TEXT)
ax1.set_title("gemma 4 31B: the MTP head in the ladder", color=TEXT, fontsize=12, pad=10)

# right: workload spread (the mechanism)
ax2.set_facecolor(BG)
xs = np.arange(len(QWEN))
w = 0.38
ax2.bar(xs - w / 2, [d[1] for d in QWEN], w, color=GOLD, edgecolor=TEXT, zorder=3, label="base decode")
ax2.bar(xs + w / 2, [d[2] for d in QWEN], w, color=CRIMSON, edgecolor=TEXT, zorder=3, label="+ MTP draft head")
for i, (label, base, mtp) in enumerate(QWEN):
    ax2.annotate(f"{mtp / base:.1f}x", (i + w / 2, mtp), color=TEXT, ha="center",
                 fontsize=12, fontweight="bold", xytext=(0, 6), textcoords="offset points")
ax2.set_xticks(xs)
ax2.set_xticklabels([d[0] for d in QWEN], color=TEXT, fontsize=10.5)
ax2.set_ylim(0, 160)
ax2.set_title("the mechanism: gain tracks output predictability (qwen3.6-27B)",
              color=TEXT, fontsize=12, pad=10)
ax2.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=10, loc="upper left")

for ax in (ax1, ax2):
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)

fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("reports/mtp-reaction-chart.png", dpi=150, facecolor=BG)
print("wrote reports/mtp-reaction-chart.png")
