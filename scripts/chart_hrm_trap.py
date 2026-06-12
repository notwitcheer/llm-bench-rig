"""t047 HRM-Text-1B chart (Gold & Crimson v2): the token_type_ids harness trap.
GSM8K-generative n=200 greedy on one RTX 5090 — paper claim vs correct protocol vs
default-harness protocol (no token_type_ids). 95% binomial CI on measured bars."""
import matplotlib.pyplot as plt
import numpy as np

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = ("#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64")

# label, acc %, color, 95% CI half-width (n=200), sublabel
BARS = [
    ("paper claim", 84.5, MUTE, 0, "(their eval,\nprotocol n/a)"),
    ("correct protocol", 79.5, GOLD, 5.6, "token_type_ids=1\n(prefixlm mask)"),
    ("default harness", 53.5, CRIMSON, 6.9, "no token_type_ids\n(causal-only mask)"),
]

fig, ax = plt.subplots(figsize=(9.5, 6.5), facecolor=BG)
ax.set_facecolor(BG)
xs = np.arange(len(BARS))
for i, (label, acc, color, ci, sub) in enumerate(BARS):
    ax.bar(i, acc, 0.58, color=color, edgecolor=TEXT, zorder=3)
    if ci:
        ax.errorbar(i, acc, yerr=ci, color=TEXT, capsize=6, linewidth=1.5, zorder=4)
    ax.annotate(f"{acc}%", (i, acc + (ci or 0)), color=TEXT, ha="center", fontsize=15,
                fontweight="bold", xytext=(0, 8), textcoords="offset points")
    ax.annotate(sub, (i, 4), color=BG, ha="center", va="bottom", fontsize=9, fontweight="bold")

# the delta arrow
ax.annotate("", xy=(2, 56), xytext=(2.42, 79),
            arrowprops=dict(arrowstyle="-|>", color=CRIMSON, linewidth=2))
ax.annotate("one missing tensor\n= -26 pts, silent", (2.45, 70), color=CRIMSON,
            ha="center", fontsize=12, fontweight="bold")

ax.set_xticks(xs)
ax.set_xticklabels([b[0] for b in BARS], color=TEXT, fontsize=12)
ax.set_xlim(-0.5, 3.1)
ax.set_ylim(0, 100)
ax.set_ylabel("GSM8K accuracy (%, generative, greedy, n=200)", color=TEXT)
ax.set_title("HRM-Text-1B: the PrefixLM eval trap, measured — RTX 5090\n"
             "standard harnesses never pass token_type_ids; the model degrades silently",
             color=GOLD, fontsize=13, pad=14)
ax.annotate("the recurrence bill: 42.9 tok/s decode (bf16, a 1.2B that runs like a ~5B) · "
            "128 KV slots · 0.88 MB/token · no llama.cpp path",
            (0.5, -0.13), xycoords="axes fraction", color=MUTE, ha="center", fontsize=9.5)
for s in ax.spines.values():
    s.set_color(GRID)
ax.tick_params(colors=MUTE)
ax.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)
fig.tight_layout()
fig.savefig("reports/hrm-trap.png", dpi=150, facecolor=BG, bbox_inches="tight")
print("wrote reports/hrm-trap.png")
