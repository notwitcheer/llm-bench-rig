"""t048 LocateAnything-3B chart (Gold & Crimson v2), two panels:
left = ScreenSpot-Pro claim vs measured + the text/icon split (the real fault line);
right = accuracy vs screenshot resolution (the consumer-card downscale penalty).
Full 1,581 instructions, RTX 5090, greedy, hybrid mode, in_token_limit 12288."""
import matplotlib.pyplot as plt
import numpy as np

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = ("#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64")

LEFT = [  # label, acc %, color, sublabel
    ("paper claim", 60.3, MUTE, "H100"),
    ("measured", 55.3, GOLD, "5090"),
    ("text\ntargets", 63.2, GOLD, "n=977"),
    ("icon\ntargets", 42.7, CRIMSON, "n=604"),
]
RIGHT = [  # label, acc %, n  (resolution buckets; <=1080p omitted, n=19)
    ("1440p to 2.5K", 58.0, 849),
    ("4K", 53.3, 632),
    ("above 4K", 48.1, 81),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), facecolor=BG,
                               gridspec_kw={"width_ratios": [1.3, 1]})
fig.suptitle("LocateAnything-3B on ScreenSpot-Pro — RTX 5090, measured",
             color=GOLD, fontsize=16, fontweight="bold", y=0.98)

ax1.set_facecolor(BG)
xs = np.arange(len(LEFT))
ax1.bar(xs, [b[1] for b in LEFT], 0.6, color=[b[2] for b in LEFT], edgecolor=TEXT, zorder=3)
for i, (label, acc, color, sub) in enumerate(LEFT):
    ax1.annotate(f"{acc}%", (i, acc), color=TEXT, ha="center", fontsize=13,
                 fontweight="bold", xytext=(0, 6), textcoords="offset points")
    ax1.annotate(sub, (i, 3), color=BG, ha="center", va="bottom", fontsize=9, fontweight="bold")
ax1.set_xticks(xs)
ax1.set_xticklabels([b[0] for b in LEFT], color=TEXT, fontsize=11)
ax1.set_ylim(0, 75)
ax1.set_ylabel("GUI grounding accuracy (%, point in target box)", color=TEXT)
ax1.set_title("the fault line is text vs icon, not the headline",
              color=TEXT, fontsize=12, pad=10)

ax2.set_facecolor(BG)
xs = np.arange(len(RIGHT))
ax2.bar(xs, [b[1] for b in RIGHT], 0.55, color=CRIMSON, edgecolor=TEXT, zorder=3)
for i, (label, acc, n) in enumerate(RIGHT):
    ax2.annotate(f"{acc}%", (i, acc), color=TEXT, ha="center", fontsize=13,
                 fontweight="bold", xytext=(0, 6), textcoords="offset points")
    ax2.annotate(f"n={n}", (i, 3), color=BG, ha="center", va="bottom", fontsize=9,
                 fontweight="bold")
ax2.set_xticks(xs)
ax2.set_xticklabels([b[0] for b in RIGHT], color=TEXT, fontsize=11)
ax2.set_ylim(0, 75)
ax2.set_title("bigger screenshots, harder squeeze\n(32GB forces extra downscale: 12288 of 25600 patches)",
              color=TEXT, fontsize=11, pad=10)

for ax in (ax1, ax2):
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)

fig.text(0.5, 0.015,
         "PBD parallel box decode: 100 to 207 tok/s (2.1x) on the SDPA fallback · "
         "7.8GB inference · official native-res protocol needs a 40.6GB tensor",
         color=MUTE, ha="center", fontsize=9.5)
fig.tight_layout(rect=[0, 0.05, 1, 0.93])
fig.savefig("reports/la-screenspot-chart.png", dpi=150, facecolor=BG)
print("wrote reports/la-screenspot-chart.png")
