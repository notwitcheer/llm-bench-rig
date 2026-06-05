#!/usr/bin/env python3
"""Gold & Crimson EV+ chart for the LFM2.5-VL-1.6B-Extract bench.
The finding: bulletproof JSON structure, but the values are only right ~80% of the time
(tax worst, because the model copies the total when there is no tax line).
Real numbers from results/lfm2-5-vl-1-6b-extract/vlm_results.json (50 CORD receipts)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BG="#0d0906"; GOLD="#e8c44a"; TEXT="#f5e6d0"; CRIMSON="#e06060"; GRID="#3a2f25"; MUTE="#8a7a64"

# (label, value, colour, note)
rows = [
    ("json validity",      1.00, GOLD,    "50/50 outputs parsed as strict json"),
    ("schema match (f1)",   1.00, GOLD,    "exactly the keys requested, every time"),
    ("total_price",         0.83, CRIMSON, "genuine misreads: 194,000 for 174,600"),
    ("subtotal_price",      0.84, CRIMSON, "right when the receipt prints one"),
    ("tax_price",           0.60, CRIMSON, "copies the total when there's no tax line"),
]

fig, ax = plt.subplots(figsize=(12, 6.8), dpi=100)
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
fig.subplots_adjust(left=0.22, right=0.97, top=0.68, bottom=0.13)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0, labelsize=13, colors=TEXT)

ys = list(range(len(rows)))[::-1]
for y, (label, val, c, note) in zip(ys, rows):
    ax.barh(y, val, height=0.46, color=c)
    ax.text(val + 0.012, y, f"{val*100:.0f}%", va="center", ha="left", color=TEXT, fontsize=14, fontweight="bold")
    ax.text(0.006, y + 0.33, note, va="bottom", ha="left", color=MUTE, fontsize=9.5, style="italic")

ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=12.5)
ax.set_xlim(0, 1.13); ax.set_ylim(-0.7, len(rows) - 0.1); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
ax.axvline(1.0, color=TEXT, linewidth=1.0, linestyle=(0, (4, 3)), alpha=0.45)

# header band: structure (gold) vs values (crimson)
fig.add_artist(Rectangle((0.22, 0.885), 0.055, 0.014, color=CRIMSON, transform=fig.transFigure))
fig.text(0.22, 0.815, "perfect json, ~80% of the values", fontsize=23, fontweight="bold", color=TEXT)
fig.text(0.22, 0.762, "LiquidAI LFM2.5-VL-1.6B-Extract on 50 real receipts (CORD-v2) — structure is solved, the numbers aren't",
         fontsize=12.0, color=GOLD)
fig.text(0.22, 0.715, "gold = structure   ·   crimson = field accuracy vs ground truth", fontsize=10.5, color=MUTE)
fig.text(0.97, 0.035, "WITCHEER · ~550 tok/s, 3GB VRAM on one RTX 5090 (llama.cpp)", fontsize=10.5, color=MUTE, ha="right")
fig.savefig("reports/chart-lfm2-vl-extract.png", facecolor=BG)
print("wrote reports/chart-lfm2-vl-extract.png")
