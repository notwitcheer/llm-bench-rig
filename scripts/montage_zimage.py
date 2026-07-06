#!/usr/bin/env python3
"""Contact-sheet montage of the Z-Image-Turbo 1024px samples (runs on the Mac).

The see-it-instantly artifact — one grid of the eight prompt categories at 8
steps, so the speed numbers land next to the actual output. Reads the staged
samples in reports/assets/zimage/, writes reports/assets/zimage-montage.png
(HF-only, like the ACE-Step wavs).

Usage: PYTHONPATH=. python3 scripts/montage_zimage.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

BG, GOLD, TEXT, MUTE = "#0d0906", "#e8c44a", "#f5e6d0", "#8a7a64"

# ordered for impact: the "hard-case wins" up front (text render, counting)
TILES = [
    ("text",         'renders "WITCHEER" — exact'),
    ("counting",     "exactly 3 ducks"),
    ("portrait",     "photoreal, 85mm"),
    ("complex",      "rainy neon night market"),
    ("two-object",   "red teapot + blue mug"),
    ("spatial",      "cat on books, plant left"),
    ("landscape",    "alpine valley, golden hour"),
    ("illustration", "flat vector fox"),
]

fig, axes = plt.subplots(2, 4, figsize=(16, 8.9), facecolor=BG)
for ax, (name, cap) in zip(axes.flat, TILES):
    img = mpimg.imread(f"reports/assets/zimage/{name}-1024-8st.png")
    ax.imshow(img)
    ax.set_title(name, color=GOLD, fontsize=13, pad=5, loc="left")
    ax.set_xlabel(cap, color=MUTE, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#3a2f25")

fig.suptitle("Z-Image-Turbo (6B) · 1024² · 8 steps · ~3.2s each on one RTX 5090 (sm_120) · bf16, no compile",
             color=TEXT, fontsize=13, y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.965])
fig.savefig("reports/assets/zimage-montage.png", dpi=110, facecolor=BG)
print("wrote reports/assets/zimage-montage.png")
