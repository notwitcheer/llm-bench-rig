#!/usr/bin/env python3
"""Gold & Crimson slope chart for the LTX-2.3 audio-video bench (runs on the Mac).

One slope per prompt, 768x512 -> 1280x704 (97 frames, ~4s of video each): the
finding is the flat slope — 2.5x the pixels costs ~14% more wall time, because
per-clip time is dominated by VAE decode + mp4/AAC encode, not the DiT. The
first-gen-at-a-new-shape outlier is highlighted crimson (dashed + annotated),
identity is carried by direct labels (never colour alone).

Usage: python3 scripts/chart_ltx.py   (reads reports/assets/ltx/bench/synth.json)
"""
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, GOLD, CRIMSON = "#0d0906", "#e8c44a", "#e06060"
TEXT, GRID, MUTE = "#f5e6d0", "#3a2f25", "#8a7a64"

payload = json.load(open("reports/assets/ltx/bench/synth.json"))
by_prompt = defaultdict(dict)
for r in payload["records"]:
    by_prompt[r["name"]][f"{r['width']}x{r['height']}"] = r

CFG_L, CFG_R = "768x512", "1280x704"
OUTLIER = "puppy-garden"  # first 768x512 gen after the smaller warm-up shape

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": TEXT,
    "ytick.color": TEXT, "axes.edgecolor": GRID, "font.size": 12,
})

fig, ax = plt.subplots(figsize=(11.5, 7))

def dodge(items, min_gap=1.3):
    """items: [(y, payload)] -> label y positions at least min_gap apart."""
    out = {}
    prev = None
    for y, key in sorted(items):
        ly = y if prev is None else max(y, prev + min_gap)
        out[key] = ly
        prev = ly
    return out

names = list(by_prompt)
lab_l = dodge([(by_prompt[n][CFG_L]["gen_seconds"], n) for n in names])
lab_r = dodge([(by_prompt[n][CFG_R]["gen_seconds"], n) for n in names])

vals_l, vals_r = [], []
for name, cfgs in by_prompt.items():
    yl, yr = cfgs[CFG_L]["gen_seconds"], cfgs[CFG_R]["gen_seconds"]
    vals_l.append(yl)
    vals_r.append(yr)
    is_out = name == OUTLIER
    color = CRIMSON if is_out else MUTE
    ax.plot([0, 1], [yl, yr], color=color, lw=2,
            ls="--" if is_out else "-", marker="o", markersize=8, zorder=3)
    ax.text(-0.04, lab_l[name], f"{name}  {yl:.0f}s", ha="right", va="center",
            fontsize=10.5, color=TEXT, zorder=4)
    ax.text(1.04, lab_r[name], f"{name}  {yr:.0f}s", ha="left", va="center",
            fontsize=10.5, color=TEXT, zorder=4)

mean_l = sum(vals_l) / len(vals_l)
mean_r = sum(vals_r) / len(vals_r)
for x, m in ((0, mean_l), (1, mean_r)):
    ax.plot([x - 0.03, x + 0.03], [m, m], color=GOLD, lw=3, zorder=5)
ax.text(-0.04, mean_l + 2.4, f"mean {mean_l:.1f}s", ha="right", va="center",
        fontsize=10.5, color=GOLD, style="italic")
ax.text(1.04, mean_r - 2.4, f"mean {mean_r:.1f}s", ha="left", va="center",
        fontsize=10.5, color=GOLD, style="italic")

out = by_prompt[OUTLIER][CFG_L]["gen_seconds"]
ax.annotate("first gen at a new shape pays one-time\nlazy init (~+20s): steady state ≈ 39s",
            xy=(0.02, out), xytext=(0.16, out - 3.6), fontsize=10, color=CRIMSON,
            arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=1.2))

video_s = 97 / 24.0
ax.set_xlim(-0.55, 1.45)
ax.set_xticks([0, 1])
ax.set_xticklabels(
    [f"768×512\n{mean_l / video_s:.1f}s per video-second",
     f"1280×704\n{mean_r / video_s:.1f}s per video-second"],
    fontsize=12.5)
ax.set_ylabel("seconds per 97-frame clip (~4s of video, with synced audio)")
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
for sp in ("top", "right", "left", "bottom"):
    ax.spines[sp].set_visible(False)

ax.set_title("2.5x the pixels, ~14% more time", color=GOLD, fontsize=17, pad=30, loc="left")
ax.text(0, 1.045, "LTX-2.3 distilled on one RTX 5090: wall time is decode/encode-bound, "
                  "not DiT-bound · peak VRAM flat at 24.2GB in both configs",
        transform=ax.transAxes, fontsize=10.5, color=TEXT)
fig.text(0.985, 0.015,
         "LTX-2.3 22B distilled (8+4 step) · fp8-cast · offload none · tiled VAE · SDPA · "
         "seed 42 · 5 prompts × 2 configs · means include the flagged first-gen outlier · h264 + AAC 48kHz out",
         ha="right", fontsize=9, color=MUTE, style="italic")

plt.tight_layout()
plt.savefig("reports/ltx-2.3.png", dpi=160, bbox_inches="tight")
print("wrote reports/ltx-2.3.png")
