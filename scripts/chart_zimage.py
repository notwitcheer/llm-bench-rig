#!/usr/bin/env python3
"""Gold & Crimson chart for the Z-Image-Turbo image-gen bench (runs on the Mac).

Panel A: seconds to generate one image across resolutions (512 -> 2048) on the
5090, annotated with images/minute and peak VRAM — the "how fast, and does it
wall" picture. Panel B: compute time vs step count at 1024px — the few-step
economy (time is linear in DiT forwards; 8 is the distilled sweet spot).

Usage: python3 scripts/chart_zimage.py   (reads results/zimage/z-image-turbo.json)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, GOLD, CRIMSON = "#0d0906", "#e8c44a", "#e06060"
TEXT, GRID, MUTE = "#f5e6d0", "#3a2f25", "#8a7a64"

R = json.load(open("results/zimage/z-image-turbo.json"))
BY_RES = R["by_resolution"]
BY_STEPS = R["by_steps"]
WALLS = R.get("vram_wall_resolutions", [])

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": TEXT,
    "ytick.color": TEXT, "axes.edgecolor": GRID, "font.size": 12,
})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6.2))

# ---- Panel A: seconds per image across resolutions -----------------------
res_ok = sorted(BY_RES, key=int)
labels = [f"{r}²" for r in res_ok]
vals = [BY_RES[r]["gen_seconds_mean"] for r in res_ok]
ypos = range(len(res_ok))
# the showcase resolution (1024) crimson, the rest gold
cols = [CRIMSON if r == "1024" else GOLD for r in res_ok]
axA.barh(list(ypos), vals, color=cols, height=0.6, zorder=3)
for y, r, v in zip(ypos, res_ok, vals):
    c = BY_RES[r]
    axA.text(v + max(vals) * 0.015, y,
             f"{v:.2f}s  ·  {c['images_per_min']:.0f} img/min  ·  {c['peak_vram_mib_max']/1024:.1f}GB",
             va="center", ha="left", fontsize=10, color=TEXT, zorder=4)
axA.set_yticks(list(ypos))
axA.set_yticklabels(labels, fontsize=12.5)
axA.invert_yaxis()
axA.set_xlim(0, max(vals) * 1.55)
axA.set_xlabel("seconds to generate one image  (8 steps, lower = faster)")
axA.set_title("A 5090 makes a 1024² image in a blink", color=GOLD, fontsize=15, pad=12, loc="left")
axA.grid(axis="x", color=GRID, lw=0.8, zorder=0)
for sp in ("top", "right", "left"):
    axA.spines[sp].set_visible(False)
sub = "every resolution fits in 32GB — no VRAM wall"
if WALLS:
    sub = f"VRAM wall (OOM) at {', '.join(f'{w}²' for w in WALLS)}"
axA.text(0.99, -0.14, sub, transform=axA.transAxes, ha="right", va="top",
         fontsize=9.5, color=MUTE, style="italic")

# ---- Panel B: step economy at 1024 ---------------------------------------
steps = sorted(BY_STEPS, key=int)
xs = [int(s) for s in steps]
ys = [BY_STEPS[s]["gen_seconds_mean"] for s in steps]
axB.plot(xs, ys, "-o", color=GOLD, lw=2.4, ms=7, zorder=3)
# mark the distilled default (8 steps) crimson
if "8" in BY_STEPS:
    axB.plot([8], [BY_STEPS["8"]["gen_seconds_mean"]], "o", color=CRIMSON, ms=13, zorder=4)
    axB.annotate("8 steps\n(distilled default)",
                 (8, BY_STEPS["8"]["gen_seconds_mean"]),
                 textcoords="offset points", xytext=(10, -28),
                 fontsize=10, color=CRIMSON)
axB.set_xlim(0, max(xs) * 1.15)
axB.set_ylim(0, max(ys) * 1.25)
axB.set_xticks(xs)
axB.set_xlabel("inference steps (= DiT forwards)")
axB.set_ylabel("generation compute time (seconds)")
axB.set_title("Time is linear in steps — few-step is the whole trick", color=GOLD, fontsize=15, pad=12, loc="left")
axB.grid(color=GRID, lw=0.8, zorder=0)
for sp in ("top", "right"):
    axB.spines[sp].set_visible(False)
axB.text(0.99, -0.14, "distilled to 8 forwards from the ~50 a normal diffusion model needs",
         transform=axB.transAxes, ha="right", va="top", fontsize=9.5, color=MUTE, style="italic")

fig.suptitle("Z-Image-Turbo (6B) text-to-image on one RTX 5090 (sm_120) · bf16 · SDPA · no compile/quant",
             color=TEXT, fontsize=11.5, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("reports/z-image-turbo.png", dpi=140)
print("wrote reports/z-image-turbo.png")
