#!/usr/bin/env python3
"""Gold & Crimson chart for the ACE-Step music-gen bench (runs on the Mac).

Panel A: seconds to generate a full 4-minute (240s) song — the 5090's two tiers
vs ACE-Step's own A100/3090 vendor claims (the "where does a gaming card land"
picture). Panel B: generation time vs song length (30/120/240s) for both tiers,
annotated with real-time factor.

Usage: python3 scripts/chart_ace.py   (reads results/ace_step/ace-step-music.json)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, GOLD, CRIMSON = "#0d0906", "#e8c44a", "#e06060"
TEXT, GRID, MUTE = "#f5e6d0", "#3a2f25", "#8a7a64"

R = json.load(open("results/ace_step/ace-step-music.json"))
T = R["tiers"]

def compute(tier, dur):
    return T[tier][str(dur)]["gen_seconds_mean"]

def rtf(tier, dur):
    return T[tier][str(dur)]["rtf_mean"]

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": TEXT,
    "ytick.color": TEXT, "axes.edgecolor": GRID, "font.size": 12,
})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6.2))

# ---- Panel A: seconds to make a 240s song --------------------------------
# bars: label, seconds, color, sublabel. subject (5090) = crimson, vendor foils = gold/mute.
bars = [
    ("RTX 3090", 10.0, MUTE, "vendor claim  <10s"),
    ("RTX 5090 · XL 4B", compute("xl", 240), CRIMSON, f"ours  {compute('xl',240):.1f}s · {rtf('xl',240):.0f}x"),
    ("RTX 5090 · 2B", compute("2b", 240), CRIMSON, f"ours  {compute('2b',240):.2f}s · {rtf('2b',240):.0f}x"),
    ("A100 80GB", 1.5, GOLD, "vendor claim  ~1-2s"),
]
labels = [b[0] for b in bars]
vals = [b[1] for b in bars]
cols = [b[2] for b in bars]
subs = [b[3] for b in bars]
ypos = range(len(bars))
axA.barh(list(ypos), vals, color=cols, height=0.62, zorder=3)
for y, v, s, c in zip(ypos, vals, subs, cols):
    axA.text(v + 0.15, y, s, va="center", ha="left", fontsize=10.5,
             color=TEXT if c == CRIMSON else MUTE, zorder=4)
axA.set_yticks(list(ypos))
axA.set_yticklabels(labels, fontsize=12.5)
axA.invert_yaxis()
axA.set_xlim(0, 11.6)
axA.set_xlabel("seconds to generate one 4-minute song  (lower = faster)")
axA.set_title("A gaming GPU writes a 4-min song in ~2s", color=GOLD, fontsize=15, pad=12, loc="left")
axA.grid(axis="x", color=GRID, lw=0.8, zorder=0)
for sp in ("top", "right", "left"):
    axA.spines[sp].set_visible(False)
axA.text(0.99, -0.14, "the song itself plays for 240s — every bar is 24–137x faster than real-time",
         transform=axA.transAxes, ha="right", va="top", fontsize=9.5, color=MUTE, style="italic")

# ---- Panel B: gen time vs song length ------------------------------------
durs = [30, 120, 240]
for tier, color, name in (("2b", CRIMSON, "2B turbo"), ("xl", GOLD, "XL 4B turbo")):
    ys = [compute(tier, d) for d in durs]
    axB.plot(durs, ys, "-o", color=color, lw=2.4, ms=7, label=name, zorder=3)
    axB.text(durs[-1], ys[-1], f"  {name}\n  {rtf(tier,240):.0f}x real-time",
             va="center", ha="left", fontsize=10, color=color)
axB.set_xlim(0, 300)
axB.set_ylim(0, max(compute("xl", 240), compute("2b", 240)) * 1.25)
axB.set_xticks(durs)
axB.set_xlabel("song length (seconds)")
axB.set_ylabel("generation compute time (seconds)")
axB.set_title("Longer songs stay cheap — RTF rises with length", color=GOLD, fontsize=15, pad=12, loc="left")
axB.grid(color=GRID, lw=0.8, zorder=0)
for sp in ("top", "right"):
    axB.spines[sp].set_visible(False)
axB.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, loc="upper left", fontsize=11)

fig.suptitle("ACE-Step 1.5 music generation on one RTX 5090 (sm_120) · DiT-only · bf16 · 8-step turbo",
             color=TEXT, fontsize=11.5, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("reports/ace-step-music.png", dpi=140)
print("wrote reports/ace-step-music.png")
