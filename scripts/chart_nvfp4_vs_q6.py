#!/usr/bin/env python3
"""Gold & Crimson EV+ chart: NVFP4 vs Q4_K_M vs Q6_K on Qwen3.6-27B (RTX 5090, llama.cpp b9365).
The finding: NVFP4's Blackwell FP4 cores win prefill big; decode tracks footprint, so the decode
gain is small vs equal-size Q4 and large vs heavier Q6. Real numbers from
results/nvfp4-vs-q6-q4-compare.json."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BG="#0d0906"; GOLD="#e8c44a"; TEXT="#f5e6d0"; CRIMSON="#e06060"; GRID="#3a2f25"; MUTE="#8a7a64"

d = json.load(open("results/nvfp4-vs-q6-q4-compare.json"))
sp = d["speed"]
ctx = [128,512,2048,4096,8192,16384]
keys = [f"pp{c}" for c in ctx]
nv = [sp["NVFP4"][k] for k in keys]
q4 = [sp["Q4_K_M"][k] for k in keys]
q6 = [sp["Q6_K"][k] for k in keys]
tg = {"NVFP4":sp["NVFP4"]["tg128"], "Q4_K_M":sp["Q4_K_M"]["tg128"], "Q6_K":sp["Q6_K"]["tg128"]}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 6.6), dpi=100, gridspec_kw={"width_ratios":[2.1,1]})
fig.patch.set_facecolor(BG)
fig.subplots_adjust(left=0.07, right=0.975, top=0.66, bottom=0.13, wspace=0.28)

# --- LEFT: prefill across context (lines) ---
axL.set_facecolor(BG)
for s in axL.spines.values(): s.set_visible(False)
axL.tick_params(length=0, labelsize=11, colors=TEXT)
axL.grid(True, color=GRID, linewidth=0.8, alpha=0.6)
axL.set_axisbelow(True)
x = list(range(len(ctx)))
axL.plot(x, nv, "-o", color=CRIMSON, linewidth=2.6, markersize=6, label="NVFP4 (4-bit, 14.6GB)")
axL.plot(x, q4, "-o", color=GOLD, linewidth=2.2, markersize=5, label="Q4_K_M (4-bit, 15.7GB)")
axL.plot(x, q6, "-o", color=MUTE, linewidth=2.0, markersize=5, label="Q6_K (6.5-bit, 21GB)")
axL.set_xticks(x); axL.set_xticklabels([str(c) for c in ctx])
axL.set_xlabel("prompt length (tokens)", fontsize=11, color=TEXT)
axL.set_ylabel("prefill throughput (tok/s)", fontsize=11, color=TEXT)
axL.set_ylim(0, 6000)
axL.text(0.5, nv[1]+170, f"+{d['deltas']['NVFP4_vs_Q4_K_M']['pp512']:.0f}% vs Q4   ·   +{d['deltas']['NVFP4_vs_Q6_K']['pp512']:.0f}% vs Q6",
         color=CRIMSON, fontsize=10.5, fontweight="bold")
leg = axL.legend(loc="lower left", frameon=False, fontsize=10.5, labelcolor=TEXT)
axL.set_title("prefill — compute-bound, FP4 cores win", color=TEXT, fontsize=12.5, pad=8, loc="left")

# --- RIGHT: decode bars ---
axR.set_facecolor(BG)
for s in axR.spines.values(): s.set_visible(False)
axR.tick_params(length=0, labelsize=11, colors=TEXT)
axR.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.6); axR.set_axisbelow(True)
bx=[0,1,2]; vals=[tg["NVFP4"],tg["Q4_K_M"],tg["Q6_K"]]; cols=[CRIMSON,GOLD,MUTE]
axR.bar(bx, vals, width=0.62, color=cols)
for i,v in zip(bx,vals): axR.text(i, v+1.2, f"{v:.0f}", ha="center", color=TEXT, fontsize=12, fontweight="bold")
axR.set_xticks(bx); axR.set_xticklabels(["NVFP4","Q4_K_M","Q6_K"], fontsize=10)
axR.set_ylim(0,100); axR.set_ylabel("decode tg128 (tok/s)", fontsize=11, color=TEXT)
axR.text(0, tg["NVFP4"]+8, f"+{d['deltas']['NVFP4_vs_Q4_K_M']['tg128']:.0f}% vs Q4\n+{d['deltas']['NVFP4_vs_Q6_K']['tg128']:.0f}% vs Q6",
         ha="center", color=CRIMSON, fontsize=9.5, fontweight="bold")
axR.set_title("decode — footprint-bound, small gain", color=TEXT, fontsize=12.5, pad=8, loc="left")

# --- header ---
fig.add_artist(Rectangle((0.07, 0.875), 0.05, 0.014, color=CRIMSON, transform=fig.transFigure))
fig.text(0.07, 0.80, "nvfp4: a big prefill win, a small decode one", fontsize=22, fontweight="bold", color=TEXT)
fig.text(0.07, 0.745, "Qwen3.6-27B on one RTX 5090 (llama.cpp b9365, Blackwell-native FP4). prefill is compute; decode is memory bandwidth.",
         fontsize=11.5, color=GOLD)
fig.text(0.07, 0.70, "and the 4-bit tax is tiny: 93.2 vs 94.0 q_avg vs Q6_K  ·  -30% VRAM (17.3 vs 23.5 GB)",
         fontsize=11, color=MUTE)
fig.text(0.975, 0.035, "WITCHEER · quant-only (spec decode off both sides)", fontsize=10, color=MUTE, ha="right")
fig.savefig("reports/chart-nvfp4-vs-q6.png", facecolor=BG)
print("wrote reports/chart-nvfp4-vs-q6.png")
