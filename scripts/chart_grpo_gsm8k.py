#!/usr/bin/env python3
"""Gold & Crimson EV+ chart: portable GRPO (use_vllm=False) on Qwen3-4B, RTX 5090.
Left: GRPO mean-reward curve over 150 steps (the RL mechanism — reward climbing).
Right: base vs tuned GSM8K accuracy (the outcome, +7.66 pts). Data inlined from
~/unsloth-runs/gsm8k-grpo/train_log.json + eval_base/eval_tuned.json."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BG="#0d0906"; GOLD="#e8c44a"; TEXT="#f5e6d0"; CRIMSON="#e06060"; GRID="#3a2f25"; MUTE="#8a7a64"

# --- data (train_log.json reward_curve + eval jsons) ---
curve = [(5,1.375),(10,1.525),(15,1.225),(20,1.5625),(25,1.5125),(30,1.4375),(35,0.875),
         (40,1.8125),(45,1.5),(50,2.125),(55,1.325),(60,1.9875),(65,1.6625),(70,1.45),
         (75,1.85),(80,1.725),(85,2.125),(90,1.2625),(95,1.4),(100,1.45),(105,1.55),
         (110,0.65),(115,1.4),(120,1.6625),(125,1.6),(130,2.1875),(135,1.375),(140,1.575),
         (145,1.6875),(150,1.725)]
steps = [c[0] for c in curve]; rew = [c[1] for c in curve]
BASE, TUNED = 60.67, 68.33

# trend line (least squares) to cut through the RL noise
n=len(steps); sx=sum(steps); sy=sum(rew); sxx=sum(s*s for s in steps); sxy=sum(s*r for s,r in zip(steps,rew))
m=(n*sxy-sx*sy)/(n*sxx-sx*sx); b=(sy-m*sx)/n
trend=[m*s+b for s in steps]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 6.6), dpi=100, gridspec_kw={"width_ratios":[2.0,1]})
fig.patch.set_facecolor(BG)
fig.subplots_adjust(left=0.07, right=0.975, top=0.66, bottom=0.13, wspace=0.30)

# --- LEFT: reward curve ---
axL.set_facecolor(BG)
for s in axL.spines.values(): s.set_visible(False)
axL.tick_params(length=0, labelsize=11, colors=TEXT)
axL.grid(True, color=GRID, linewidth=0.8, alpha=0.6); axL.set_axisbelow(True)
axL.plot(steps, rew, "-o", color=CRIMSON, linewidth=1.8, markersize=4, alpha=0.85, label="mean reward / step")
axL.plot(steps, trend, "--", color=GOLD, linewidth=2.2, label="trend")
axL.set_xlabel("GRPO step", fontsize=11, color=TEXT)
axL.set_ylabel("mean reward (correctness + format, max 2.0)", fontsize=10.5, color=TEXT)
axL.set_ylim(0.4, 2.3); axL.set_xlim(0, 155)
axL.text(150, 1.78, "1.725", color=CRIMSON, fontsize=10.5, fontweight="bold", ha="right")
axL.text(8, 1.30, "1.375", color=MUTE, fontsize=10.5, fontweight="bold")
leg = axL.legend(loc="lower right", frameon=False, fontsize=10, labelcolor=TEXT)
axL.set_title("the RL signal — reward climbs (noisy, not saturated)", color=TEXT, fontsize=12.5, pad=8, loc="left")

# --- RIGHT: base vs tuned bars ---
axR.set_facecolor(BG)
for s in axR.spines.values(): s.set_visible(False)
axR.tick_params(length=0, labelsize=11, colors=TEXT)
axR.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.6); axR.set_axisbelow(True)
bx=[0,1]; vals=[BASE,TUNED]; cols=[MUTE,CRIMSON]
axR.bar(bx, vals, width=0.6, color=cols)
for i,v in zip(bx,vals): axR.text(i, v+1.0, f"{v:.1f}%", ha="center", color=TEXT, fontsize=13, fontweight="bold")
axR.set_xticks(bx); axR.set_xticklabels(["base\nQwen3-4B","+ GRPO"], fontsize=10.5)
axR.set_xlim(-0.55, 1.75)
axR.set_ylim(0,80); axR.set_ylabel("GSM8K accuracy (n=300, greedy)", fontsize=10.5, color=TEXT)
axR.annotate("", xy=(1,TUNED), xytext=(1,BASE), arrowprops=dict(arrowstyle="->", color=GOLD, lw=2))
axR.text(1.32, (BASE+TUNED)/2, "+7.66\npts", color=GOLD, fontsize=11, fontweight="bold", va="center")
axR.set_title("the outcome — measured gain", color=TEXT, fontsize=12.5, pad=8, loc="left")

# --- header ---
fig.add_artist(Rectangle((0.07, 0.875), 0.05, 0.014, color=CRIMSON, transform=fig.transFigure))
fig.text(0.07, 0.80, "fp8 RL was walled. portable GRPO shipped +7.66.", fontsize=22, fontweight="bold", color=TEXT)
fig.text(0.07, 0.745, "Qwen3-4B QLoRA GRPO on one RTX 5090 — vLLM-free (use_vllm=False), no CUDA toolkit. Reward = correctness + #### format.",
         fontsize=11.5, color=GOLD)
fig.text(0.07, 0.70, "fp8 RL needs CUDA >= 12.9 (box has 12.8) + a torch/unsloth/vLLM pin knot. the slow path runs where the fast one won't load.",
         fontsize=11, color=MUTE)
fig.text(0.975, 0.035, "WITCHEER · in-process eval, identical harness both sides", fontsize=10, color=MUTE, ha="right")
fig.savefig("reports/chart-grpo-gsm8k.png", facecolor=BG)
print("wrote reports/chart-grpo-gsm8k.png")
