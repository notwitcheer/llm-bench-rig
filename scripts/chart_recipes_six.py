#!/usr/bin/env python3
"""recipes, six models: what each flag does to decode speed against a 16k prefix, as a ratio to the base set.
left: q8 KV, q4 KV, -np 1 ratios per model (grouped bars, 1.0 line). right: MTP ratio where a head exists, labelled with acceptance."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG, TEXT, MUTE, GRID, CALL, RED = "#0e1420", "#eaf0f7", "#7d8ba0", "#20293a", "#e8b64c", "#e0645f"
C = {"c_fa_q8kv": "#4c9be8", "d_fa_q4kv": "#e0645f", "e_fa_np1": "#57c78a"}
MODELS = [("qwen3-8-27b-q6-k", "Qwen3.8-27B\nQ6_K"), ("gemma-4-31b-q4-0-it", "Gemma 4 31B\nQ4_0"),
          ("qwopus3-8-27b-flash-q6-k", "Qwopus3.8\nFlash Q6_K"), ("qwen3-6-35b-a3b-ud-q5-k-m", "Qwen3.6-35B\nA3B Q5_K_M"),
          ("ornith-1-5-35b-q4-k-m", "Ornith 1.5\n35B-A3B Q4"), ("nvidia-nemotron-3-5-lightning-30b-a3b-q4-k-m", "Lightning\n30B-A3B Q4")]
D = {s: {r["set"]: r for r in json.load(open(f"/opt/data/tmp/recipe/{s}.json"))["sets"]} for s, _ in MODELS}
def long(s, k): return D[s][k]["long_warm"]["perceived_tps_p50"]
def ratio(s, k): return long(s, k) / long(s, "a_base") if k in D[s] else None

fig = plt.figure(figsize=(14, 7.4), dpi=110); fig.patch.set_facecolor(BG)
gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1], left=0.06, right=0.98, top=0.76, bottom=0.21, wspace=0.14)
axL, axR = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
for ax in (axL, axR):
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0, labelsize=10.5, colors=MUTE)
    ax.grid(axis="y", color=GRID, linewidth=1, alpha=0.7); ax.set_axisbelow(True)

x = np.arange(len(MODELS)); w = 0.26
for i, (k, lab) in enumerate([("c_fa_q8kv", "KV q8_0"), ("d_fa_q4kv", "KV q4_0"), ("e_fa_np1", "--parallel 1")]):
    vals = [ratio(s, k) for s, _ in MODELS]
    bars = axL.bar(x + (i - 1) * w, vals, w, color=C[k], label=lab, zorder=3)
    for b, v in zip(bars, vals):
        axL.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{(v - 1) * 100:+.0f}%", ha="center", va="bottom", color=TEXT, fontsize=9.3)
axL.axhline(1.0, color=TEXT, lw=1, alpha=0.6)
axL.set_xticks(x); axL.set_xticklabels([m for _, m in MODELS], color=TEXT, fontsize=9.8)
axL.set_ylim(0.6, 1.2); axL.set_ylabel("decode speed vs base set, 16k prefix, p50 ratio", color=MUTE, fontsize=11)
axL.set_title("KV-cache quant and slot count: what they do to decode at depth", color=TEXT, fontsize=12.5, loc="left", pad=10)
axL.legend(loc="upper right", frameon=False, labelcolor=TEXT, fontsize=10.5, ncol=3)
axL.text(0.0, -0.2, "base = --jinja -ngl 99 -c 32768 (fa auto). all three sets add --flash-attn on, which alone changes nothing (6/6).",
         transform=axL.transAxes, color=MUTE, fontsize=9.2, clip_on=False)

mtp = [(lab.replace("\n", " "), ratio(s, "f_fa_mtp"), D[s]["f_fa_mtp"]["short"]["code"]["acceptance_rate"], long(s, "a_base"))
       for s, lab in MODELS if "f_fa_mtp" in D[s]]
xs = np.arange(len(mtp))
cols = [CALL if r >= 1 else RED for _, r, _, _ in mtp]
bars = axR.bar(xs, [r for _, r, _, _ in mtp], 0.55, color=cols, zorder=3)
for b, (lab, r, acc, base) in zip(bars, mtp):
    axR.text(b.get_x() + b.get_width() / 2, r + 0.03, f"{r:.2f}x", ha="center", va="bottom", color=TEXT, fontsize=11, fontweight="bold")
axR.axhline(1.0, color=TEXT, lw=1, alpha=0.6)
axR.set_xticks(xs)
axR.set_xticklabels([l.replace(" Q6_K", "\nQ6_K").replace(" Q4_0", "\nQ4_0").replace(" 30B-A3B Q4", "\n30B-A3B Q4").replace("Qwopus3.8 Flash", "Qwopus3.8\nFlash") + f"\naccept {acc:.2f}\nbase {base:.0f} tok/s" for l, _, acc, base in mtp], color=TEXT, fontsize=9.3)
axR.set_ylim(0, 2.6); axR.tick_params(axis="x", pad=2); axR.set_ylabel("decode speed with MTP head vs without, 16k prefix", color=MUTE, fontsize=11)
axR.set_title("MTP draft head: pays below ~80 tok/s, loses at 360", color=TEXT, fontsize=12.5, loc="left", pad=10)

fig.text(0.06, 0.93, "six llama-server flags, six models, one 16k prompt: q4_0 KV is a 17 to 29% decode tax on all six",
         color=TEXT, fontsize=16, fontweight="bold", ha="left")
fig.text(0.06, 0.875, "RTX 5090, llama.cpp b10371, ctx 32768, each flag set served on its own llama-server, 8 requests per cell against a 16,384-token cached system prompt, temperature 0, p50. "
         "q8_0 KV saves 0.9 to 2.6 GiB on dense models and 0.1 to 0.25 GiB on 3B-active MoEs. -np 1 is free everywhere and +10% on Gemma.",
         color=MUTE, fontsize=10.2, ha="left", wrap=True)
fig.text(0.98, 0.965, "witcheer / llm-bench-rig", color=MUTE, fontsize=11, ha="right")
fig.text(0.06, 0.035, "sources: results/<slug>/recipe/recipe.json, 2026-09-09 (recipe_bench.py served lane). 40-item GPQA-diamond spot checks on base / q8 / q4 KV: within 2 items on five models; Ornith q4 KV 20 -> 15 (full pass queued). "
         "pages: reports/recipes/",
         color=MUTE, fontsize=8.6, ha="left", wrap=True)
out = "/opt/data/tmp/recipe/recipes-six.png"; fig.savefig(out, facecolor=BG); print(out)
