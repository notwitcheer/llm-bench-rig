#!/usr/bin/env python3
"""recipes pair 1: which llama-server flags matter on a 5090. two models x six flag sets.
left: decode tok/s against a 16k prefix (the agent case), grouped bars per set, two models.
right: VRAM peak per set, same grouping. data: results/<slug>/recipe/recipe.json."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG, TEXT, MUTE, GRID, CALL = "#0e1420", "#eaf0f7", "#7d8ba0", "#20293a", "#e8b64c"
C_Q, C_G = "#4c9be8", "#e8b64c"
SETS = ["a_base", "b_fa", "c_fa_q8kv", "d_fa_q4kv", "e_fa_np1", "f_fa_mtp"]
LBL = ["base", "+fa", "+fa\nq8 KV", "+fa\nq4 KV", "+fa\n-np 1", "+fa\nMTP n=2"]
q = json.load(open("/opt/data/tmp/recipe/qwen3-8-27b-q6-k.json"))
g = json.load(open("/opt/data/tmp/recipe/gemma-4-31b-q4-0-it.json"))
def col(d, f):
    m = {r["set"]: r for r in d["sets"]}
    return [f(m[s]) for s in SETS]
q_long = col(q, lambda r: r["long_warm"]["perceived_tps_p50"]); g_long = col(g, lambda r: r["long_warm"]["perceived_tps_p50"])
q_code = col(q, lambda r: r["short"]["code"]["perceived_tps_p50"]); g_code = col(g, lambda r: r["short"]["code"]["perceived_tps_p50"])
q_vr = col(q, lambda r: r["vram_peak_mib"] / 1024); g_vr = col(g, lambda r: r["vram_peak_mib"] / 1024)

fig = plt.figure(figsize=(14, 7.4), dpi=110); fig.patch.set_facecolor(BG)
gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], left=0.06, right=0.98, top=0.76, bottom=0.16, wspace=0.16)
axL, axR = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
for ax in (axL, axR):
    ax.set_facecolor(BG)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0, labelsize=11, colors=MUTE)
    ax.grid(axis="y", color=GRID, linewidth=1, alpha=0.7); ax.set_axisbelow(True)
    ax.set_xticks(range(len(SETS))); ax.set_xticklabels(LBL, color=TEXT, fontsize=10.5)

x = np.arange(len(SETS)); w = 0.36
for ax, qv, gv, qc, gc, ylab, title, fmt in (
    (axL, q_long, g_long, q_code, g_code, "decode tok/s, p50", "decode against a 16k-token context", "{:.0f}"),
    (axR, q_vr, g_vr, None, None, "VRAM peak, GiB (32k ctx)", "what each set costs in VRAM", "{:.1f}"),
):
    b1 = ax.bar(x - w / 2, qv, w, color=C_Q, label="Qwen3.8-27B Q6_K", zorder=3)
    b2 = ax.bar(x + w / 2, gv, w, color=C_G, label="Gemma 4 31B Q4_0 (QAT)", zorder=3)
    for bars, vals in ((b1, qv), (b2, gv)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() - (4 if ax is axL else 0.4), fmt.format(v), ha="center", va="top", color=BG, fontsize=9.5, fontweight="bold")
    if qc is not None:
        ax.scatter(x - w / 2, qc, s=40, marker="_", color=TEXT, linewidth=2.2, zorder=5)
        ax.scatter(x + w / 2, gc, s=40, marker="_", color=TEXT, linewidth=2.2, zorder=5)
        ax.plot([], [], marker="_", ls="", color=TEXT, markersize=9, markeredgewidth=2.2, label="tick: short code prompt, same set")
    ax.set_ylabel(ylab, color=MUTE, fontsize=11)
    ax.set_title(title, color=TEXT, fontsize=12.5, loc="left", pad=10)
axL.set_ylim(0, 165); axR.set_ylim(15, 27)
axL.legend(loc="upper left", frameon=False, labelcolor=TEXT, fontsize=10.5)
axL.annotate("q4 KV: -17% / -23%\nat depth", (3, max(q_long[3], g_long[3]) + 6), xytext=(0, 40), textcoords="offset points", ha="center", color=CALL, fontsize=10,
             arrowprops=dict(arrowstyle="-", color=CALL, lw=0.9))

fig.text(0.06, 0.93, "which llama-server flags matter on a 5090: two models, six flag sets, measured on a 16k prompt",
         color=TEXT, fontsize=17, fontweight="bold", ha="left")
fig.text(0.06, 0.875, "llama.cpp b10371, ctx 32768, 8 requests per cell, temperature 0, p50. MTP n=2 pays 2.2x (Qwen, embedded head, 0.85 accept) and 1.5x (Gemma, community head, 0.39). "
         "q4 KV cache costs 17 to 23% of decode at depth for 1.4 to 4.1 GiB. flash-attn: no change. -np 1: free, +10% on Gemma at depth.",
         color=MUTE, fontsize=10.2, ha="left", wrap=True)
fig.text(0.98, 0.965, "witcheer / llm-bench-rig", color=MUTE, fontsize=11, ha="right")
fig.text(0.06, 0.05, "sources: results/<slug>/recipe/recipe.json (recipe_bench.py served lane, 2026-09-09). 16k prefix = synthetic repo dump sized with /tokenize, cache_prompt warm after one primer. "
         "GPQA-diamond 40-item spot checks on base / q8 KV / q4 KV moved by at most 2 items on both models. reports/recipes/.",
         color=MUTE, fontsize=8.6, ha="left", wrap=True)
out = "/opt/data/tmp/recipe/recipes-pair1.png"; fig.savefig(out, facecolor=BG); print(out)
