#!/usr/bin/env python3
"""spec decode + prefix cache on the box's served model (Qwen3.8-27B Q6_K, one RTX 5090), rerun of chart_spec_cache. Two panels:
left: decode tok/s by workload x mode (base / MTP n2 / MTP n4 / ngram), acceptance printed on MTP bars;
right: time to first token, cold vs warm cache, two prompt sizes, log axis, plus the stacked cell.
Data: results/qwen3-8-27b-q6-k-speccache/records.jsonl (304 records, 2026-09-06), p50 per cell."""
import json, statistics as st, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG, TEXT, MUTE, GRID, CALL = "#0e1420", "#eaf0f7", "#7d8ba0", "#20293a", "#e8b64c"
COL = {"base": "#7d8ba0", "mtp_n2": "#4c9be8", "mtp_n4": "#1f9e8a", "ngram": "#a86fe0"}
NAME = {"base": "no speculation", "mtp_n2": "MTP draft, n=2 (as deployed)", "mtp_n4": "MTP draft, n=4", "ngram": "n-gram lookup"}

recs = [json.loads(l) for l in open("results/qwen3-8-27b-q6-k-speccache/records.jsonl")]
g = collections.defaultdict(list)
for r in recs: g[(r["leg"], r["mode"], r["workload"])].append(r)
def cell(leg, mode, wl): return g[(leg, mode, wl)]
def p50(v): return st.median(v)
def acc(rs):
    dn = sum(r["timings"].get("draft_n", 0) or 0 for r in rs); da = sum(r["timings"].get("draft_n_accepted", 0) or 0 for r in rs)
    return da / dn if dn else None

fig = plt.figure(figsize=(14, 7.4), dpi=110); fig.patch.set_facecolor(BG)
gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], left=0.055, right=0.98, top=0.76, bottom=0.14, wspace=0.14)
axL, axR = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
for ax in (axL, axR):
    ax.set_facecolor(BG)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0, labelsize=11.5, colors=MUTE)
    ax.grid(color=GRID, linewidth=1, alpha=0.7); ax.set_axisbelow(True)

# ---- left
wls = ["prose", "chat", "code", "repetitive"]
modes = ["base", "mtp_n2", "mtp_n4", "ngram"]
x = np.arange(len(wls)); w = 0.2
for j, m in enumerate(modes):
    vals = [p50([r["timings"]["predicted_per_second"] for r in cell("spec", m, wl)]) for wl in wls]
    bars = axL.bar(x + (j - 1.5) * w, vals, w, color=COL[m], zorder=3, label=NAME[m])
    for b, wl, v in zip(bars, wls, vals):
        axL.text(b.get_x() + w / 2, v + 2, f"{v:.0f}", ha="center", va="bottom", color=TEXT, fontsize=10)
        a = acc(cell("spec", m, wl))
        if a and a > 0.2:
            axL.text(b.get_x() + w / 2, 8, f"{a:.2f}", ha="center", va="bottom", color=BG, fontsize=8.5, fontweight="bold", rotation=90)
axL.set_xticks(x); axL.set_xticklabels(wls, color=TEXT, fontsize=12)
axL.set_ylim(0, 205); axL.set_ylabel("decode tokens per second, p50 of 16 requests", color=MUTE, fontsize=11)
axL.xaxis.grid(False)
axL.set_title("speculative decoding: 256-token answers, greedy", color=TEXT, fontsize=13, loc="left", pad=10)
axL.legend(loc="upper left", frameon=False, fontsize=10.5, labelcolor=TEXT, ncol=2)
axL.text(0.01, 0.80, "number inside a bar = draft acceptance rate", transform=axL.transAxes, color=MUTE, fontsize=9.5, ha="left", va="top")

# ---- right: TTFT
def split(wl):
    rs = cell("cache", "base", wl)
    cold = [r["ttft_s"] for r in rs if r["timings"]["cache_n"] < 100]
    warm = [r["ttft_s"] for r in rs if r["timings"]["cache_n"] >= 100]
    stacked = [r["ttft_s"] for r in cell("cache", "mtp_n2", wl)]
    return p50(cold), p50(warm), p50(stacked)
labels = ["2,954-token\nsystem prompt", "11,570-token\nsystem prompt"]
c4, w4, s4 = split("handbook4096"); c16, w16, s16 = split("handbook16384")
xx = np.arange(2); ww = 0.26
cold = [c4, c16]; warm = [w4, w16]; stk = [s4, s16]
b1 = axR.bar(xx - ww, cold, ww, color="#e0645f", zorder=3, label="cold: prompt re-read every request")
b2 = axR.bar(xx, warm, ww, color="#57c78a", zorder=3, label="warm: prefix cache hit")
b3 = axR.bar(xx + ww, stk, ww, color="#4c9be8", zorder=3, label="warm + MTP draft (stacked)")
for bars, vals in [(b1, cold), (b2, warm), (b3, stk)]:
    for b, v in zip(bars, vals):
        axR.text(b.get_x() + ww / 2, v * 1.12, f"{v:.3f} s" if v < 1 else f"{v:.2f} s", ha="center", va="bottom", color=TEXT, fontsize=10.5)
axR.set_yscale("log"); axR.set_ylim(0.03, 12)
axR.set_yticks([0.05, 0.1, 0.2, 0.5, 1, 2, 5]); axR.set_yticks([], minor=True)
axR.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g} s"))
axR.set_xticks(xx); axR.set_xticklabels(labels, color=TEXT, fontsize=11.5)
axR.xaxis.grid(False)
axR.set_ylabel("time to first token, p50 of 8 requests (log)", color=MUTE, fontsize=11)
axR.set_title("prefix cache: same system prompt, new question", color=TEXT, fontsize=13, loc="left", pad=10)
axR.legend(loc="upper left", frameon=False, fontsize=10.5, labelcolor=TEXT)
axR.text(1 - ww, c16 * 0.5, f"{c16 / w16:.0f}x", color=CALL, fontsize=13, fontweight="bold", ha="center")
axR.text(0 - ww, c4 * 0.5, f"{c4 / w4:.0f}x", color=CALL, fontsize=13, fontweight="bold", ha="center")

fig.text(0.055, 0.93, "the two free speedups on one RTX 5090: 1.8 to 2.4x decode, 15 to 44x on time to first token",
         color=TEXT, fontsize=17.5, fontweight="bold", ha="left")
fig.text(0.055, 0.875, "Qwen3.8-27B Q6_K, llama.cpp b9653, the box's own always-on server flags, 304 requests, one variable at a time, 2026-09-13",
         color=MUTE, fontsize=12, ha="left")
fig.text(0.98, 0.965, "witcheer / llm-bench-rig", color=MUTE, fontsize=11, ha="right")
fig.text(0.055, 0.04, "sources: results/qwen3-8-27b-q6-k-speccache/records.jsonl, server timings (predicted_per_second, draft_n, draft_n_accepted, cache_n) + client TTFT. decode is unchanged by caching (62.8 vs 62.8 tok/s). MTP outputs match plain decode on 59% of answers; every difference is a near-tie flip to the model's own second choice.",
         color=MUTE, fontsize=8.8, ha="left", wrap=True)
out = "reports/spec-cache-study-qwen38.png"
fig.savefig(out, facecolor=BG); print(out)
