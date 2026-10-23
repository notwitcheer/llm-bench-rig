#!/usr/bin/env python3
"""Humming vs Marlin vs stock default on one RTX 5090: three-panel horizontal bars, numbers pulled from the raw jsons."""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

D = sys.argv[1]; OUT = sys.argv[2]
BG = "#0f1420"; TEXT = "#e8e6e1"; MUTE = "#9aa3b2"
C_DEF = "#7d8aa3"; C_MAR = "#4fa3d9"; C_HUM = "#f2b134"


def val(legs, key):
    vals = []
    for l in legs:
        if key == "pf":
            p = os.path.join(D, f"p_{l.rstrip('2').replace('moe_auto','moe_marlin')}.prefill.json")
            if os.path.exists(p):
                vals = [json.load(open(p))["summary"]["ttft_p50_s"] * 1000]
            continue
        p = os.path.join(D, l + ".json")
        if os.path.exists(p):
            s = json.load(open(p))["summary"]
            vals.append(s["c1"]["perceived_tps_p50"] if key == "c1" else s["c8"]["total_tps_median"])
    vals.sort(); n = len(vals)
    return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2

ROWS = [
    ("Qwen3.6-27B AWQ int4, dense", "Marlin (default)", ["awq_marlin", "awq_marlin2"], C_MAR, True),
    ("Qwen3.6-27B AWQ int4, dense", "Humming", ["awq_humming", "awq_humming2"], C_HUM, False),
    ("Qwen3.8-27B NVFP4, dense", "CUTLASS FP4 (default)", ["nv_cutlass", "nv_cutlass2"], C_DEF, True),
    ("Qwen3.8-27B NVFP4, dense", "Marlin", ["nv_marlin"], C_MAR, False),
    ("Qwen3.8-27B NVFP4, dense", "Humming", ["nv_humming", "nv_humming2"], C_HUM, False),
    ("Ornith 1.5 35B-A3B NVFP4, MoE", "Marlin (default)", ["moe_marlin", "moe_auto"], C_MAR, True),
    ("Ornith 1.5 35B-A3B NVFP4, MoE", "Humming", ["moe_humming", "moe_humming2"], C_HUM, False),
]
fig, axes = plt.subplots(1, 3, figsize=(15, 7.6), facecolor=BG, gridspec_kw={"width_ratios": [1.15, 1.15, 1]})
fig.subplots_adjust(left=0.165, right=0.975, top=0.79, bottom=0.11, wspace=0.16)
ypos = []; y = 0; prev = None; group_y = {}
for g, *_ in ROWS:
    if prev is not None and g != prev:
        y += 0.85
    ypos.append(y); group_y.setdefault(g, []).append(y); y += 1; prev = g
PANELS = [("c1", "one request: tok/s per user", "{:.1f}", 400),
          ("c8", "8 requests: total tok/s", "{:,.0f}", 1650),
          ("pf", "2k-token prompt: first token, ms\n(lower is better)", "{:.0f}", 700)]
for ax, (key, title, fmt, xmax) in zip(axes, PANELS):
    ax.set_facecolor(BG)
    for (g, lab, legs, col, dflt), yy in zip(ROWS, ypos):
        v = val(legs, key)
        ax.barh(yy, v, color=col, height=0.78, edgecolor=TEXT if dflt else col, linewidth=1.6 if dflt else 0)
        ax.text(v + xmax * 0.015, yy, fmt.format(v), va="center", ha="left", color=TEXT, fontsize=12, fontweight="bold")
    ax.set_xlim(0, xmax); ax.set_xticks([])
    ax.invert_yaxis()
    ax.set_yticks(ypos if ax is axes[0] else [])
    if ax is axes[0]:
        ax.set_yticklabels([r[1] for r in ROWS], color=TEXT, fontsize=12)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(title, color=TEXT, fontsize=13, loc="left", pad=10, va="bottom")
for g, ys in group_y.items():
    axes[0].text(-0.62, ys[0] - 0.58, g, transform=axes[0].get_yaxis_transform(), color=MUTE, fontsize=11.5,
                 fontweight="bold", ha="left", va="bottom")
fig.text(0.015, 0.945, "Humming vs Marlin on one RTX 5090", color=TEXT, fontsize=21, fontweight="bold", ha="left")
fig.text(0.015, 0.898, "vLLM 0.32.0 + humming-kernels 0.1.17. same checkpoint per group, only the kernel backend flag changes.",
         color=MUTE, fontsize=12.5, ha="left")
fig.text(0.015, 0.866, "chat, thinking off, temperature 0, 512 output tokens. bars = median of server starts.", color=MUTE, fontsize=12.5, ha="left")
fig.text(0.015, 0.035, "outlined bar = what stock vLLM picks on this card", color=TEXT, fontsize=13, fontweight="bold", ha="left")
fig.text(0.985, 0.035, "WITCHEER · RTX 5090", color=MUTE, fontsize=10.5, ha="right")
fig.patches.append(Rectangle((0.0, 0.975), 0.03, 0.012, transform=fig.transFigure, color="#c8323c"))
fig.savefig(OUT, dpi=150, facecolor=BG)
print("wrote", OUT)
