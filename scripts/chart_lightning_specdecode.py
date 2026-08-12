#!/usr/bin/env python3
"""Nemotron 3.5 Lightning spec-decode leg: every drafter config slows the model down.

Grouped bars of measured speedup ratio vs plain decoding (chat-server conditions, four
workloads, 8 prompts x 256 tok each, temperature 0) for NVIDIA's two shipped drafters on
one RTX 5090 via llama.cpp b10371 (dflash support merged 2026-08-11, PR #26905).

The 1.0 line is break-even. Every bar sits under it -- the chart IS the finding.
Data: capsule ~/nl-spec-results.json (base 328.6-335.4 tok/s; ratios below).
House palette shared with chart_quadrant.py / chart_ranked_field.py.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG   = "#0e1420"; TEXT = "#eaf0f7"; MUTE = "#7d8ba0"; GRID = "#20293a"
BREAK = "#1f9e8a"   # break-even line, teal

TITLE    = "Two drafters ship with the fastest model. Both slow it down."
SUBTITLE = "Nemotron 3.5 Lightning Q4_K_M on one RTX 5090 · measured speedup vs plain decoding, llama.cpp, 4 workloads · thinking off"
OUT      = "/opt/data/mercury-cards/nemotron-lightning/spec-decode-negative.png"

WORKLOADS = ["prose", "code", "repetitive", "chat"]
BASE = {"prose": 328.61, "code": 335.44, "repetitive": 335.12, "chat": 335.32}
LEGS = [  # (label, colour, {workload: tok/s})
    ("MTP  n=2",  "#e0645f", {"prose": 239.00, "code": 256.04, "repetitive": 260.11, "chat": 248.69}),
    ("MTP  n=4",  "#c94f4a", {"prose": 230.55, "code": 252.93, "repetitive": 247.82, "chat": 235.08}),
    ("MTP  n=8",  "#a83c38", {"prose": 186.89, "code": 201.91, "repetitive": 211.89, "chat": 195.42}),
    ("DFlash",    "#e8b64c", {"prose": 175.45, "code": 206.33, "repetitive": 203.30, "chat": 186.45}),
]

def render():
    fig, ax = plt.subplots(figsize=(12, 6.8), dpi=110)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    fig.subplots_adjust(left=0.075, right=0.965, top=0.78, bottom=0.13)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0, labelsize=13, colors=TEXT)
    ax.grid(color=GRID, linewidth=1, alpha=0.6, axis="y"); ax.set_axisbelow(True)

    n_legs = len(LEGS)
    width = 0.19
    x = np.arange(len(WORKLOADS))
    for i, (label, colour, vals) in enumerate(LEGS):
        ratios = [vals[w] / BASE[w] for w in WORKLOADS]
        pos = x + (i - (n_legs - 1) / 2) * width
        bars = ax.bar(pos, ratios, width * 0.92, color=colour, zorder=3,
                      edgecolor=BG, linewidth=1.2, label=label)
        for p, r in zip(pos, ratios):
            ax.text(p, r - 0.018, f"{r:.2f}x", ha="center", va="top",
                    color=BG, fontsize=10.5, fontweight="bold", zorder=4, rotation=90)

    ax.axhline(1.0, color=BREAK, linewidth=2, zorder=2)
    ax.text(len(WORKLOADS) - 0.52, 1.012, "break-even (plain decoding, 329-335 tok/s)",
            color=BREAK, fontsize=11.5, fontweight="bold", ha="right", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(WORKLOADS, fontsize=13.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("speedup vs plain decoding", fontsize=13, color=TEXT)

    leg = ax.legend(loc="upper right", frameon=False, fontsize=12, ncols=4,
                    bbox_to_anchor=(1.0, 1.09))
    for t in leg.get_texts(): t.set_color(MUTE)

    fig.text(0.075, 0.925, TITLE, fontsize=21, fontweight="bold", color=TEXT)
    fig.text(0.075, 0.85, SUBTITLE, fontsize=12.5, color=MUTE)
    fig.text(0.965, 0.035, "WITCHEER  ·  RTX 5090 benchmarks", fontsize=11.5, color=MUTE, ha="right")

    fig.savefig(OUT, facecolor=BG)
    print("wrote", OUT)

if __name__ == "__main__":
    render()
