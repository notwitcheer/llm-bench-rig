#!/usr/bin/env python3
"""Gold & Crimson v2 chart for the spec-decode three-way (per target).

Two panels, each carrying a distinct finding (cards must ADD info, not restate):
  LEFT  — single-stream speedup by method (the ranking) + acceptance-length labels (the mechanism).
  RIGHT — aggregate tok/s vs concurrency, one line per method (the crossover: spec decode is a
          single-user latency win, and all methods converge toward / below baseline as the batch fills).

Reads results/<target>/specdecode-summary.json (from aggregate_specdecode.py).
Render on the Mac (matplotlib lives on system python3, not .venv):
  python3 scripts/chart_specdecode.py gemma-4-26b-a4b
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Gold & Crimson v2
BG, GOLD, CRIMSON, TEXT, GRID, MUTE = "#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
    "axes.edgecolor": GRID, "font.family": "DejaVu Sans", "font.size": 11,
})

METHOD_LABEL = {"mtp": "MTP", "eagle3": "EAGLE-3", "dflash": "DFlash"}
LINE_COLOR = {"baseline": MUTE, "eagle3": "#c98a3a", "mtp": GOLD, "dflash": CRIMSON}


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "gemma-4-26b-a4b"
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", target)
    summary = json.load(open(os.path.join(root, "specdecode-summary.json")))
    methods = summary["methods"]
    base_tps = summary["baseline_avg_tps"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.6))
    fig.suptitle(f"Speculative decoding on a single RTX 5090 — {target}",
                 color=TEXT, fontsize=15, fontweight="bold", x=0.5, y=0.98)

    # LEFT: single-stream speedup by method (ascending), winner crimson.
    order = [m for m in ["eagle3", "mtp", "dflash"] if m in methods and methods[m].get("speedup")]
    order.sort(key=lambda m: methods[m]["speedup"])
    ys = list(range(len(order)))
    speedups = [methods[m]["speedup"] for m in order]
    colors = [CRIMSON if m == "dflash" else (GOLD if m == "mtp" else "#c98a3a") for m in order]
    axL.barh(ys, speedups, color=colors, height=0.62, zorder=3)
    axL.axvline(1.0, color=MUTE, ls="--", lw=1.2, zorder=2)
    for y, s in zip(ys, speedups):
        axL.text(s + 0.07, y, f"{s:.2f}×", va="center", ha="left", color=TEXT, fontweight="bold", fontsize=13, zorder=4)
    axL.set_yticks(ys)
    axL.set_yticklabels([f"{METHOD_LABEL[m]}\naccept-len {methods[m]['acceptance_length']} · k={methods[m]['num_spec']}"
                         for m in order], fontsize=10)
    axL.set_xlim(0, max(speedups) * 1.28)
    axL.set_xlabel("single-stream decode speedup vs no-spec baseline")
    axL.set_title(f"DFlash ≈ MTP (~2.2x); EAGLE-3 trails  ·  baseline {base_tps:.0f} tok/s", color=TEXT, fontsize=12, pad=10)
    for s in ("top", "right", "left"):
        axL.spines[s].set_visible(False)
    axL.tick_params(axis="y", length=0)
    axL.grid(axis="x", color=GRID, lw=0.7)

    # RIGHT: per-workload speedup, line per method — DFlash's bimodality vs the steady others.
    WL = ["prose", "Q&A", "code", "JSON", "repetitive"]  # ~increasing predictability left→right
    for m in ["eagle3", "mtp", "dflash"]:
        mm = methods.get(m)
        if not mm:
            continue
        pw = mm.get("per_workload_speedup", {})
        ys2 = [pw.get(w) for w in WL]
        if not any(ys2):
            continue
        axR.plot(range(len(WL)), ys2, marker="o", color=LINE_COLOR[m], lw=2.6 if m == "dflash" else 1.9,
                 label=METHOD_LABEL[m], zorder=4 if m == "dflash" else 3)
    axR.axhline(1.0, color=MUTE, ls="--", lw=1.2, zorder=2)
    axR.set_xticks(range(len(WL))); axR.set_xticklabels(WL)
    axR.set_xlabel("workload  (less predictable → more predictable)")
    axR.set_ylabel("speedup vs baseline")
    axR.set_title("DFlash is feast-or-famine; MTP is the steady all-rounder", color=TEXT, fontsize=12, pad=10)
    axR.grid(color=GRID, lw=0.7)
    axR.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=10, loc="upper left")
    for s in ("top", "right"):
        axR.spines[s].set_visible(False)

    fig.text(0.5, 0.005, "WITCHEER", color=MUTE, fontsize=10, ha="center", fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", f"specdecode-{target}.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
