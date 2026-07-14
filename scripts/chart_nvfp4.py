#!/usr/bin/env python3
"""Gold & Crimson two-panel chart for the Qwen3.6-27B NVFP4 serving bench (runs on the Mac).

Left: prefill tok/s vs context (log2 x) — the FP4 compute win lives here, peaking
10.6k tok/s at 4k ctx. pp128 carries first-request warmup and is drawn hollow with
a note, not silently dropped. Right: decode is a different magnitude regime
(54 vs 402 tok/s), so it gets its own panel and axis, never a second y-scale.
Identity is carried by direct labels, never colour alone.

Usage: python3 scripts/chart_nvfp4.py  (writes reports/chart_nvfp4_5090.png)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, GOLD, CRIMSON = "#0d0906", "#e8c44a", "#e06060"
TEXT, GRID, MUTE = "#f5e6d0", "#3a2f25", "#8a7a64"

PREFILL = [(128, 768.33), (512, 8049.31), (2048, 10560.43),
           (4096, 10642.77), (8192, 10108.17), (16384, 9165.07)]
DECODE = [("b1\nsingle stream", 54.39), ("b8\n8 streams, aggregate", 401.78)]

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": TEXT,
    "ytick.color": TEXT, "axes.edgecolor": GRID, "font.size": 12,
})

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(12.5, 6.4), gridspec_kw={"width_ratios": [2.4, 1]})
fig.subplots_adjust(top=0.80, bottom=0.14, left=0.09, right=0.97, wspace=0.28)

# ---- left: prefill curve ----
xs = [p[0] for p in PREFILL]
ys = [p[1] for p in PREFILL]
ax1.plot(xs[1:], ys[1:], color=CRIMSON, lw=2, marker="o", markersize=8, zorder=3)
ax1.plot(xs[:2], ys[:2], color=CRIMSON, lw=2, ls=":", zorder=2)
ax1.plot([xs[0]], [ys[0]], marker="o", markersize=8, mfc=BG, mec=CRIMSON,
         mew=2, zorder=3)
ax1.set_xscale("log", base=2)
ax1.set_xticks(xs)
ax1.set_xticklabels(["128*", "512", "2k", "4k", "8k", "16k"])
ax1.set_ylim(0, 12200)
ax1.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
ax1.tick_params(length=0)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)

ax1.annotate("peak 10,643 tok/s @ 4k", (4096, 10642.77), textcoords="offset points",
             xytext=(0, 14), ha="center", fontsize=11.5, color=TEXT, zorder=4)
ax1.annotate("9,165 @ 16k", (16384, 9165.07), textcoords="offset points",
             xytext=(-6, -20), ha="center", fontsize=10.5, color=TEXT)
ax1.text(150, 1650, "768\n*first-request\nwarmup", ha="left", fontsize=9.5,
         color=MUTE)
ax1.set_title("prefill tok/s vs context: compute-bound, FP4 GEMM pays",
              fontsize=12.5, color=TEXT, loc="left", pad=10)
ax1.set_xlabel("prompt tokens (log2)", color=MUTE)

# ---- right: decode panel ----
labels = [d[0] for d in DECODE]
vals = [d[1] for d in DECODE]
bars = ax2.bar(labels, vals, width=0.52, color=CRIMSON, zorder=3)
for rect, v in zip(bars, vals):
    ax2.text(rect.get_x() + rect.get_width() / 2, v + 9, f"{v:.0f}",
             ha="center", fontsize=12.5, color=TEXT, zorder=4)
ax2.set_ylim(0, 460)
ax2.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
ax2.tick_params(length=0)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
ax2.text(-0.42, 432, "+4% latency for 8 streams", ha="left", fontsize=10.5,
         color=GOLD, transform=ax2.transData)
ax2.set_title("decode tok/s: memory-bound", fontsize=12.5, color=TEXT,
              loc="left", pad=10)

fig.suptitle("unsloth Qwen3.6-27B NVFP4 on one RTX 5090: the FP4 win lives in "
             "prefill and batch, not single-stream chat",
             fontsize=14.5, color=TEXT, x=0.09, ha="left")
fig.text(0.09, 0.895, "vLLM 0.21.0 · flashinfer cutlass sm_120 FP4 (native, no marlin fallback) · "
         "32k ctx · chunked prefill 2048 · 28.9GB VRAM peak · greedy",
         fontsize=10.5, color=MUTE)

fig.savefig("reports/chart_nvfp4_5090.png", dpi=170)
print("wrote reports/chart_nvfp4_5090.png")
