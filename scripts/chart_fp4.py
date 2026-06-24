"""FP4 on a consumer 5090: does it actually go fast? (Gold & Crimson v2)
Left: vLLM batch-1 decode tok/s per quant on Qwen3-14B, with the ACTUAL kernel under each bar —
NVFP4 is crimson to flag that it silently runs the Marlin dequant (no FP4 win vs AWQ/FP8).
Right (if QuTLASS ran): MXFP4-vs-bf16 speedup across the batch sweep — the one path with real FP4."""
import glob
import json
import os

import matplotlib.pyplot as plt

from lib.fp4 import parse_quant_kernel, speedup

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = ("#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64")

vllm = {}
for f in glob.glob("results/fp4/vllm__*.json"):
    d = json.load(open(f))
    vllm[d["label"]] = d


def tps_at(d, batch):
    return next((r["tps"] for r in d["batches"] if r["batch"] == batch), None)


q = {}
for f in glob.glob("results/fp4/qutlass__*.json"):
    d = json.load(open(f))
    q[d["arm"]] = d
has_q = "bf16" in q and "mxfp4" in q

fig, axes = plt.subplots(1, 2 if has_q else 1, figsize=(12.5 if has_q else 7.2, 6.2),
                         facecolor=BG, squeeze=False)
fig.suptitle("FP4 on a consumer RTX 5090: does it actually go fast?",
             color=GOLD, fontsize=14.5, fontweight="bold", y=0.97)

# --- left: vLLM batch-1 tok/s per quant + kernel labels ---
ax = axes[0][0]
ax.set_facecolor(BG)
order = [l for l in ["bf16", "awq", "fp8", "nvfp4"] if l in vllm]
xs = range(len(order))
vals, kernels, colors = [], [], []
for lab in order:
    vals.append(tps_at(vllm[lab], 1) or 0)
    lp = f"results/fp4/{lab}.initlog"
    kernels.append(parse_quant_kernel(open(lp).read()) if os.path.exists(lp) else "?")
    colors.append(CRIMSON if lab == "nvfp4" else (GOLD if lab in ("awq", "fp8") else MUTE))
ax.bar(xs, vals, 0.62, color=colors, edgecolor=TEXT, zorder=3)
for i, lab in enumerate(order):
    ax.annotate(f"{vals[i]:.0f}", (i, vals[i]), color=TEXT, ha="center", va="bottom",
                fontsize=11, fontweight="bold", xytext=(0, 4), textcoords="offset points")
    ax.annotate(kernels[i], (i, vals[i] * 0.5), color=BG, ha="center", va="center",
                fontsize=9, fontweight="bold", rotation=90)
ax.set_xticks(list(xs))
ax.set_xticklabels(order, color=TEXT, fontsize=11)
ax.set_ylabel("batch-1 decode tok/s (vLLM, Qwen3-14B)", color=TEXT, fontsize=10.5)
ax.set_title("NVFP4 'loads' but routes to Marlin dequant — no FP4 win", color=MUTE, fontsize=9.5, pad=8)

# --- right: QuTLASS MXFP4 speedup vs batch ---
if has_q:
    ax2 = axes[0][1]
    ax2.set_facecolor(BG)
    qb = {r["batch"]: r["tps"] for r in q["bf16"]["batches"]}
    pts = [(r["batch"], speedup(r["tps"], qb[r["batch"]]))
           for r in q["mxfp4"]["batches"] if r.get("tps") and qb.get(r["batch"])]
    if pts:
        bxs, sys_ = zip(*pts)
        ax2.plot(range(len(bxs)), sys_, "-o", color=CRIMSON, linewidth=2.6, markersize=9,
                 markeredgecolor=TEXT, zorder=3)
        for i, (b, s) in enumerate(pts):
            ax2.annotate(f"{s:.2f}x", (i, s), color=CRIMSON, ha="center", va="bottom",
                         fontsize=10.5, fontweight="bold", xytext=(0, 6), textcoords="offset points")
        ax2.axhline(1.0, color=GRID, linewidth=1.0, linestyle="--", zorder=1)
        ax2.set_xticks(range(len(bxs)))
        ax2.set_xticklabels([f"batch {b}" for b in bxs], color=TEXT, fontsize=10.5)
        ax2.set_ylabel("MXFP4 speedup vs bf16 (Transformers)", color=TEXT, fontsize=10.5)
        ax2.set_title("QuTLASS MXFP4: real FP4 kernels (claim: 4x at large batch)",
                      color=MUTE, fontsize=9.5, pad=8)
        for s in ax2.spines.values():
            s.set_color(GRID)
        ax2.tick_params(colors=MUTE)
        ax2.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)

for ax_ in [axes[0][0]] + ([axes[0][1]] if has_q else []):
    for s in ax_.spines.values():
        s.set_color(GRID)
    ax_.tick_params(colors=MUTE)
    ax_.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)

fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("reports/fp4-consumer-blackwell.png", dpi=150, facecolor=BG)
print("wrote reports/fp4-consumer-blackwell.png")
