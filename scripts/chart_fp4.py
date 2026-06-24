"""FP4 on a consumer 5090: does it actually go fast? (Gold & Crimson v2)
vLLM batch-1 decode tok/s on Qwen3-14B per quant. AWQ int4 (gold) wins; NVFP4 (crimson) runs the
NATIVE cutlass FP4 kernel on sm_120 (JIT-compiled, after a toolchain fight) and is STILL slower than
AWQ; FP8 is slowest (biggest weights). bf16 doesn't fit 32GB. The FP4 headline is a B200 story."""
import glob
import json

import matplotlib.pyplot as plt

from lib.fp4 import speedup

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = ("#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64")
KERNEL = {"awq_marlin": "awq-marlin (int4)", "fp8": "fp8-marlin", "modelopt_fp4": "nvfp4 cutlass (JIT)"}

vllm = {}
for f in glob.glob("results/fp4/vllm__*.json"):
    d = json.load(open(f))
    vllm[d["label"]] = d


def tps_at(d, batch):
    return next((r["tps"] for r in d["batches"] if r["batch"] == batch), None)


order = [l for l in ["awq", "fp8", "nvfp4"] if l in vllm]
awq1 = tps_at(vllm["awq"], 1) if "awq" in vllm else None

fig, ax = plt.subplots(figsize=(8.6, 6.3), facecolor=BG)
ax.set_facecolor(BG)
fig.suptitle("FP4 on a consumer RTX 5090: the Blackwell headline loses to plain int4",
             color=GOLD, fontsize=14, fontweight="bold", y=0.97)

xs = range(len(order))
vals = [tps_at(vllm[l], 1) or 0 for l in order]
colors = [GOLD if l == "awq" else (CRIMSON if l == "nvfp4" else MUTE) for l in order]
ax.bar(xs, vals, 0.6, color=colors, edgecolor=TEXT, zorder=3)
for i, l in enumerate(order):
    ax.annotate(f"{vals[i]:.0f}", (i, vals[i]), color=TEXT, ha="center", va="bottom",
                fontsize=12, fontweight="bold", xytext=(0, 4), textcoords="offset points")
    if l != "awq" and awq1:
        ax.annotate(f"{speedup(vals[i], awq1):.2f}x", (i, vals[i] * 0.5), color=BG, ha="center",
                    va="center", fontsize=11, fontweight="bold")
ax.set_xticks(list(xs))
ax.set_xticklabels([f"{l}\n{KERNEL.get(str(vllm[l]['declared_quant']), '')}" for l in order],
                   color=TEXT, fontsize=10.5)
ax.set_ylabel("batch-1 decode tok/s  (vLLM, Qwen3-14B)", color=TEXT, fontsize=11)
ax.set_ylim(0, max(vals) * 1.18)
ax.set_title("AWQ runs out of the box; NVFP4 needs nvcc+ninja+CUDA_HOME to JIT its kernels, then "
             "still loses (bf16 OOMs on 32GB)", color=MUTE, fontsize=9, pad=8)
for s in ax.spines.values():
    s.set_color(GRID)
ax.tick_params(colors=MUTE)
ax.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)

fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("reports/fp4-consumer-blackwell.png", dpi=150, facecolor=BG)
print("wrote reports/fp4-consumer-blackwell.png")
