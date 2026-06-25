"""FP4 on a consumer 5090: does it actually go fast? (Gold & Crimson v2)
Left  — Tier 1: vLLM batch-1 decode tok/s on Qwen3-14B per quant. AWQ int4 (gold) wins; NVFP4
        (crimson) runs the NATIVE cutlass FP4 kernel on sm_120 (JIT, after a toolchain fight) and is
        STILL slower than AWQ; FP8 slowest; bf16 OOMs.
Right — Tier 2: QuTLASS MXFP4 (the academic real-FP4 path) vs bf16 on Qwen3-8B. The MXFP4 GEMM
        crosses the claimed 4x and peaks ~6x as batch grows (gold) — but end-to-end DECODE never
        reaches parity (crimson, ~0.3x). FP4's win is real at the GEMM, lost at the token."""
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


fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.4, 6.2), facecolor=BG)
fig.suptitle("FP4 on a consumer RTX 5090: loses to int4 at decode — and its real 4x lives in the GEMM, not the token",
             color=GOLD, fontsize=13.5, fontweight="bold", y=0.975)

# ---------- LEFT: Tier 1 vLLM batch-1 bars ----------
axL.set_facecolor(BG)
order = [l for l in ["awq", "fp8", "nvfp4"] if l in vllm]
awq1 = tps_at(vllm["awq"], 1) if "awq" in vllm else None
xs = range(len(order))
vals = [tps_at(vllm[l], 1) or 0 for l in order]
colors = [GOLD if l == "awq" else (CRIMSON if l == "nvfp4" else MUTE) for l in order]
axL.bar(xs, vals, 0.6, color=colors, edgecolor=TEXT, zorder=3)
for i, l in enumerate(order):
    axL.annotate(f"{vals[i]:.0f}", (i, vals[i]), color=TEXT, ha="center", va="bottom",
                 fontsize=12, fontweight="bold", xytext=(0, 4), textcoords="offset points")
    if l != "awq" and awq1:
        axL.annotate(f"{speedup(vals[i], awq1):.2f}x", (i, vals[i] * 0.5), color=BG, ha="center",
                     va="center", fontsize=11, fontweight="bold")
axL.set_xticks(list(xs))
axL.set_xticklabels([f"{l}\n{KERNEL.get(str(vllm[l]['declared_quant']), '')}" for l in order],
                    color=TEXT, fontsize=10)
axL.set_ylabel("batch-1 decode tok/s  (vLLM, Qwen3-14B)", color=TEXT, fontsize=11)
axL.set_ylim(0, (max(vals) if vals else 1) * 1.18)
axL.set_title("Tier 1 — vLLM/NVFP4: native FP4 kernels, still < AWQ int4", color=MUTE, fontsize=9.5, pad=8)
for s in axL.spines.values():
    s.set_color(GRID)
axL.tick_params(colors=MUTE)
axL.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)

# ---------- RIGHT: Tier 2 QuTLASS MXFP4 speedup divergence ----------
axR.set_facecolor(BG)
ker = json.load(open("results/fp4/qutlass_kernel_gemm.json"))
batch = ker["batch"]
g = ker["shapes"]["gate+up (K=4096,N=24576)"]
gemm = [m / b for m, b in zip(g["mxfp4"], g["torch-bf16"])]
gemm_nq = [m / b for m, b in zip(g["mxfp4-noquant"], g["torch-bf16"])]

mx = json.load(open("results/fp4/qutlass__mxfp4.json"))
bf = json.load(open("results/fp4/qutlass__bf16.json"))
dec_b = [r["batch"] for r in mx["batches"] if r.get("tps")]
dec_s = [tps_at(mx, b) / tps_at(bf, b) for b in dec_b]

axR.axhline(4.0, color=MUTE, ls=":", lw=1.2, zorder=1)
axR.axhline(1.0, color=TEXT, ls=":", lw=1.0, zorder=1)
axR.annotate("claimed 4x", (batch[-1], 4.0), color=MUTE, fontsize=8.5, va="bottom", ha="right")
axR.annotate("bf16 parity (1x)", (batch[0], 1.0), color=TEXT, fontsize=8.5, va="bottom", ha="left")

axR.plot(batch, gemm_nq, color=GOLD, lw=2.0, ls="--", marker="o", ms=3.5, zorder=4,
         label="MXFP4 GEMM, pre-quant act (best case)")
axR.plot(batch, gemm, color=GOLD, lw=2.4, marker="o", ms=4, zorder=5,
         label="MXFP4 GEMM, on-the-fly act quant")
axR.plot(dec_b, dec_s, color=CRIMSON, lw=2.6, marker="s", ms=6, zorder=6,
         label="MXFP4 end-to-end decode (model)")
axR.annotate(f"{gemm[-1]:.1f}x", (batch[-1], gemm[-1]), color=GOLD, fontsize=10, fontweight="bold",
             va="center", ha="left", xytext=(4, 0), textcoords="offset points")
axR.annotate(f"~{dec_s[-1]:.2f}x  (slower)", (dec_b[-1], dec_s[-1]), color=CRIMSON, fontsize=9.5,
             fontweight="bold", va="top", ha="center", xytext=(0, -6), textcoords="offset points")

axR.set_xscale("log", base=2)
axR.set_xticks(batch)
axR.set_xticklabels([str(b) for b in batch], color=MUTE, fontsize=8, rotation=0)
axR.set_xlabel("batch size  (Qwen3-8B)", color=TEXT, fontsize=11)
axR.set_ylabel("speedup vs bf16  (x)", color=TEXT, fontsize=11)
axR.set_ylim(0, max(gemm_nq) * 1.12)
axR.set_title("Tier 2 — QuTLASS MXFP4: 4x is real at the GEMM, the token never reaches parity",
              color=MUTE, fontsize=9.5, pad=8)
for s in axR.spines.values():
    s.set_color(GRID)
axR.tick_params(colors=MUTE)
axR.grid(True, color=GRID, linewidth=0.5, zorder=0)
axR.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=8.3, loc="upper left")

fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("reports/fp4-consumer-blackwell.png", dpi=150, facecolor=BG)
print("wrote reports/fp4-consumer-blackwell.png")
