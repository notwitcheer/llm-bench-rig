"""EV+ card for the QuTLASS MXFP4 post (Gold & Crimson v2). Standalone single panel.
Shows the full speedup-vs-batch curve the post only summarises: the MXFP4 GEMM crosses the
claimed 4x and peaks ~6x over bf16, while end-to-end model decode never reaches parity (~0.3x).
The data the words don't carry — the shape of the divergence across batch size."""
import json

import matplotlib.pyplot as plt

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = ("#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64")


def tps_at(d, b):
    return next((r["tps"] for r in d["batches"] if r["batch"] == b), None)


ker = json.load(open("results/fp4/qutlass_kernel_gemm.json"))
batch = ker["batch"]
g = ker["shapes"]["gate+up (K=4096,N=24576)"]
gemm = [m / b for m, b in zip(g["mxfp4"], g["torch-bf16"])]
gemm_nq = [m / b for m, b in zip(g["mxfp4-noquant"], g["torch-bf16"])]

mx = json.load(open("results/fp4/qutlass__mxfp4.json"))
bf = json.load(open("results/fp4/qutlass__bf16.json"))
dec_b = [r["batch"] for r in mx["batches"] if r.get("tps")]
dec_s = [tps_at(mx, b) / tps_at(bf, b) for b in dec_b]

fig, ax = plt.subplots(figsize=(8.6, 5.6), facecolor=BG)
ax.set_facecolor(BG)
fig.suptitle("FP4 on a consumer RTX 5090: real at the GEMM, lost at the token",
             color=GOLD, fontsize=15, fontweight="bold", y=0.97)
ax.set_title("QuTLASS MXFP4 vs bf16, Qwen3-8B  ·  built from source on sm_120a",
             color=MUTE, fontsize=10, pad=8)

ax.axhline(4.0, color=MUTE, ls=":", lw=1.2, zorder=1)
ax.axhline(1.0, color=TEXT, ls=":", lw=1.0, zorder=1)
ax.annotate("claimed 4x", (batch[-1], 4.0), color=MUTE, fontsize=9, va="bottom", ha="right")
ax.annotate("bf16 parity (1x)", (batch[0], 1.0), color=TEXT, fontsize=9, va="bottom", ha="left")

ax.plot(batch, gemm_nq, color=GOLD, lw=2.0, ls="--", marker="o", ms=3.5, zorder=4,
        label="MXFP4 GEMM — activation pre-quantised (best case)")
ax.plot(batch, gemm, color=GOLD, lw=2.6, marker="o", ms=4.5, zorder=5,
        label="MXFP4 GEMM — activation quantised on the fly")
ax.plot(dec_b, dec_s, color=CRIMSON, lw=2.8, marker="s", ms=7, zorder=6,
        label="MXFP4 end-to-end decode (the actual model)")

ax.annotate(f"{gemm[-1]:.1f}x", (batch[-1], gemm[-1]), color=GOLD, fontsize=12, fontweight="bold",
            va="center", ha="left", xytext=(5, 0), textcoords="offset points")
ax.annotate("3-4x SLOWER than bf16", (dec_b[-1], dec_s[-1]), color=CRIMSON, fontsize=10,
            fontweight="bold", va="bottom", ha="center", xytext=(18, 6), textcoords="offset points")

ax.set_xscale("log", base=2)
ax.set_xticks(batch)
ax.set_xticklabels([str(b) for b in batch], color=MUTE, fontsize=8.5)
ax.set_xlabel("batch size", color=TEXT, fontsize=11.5)
ax.set_ylabel("speedup vs bf16  (x)", color=TEXT, fontsize=11.5)
ax.set_ylim(0, max(gemm_nq) * 1.12)
for s in ax.spines.values():
    s.set_color(GRID)
ax.tick_params(colors=MUTE)
ax.grid(True, color=GRID, linewidth=0.5, zorder=0)
ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=9, loc="upper left")

fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("reports/fp4-qutlass-card.png", dpi=140, facecolor=BG)
print("wrote reports/fp4-qutlass-card.png")
