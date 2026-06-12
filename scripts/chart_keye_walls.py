"""t045 Keye-VL-2.0 chart (Gold & Crimson v2), two panels:
left = memory vs context: the reference-DSA indexer score matrix is O(N^2)
(16 heads x N^2 x bf16) vs the linear GQA KV cache (96 KiB/token), against the
RTX 5090's 31.4GB — with the measured 30.65GiB OOM at ~32K;
right = the quant wall: bnb-4bit reaches 1.46B of 31.12B params (4.7%) because
the fused experts are 3D Parameters, invisible to Linear-swap quantizers."""
import matplotlib.pyplot as plt
import numpy as np

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = ("#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), facecolor=BG,
                               gridspec_kw={"width_ratios": [1.5, 1]})
fig.suptitle('Keye-VL-2.0-30B "lossless 256K" vs one RTX 5090 — measured',
             color=GOLD, fontsize=15, fontweight="bold", y=0.98)

# left: memory vs context (log-log)
ctx = np.array([4096, 8192, 16384, 32768, 65536, 131072, 262144])
indexer_gb = 16 * ctx.astype(float) ** 2 * 2 / 1e9      # score matrix, bf16
kv_gb = 96 * 1024 * ctx.astype(float) / 1e9             # 96 KiB/token GQA KV
ax1.set_facecolor(BG)
ax1.plot(ctx, indexer_gb, color=CRIMSON, linewidth=2.5, marker="o",
         label="DSA indexer scores (O(N²), shipped impl)")
ax1.plot(ctx, kv_gb, color=GOLD, linewidth=2.5, marker="s",
         label="KV cache (linear, GQA-4)")
ax1.axhline(31.4, color=MUTE, linestyle="--", linewidth=1.5)
ax1.annotate("RTX 5090 (31.4GB total)", (4300, 36), color=MUTE, fontsize=10)
ax1.scatter([32768], [32.9], s=160, color=CRIMSON, zorder=5, marker="X")
ax1.annotate("measured OOM:\n30.65GiB tensor at ~32K\n(a 60s video prompt)",
             (36000, 14), color=CRIMSON, fontsize=10, fontweight="bold")
ax1.annotate("256K needs 2.1TB\nfor scores alone", (115000, 700), color=TEXT, fontsize=10)
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xticks(ctx)
ax1.set_xticklabels(["4K", "8K", "16K", "32K", "64K", "128K", "256K"], color=TEXT)
ax1.set_xlabel("context length (tokens)", color=TEXT)
ax1.set_ylabel("memory (GB, log scale)", color=TEXT)
ax1.set_title("the sparse attention is O(N²)-memory outside Hopper kernels",
              color=TEXT, fontsize=11.5, pad=10)
ax1.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=9.5, loc="upper left")

# right: quant reach
ax2.set_facecolor(BG)
ax2.barh([0], [1.46], color=GOLD, edgecolor=TEXT, zorder=3, height=0.5,
         label="quantizable (nn.Linear)")
ax2.barh([0], [31.12 - 1.46], left=[1.46], color=CRIMSON, edgecolor=TEXT,
         zorder=3, height=0.5, label="unreachable (fused 3D experts)")
ax2.annotate("4.7%", (0.6, 0.32), color=GOLD, fontsize=14, fontweight="bold")
ax2.annotate("95.3% of weights:\nno consumer quant path\n(no GGUF · no vLLM arch ·\nbnb/torchao see nn.Linear only)",
             (14, -0.02), color=TEXT, fontsize=10.5, ha="center", va="center")
ax2.set_xlim(0, 31.12)
ax2.set_ylim(-0.6, 0.6)
ax2.set_yticks([])
ax2.set_xlabel("parameters (B)", color=TEXT)
ax2.set_title("what 4-bit quantization can reach: 1.46B of 31.12B",
              color=TEXT, fontsize=11.5, pad=10)
ax2.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=9.5,
           loc="lower right")

for ax in (ax1, ax2):
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTE)
    ax.grid(True, color=GRID, linewidth=0.5, zorder=0)

fig.text(0.5, 0.015,
         "weights 62.3GB bf16 · only transformers 5.0.0rc0/rc1 satisfies the code's API window · "
         "13 run attempts, 6 shims: output still incoherent — official path is Hopper-only docker",
         color=MUTE, ha="center", fontsize=9.5)
fig.tight_layout(rect=[0, 0.05, 1, 0.93])
fig.savefig("reports/keye-walls-chart.png", dpi=150, facecolor=BG)
print("wrote reports/keye-walls-chart.png")
