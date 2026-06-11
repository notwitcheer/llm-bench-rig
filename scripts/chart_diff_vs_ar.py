"""Gold-&-Crimson AR-vs-diffusion throughput chart from results/diffusion-ar/results.json.
Grouped bars per prompt: AR decode tok/s (GOLD, the winner) vs block-diffusion *effective*
answer tok/s (CRIMSON, the subject), annotated with the AR/diffusion speedup. Same model
family, same Q4_K_M, same RTX 5090 — the only variable is the generation paradigm."""
import json
import sys
from pathlib import Path

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = ("#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64")


def main(path="results/diffusion-ar/results.json"):
    import matplotlib.pyplot as plt
    import numpy as np
    d = json.loads(Path(path).read_text())
    items = d["items"]
    labels = [f"{it['tag']}\n(ans {it['diff']['answer_tokens']}t)" for it in items]
    ar = [it["ar"]["tok_s"] for it in items]
    df = [it["diff"]["eff_answer_tok_s"] for it in items]
    x = np.arange(len(items))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11.5, 6.5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.bar(x - w / 2, ar, w, color=GOLD, edgecolor=TEXT, linewidth=0.8, zorder=3,
           label="autoregressive — gemma-4-26B-A4B-it")
    ax.bar(x + w / 2, df, w, color=CRIMSON, edgecolor=TEXT, linewidth=0.8, zorder=3,
           label="block diffusion — diffusiongemma-26B-A4B-it")
    for i in range(len(items)):
        ratio = ar[i] / df[i] if df[i] else 0
        tag = f"{ratio:.0f}x" if ratio >= 10 else f"{ratio:.1f}x"
        ax.text(x[i], max(ar[i], df[i]) + max(ar) * 0.02, f"AR {tag} faster",
                ha="center", color=TEXT, fontsize=8.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=MUTE, fontsize=8.5)
    ax.set_ylabel("effective answer tok/s  (higher = faster)", color=TEXT)
    ax.set_title("Same model, one RTX 5090, Q4_K_M: autoregressive beats block-diffusion at every answer length",
                 color=GOLD, pad=16, fontsize=12)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=9, loc="upper right")
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=MUTE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_ylim(0, max(ar) * 1.2)
    fig.tight_layout()
    fig.savefig("reports/diffusion-vs-ar.png", dpi=150, facecolor=BG)
    print("wrote reports/diffusion-vs-ar.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/diffusion-ar/results.json")
