"""EV+ scatter: nDCG@10 (quality) vs doc encode docs/s (cost). Gold & Crimson v2.
Subject (qwen3-vl) CRIMSON, foil (qwen3-text) GOLD, baseline (e5-small) MUTE.
"""
import glob
import json
import matplotlib.pyplot as plt

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = (
    "#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64")
COLOR = {"qwen3-vl": CRIMSON, "qwen3-text": GOLD, "e5-small": MUTE}
LABEL = {"qwen3-vl": "Qwen3-VL-Embedding-2B", "qwen3-text": "Qwen3-Embedding-0.6B", "e5-small": "e5-small-v2"}


def load(dataset="scifact"):
    rows = []
    for p in sorted(glob.glob(f"results/embed-{dataset}/*.json")):
        if p.endswith("-matryoshka.json"):
            continue
        with open(p) as f:
            rows.append(json.load(f))
    return rows


def main():
    rows = load()
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG)
    ax.set_facecolor(BG)
    for r in rows:
        arm = r["arm"]
        ax.scatter(r["doc_encode_dps"], r["scores"]["ndcg@10"], s=260,
                   color=COLOR.get(arm, MUTE), edgecolor=TEXT, linewidth=1.2, zorder=3)
        ax.annotate(f"{LABEL.get(arm, arm)}\ndim {r['dim']} · {r['peak_vram_gb']}GB",
                    (r["doc_encode_dps"], r["scores"]["ndcg@10"]),
                    color=TEXT, fontsize=9, xytext=(8, 8), textcoords="offset points")
    ax.set_xlabel("doc encode throughput (docs/s)  →  cheaper", color=TEXT)
    ax.set_ylabel("nDCG@10 on BEIR/SciFact  →  better retrieval", color=TEXT)
    ax.set_title("Embedding bench: VL vs text vs e5 on pure-text retrieval (RTX 5090)", color=GOLD)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTE)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.margins(0.16)  # headroom so edge points' labels clear the title/frame
    ax.set_title(ax.get_title(), color=GOLD, pad=16)
    fig.tight_layout()
    fig.savefig("reports/embed-bench-scatter.png", dpi=150, facecolor=BG)
    print("wrote reports/embed-bench-scatter.png")


if __name__ == "__main__":
    main()
