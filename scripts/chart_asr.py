"""Gold-&-Crimson ASR chart: WER (x, lower better) vs RTFx (y, higher better) from
results/asr/summary.json. The subject (parakeet) is crimson; whisper baselines gold/mute."""
import json
import sys
from pathlib import Path

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = ("#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64")


def label_for(s: dict) -> str:
    return f"{s['model']}\nWER {s['wer_clean']*100:.1f}/{s['wer_other']*100:.1f}%  ·  {s['rtfx']:.0f}x"


def main(summary_path="results/asr/summary.json"):
    import matplotlib.pyplot as plt
    rows = json.loads(Path(summary_path).read_text())
    fig, ax = plt.subplots(figsize=(9.5, 6.5), facecolor=BG); ax.set_facecolor(BG)
    for s in rows:
        color = CRIMSON if s["model"].startswith("parakeet") else (GOLD if "turbo" in s["model"] else MUTE)
        ax.scatter([s["wer_other"] * 100], [s["rtfx"]], s=320, color=color, edgecolor=TEXT, zorder=4, linewidth=1.2)
        ax.annotate(label_for(s), (s["wer_other"] * 100, s["rtfx"]), color=TEXT, fontsize=9.5,
                    ha="center", va="bottom", xytext=(0, 13), textcoords="offset points")
    ax.set_xlabel("WER % on test-other (lower better)", color=TEXT)
    ax.set_ylabel("RTFx — audio-sec / proc-sec (higher better)", color=TEXT)
    ax.set_title("Sovereign ASR on one RTX 5090: 2026 TDT transducer vs 2022 attention",
                 color=GOLD, pad=16, fontsize=12.5)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.tick_params(colors=MUTE); ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    fig.tight_layout(); fig.savefig("reports/asr-head-to-head.png", dpi=150, facecolor=BG)
    print("wrote reports/asr-head-to-head.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/asr/summary.json")
