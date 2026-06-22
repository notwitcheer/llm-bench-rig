# scripts/chart_tts.py
"""Gold & Crimson v2: Sovereign TTS head-to-head on one RTX 5090 (sm_120). One horizontal DUMBBELL
per axis (round-trip WER, SIM-o, RTFx, first-audio latency) — two dots (Qwen vs Fish) joined by a
bar so the gap reads directly, not a wall of bars. Reads results/tts/<slug>.json (the `agg` block).
Runs on the Mac (matplotlib). Honest framing: neither model compiled; vendor speed claims were H200."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = ("#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64")

# (slug, label, color)
MODELS = [("qwen3-tts", "Qwen3-TTS-1.7B", GOLD), ("fish-s2-pro", "Fish-S2-Pro (4B)", CRIMSON),
          ("fish-s2-pro-compiled", "Fish-S2-Pro compiled", "#e0a030")]
# (agg key, axis label, higher_is_better, value format)
AXES = [
    ("rtfx_mean", "RTFx  (higher = faster than realtime)", True, "{:.2f}×"),
    ("sim_mean", "SIM-o  (higher = better speaker clone)", True, "{:.3f}"),
    ("wer_mean", "round-trip WER  (lower = better)", False, "{:.1%}"),
    ("first_audio_s_mean", "first-audio latency s  (lower = better)", False, "{:.2f}s"),
]

D = {s: json.load(open(f"results/tts/{s}.json"))["agg"] for s, _, _ in MODELS}
n = {s: json.load(open(f"results/tts/{s}.json"))["agg"].get("n", "?") for s, _, _ in MODELS}

fig, axs = plt.subplots(len(AXES), 1, figsize=(11, 7.5), facecolor=BG)
for ax, (key, label, hib, fmt) in zip(axs, AXES):
    ax.set_facecolor(BG)
    vals = [(D[s].get(key) or 0.0, lbl, col) for s, lbl, col in MODELS]
    lo, hi = min(v for v, _, _ in vals), max(v for v, _, _ in vals)
    # the connecting bar
    ax.plot([lo, hi], [0, 0], color=GRID, lw=3, zorder=1, solid_capstyle="round")
    span = (hi * 1.20) or 1.0
    last_x = -1e9
    for v, lbl, col in sorted(vals, key=lambda t: t[0]):
        ax.scatter([v], [0], s=240, color=col, edgecolor=TEXT, lw=1.2, zorder=3)
        close = (v - last_x) < 0.10 * span      # too near the previous label -> drop this one below
        ax.annotate(fmt.format(v), (v, 0), xytext=(0, -24 if close else 13),
                    textcoords="offset points", color=TEXT, ha="center",
                    va="top" if close else "bottom", fontsize=10, fontweight="bold", zorder=4)
        last_x = v
    # verdict: a near-equal gap is a tie, not a "winner" (avoids exaggerating noise)
    rel = abs(hi - lo) / (hi or 1.0)
    if rel < 0.03:
        verdict = "≈ tie"
    else:
        w = (max if hib else min)(vals, key=lambda t: t[0])
        verdict = f"winner: {w[1].split()[0]}"
    ax.set_title(f"{label}   — {verdict}", color=GOLD, fontsize=11, loc="left", pad=10)
    # baseline at 0 so tiny absolute gaps (e.g. a WER tie) aren't exaggerated by auto-scale
    ax.set_xlim(0, hi * 1.20)
    ax.set_ylim(-1.8, 1.6)
    ax.get_yaxis().set_visible(False)
    ax.tick_params(colors=MUTE, labelsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)

# legend via proxy dots
handles = [plt.Line2D([0], [0], marker="o", color=BG, markerfacecolor=c, markeredgecolor=TEXT,
                      markersize=11, label=f"{l}  (n={n[s]})") for s, l, c in MODELS]
fig.legend(handles=handles, loc="upper right", frameon=False, labelcolor=TEXT, fontsize=9,
           bbox_to_anchor=(0.99, 0.99))
fig.suptitle("Sovereign TTS head-to-head on one RTX 5090 — Seed-TTS-eval EN",
             color=GOLD, fontsize=15, x=0.06, ha="left", y=0.99)
fig.text(0.06, 0.005, "Fish shown bf16 out-of-box and with torch.compile (4.1x faster, still behind the 1.7B). Vendor RTF claims were H100/H200-class.",
         color=MUTE, fontsize=8, ha="left")
fig.tight_layout(rect=[0, 0.02, 1, 0.94])
fig.savefig("reports/tts-head-to-head.png", dpi=150, facecolor=BG)
print("wrote reports/tts-head-to-head.png")
