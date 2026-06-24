"""Cacheback / draft-free spec-decode chart (Gold & Crimson v2).

Slope ladder, NOT bars: x = drafter ladder (AR -> PLD -> Cacheback), y = realized speedup
vs AR, one line per workload. MAT annotated at each point. The shape IS the finding: which
workload climbs, and whether the fancy LRU table (Cacheback) beats dumb prompt-lookup (PLD)
or just lands on top of it. Measured on Qwen3-8B bf16, RTX 5090, batch=1 greedy."""
import glob
import json

import matplotlib.pyplot as plt

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = ("#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64")

rows = [json.load(open(f)) for f in glob.glob("results/cacheback/*.json")]
arms = [a for a in ("ar", "pld", "cacheback") if any(r["arm"] == a for r in rows)]
workloads = ["code", "copyctx", "chat"]            # ordered by expected n-gram repetition
LABELS = {"ar": "AR\n(no draft)", "pld": "PLD\n(prompt-lookup)", "cacheback": "Cacheback\n(LRU table)"}
COLOR = {"code": CRIMSON, "copyctx": GOLD, "chat": MUTE}  # code = the sweet spot, highlighted

def cell(wl, arm):
    return next(r for r in rows if r["workload"] == wl and r["arm"] == arm)

base = {wl: cell(wl, "ar")["tok_per_s"] for wl in workloads}
xs = list(range(len(arms)))

fig, ax = plt.subplots(figsize=(9.6, 6.6), facecolor=BG)
ax.set_facecolor(BG)
fig.suptitle("Draft-free spec-decode on Qwen3-8B  ·  RTX 5090",
             color=GOLD, fontsize=15, fontweight="bold", y=0.965)

# stagger label vertical offset per workload so the close copyctx/chat pair doesn't collide
VOFF = {"code": 9, "copyctx": 9, "chat": -24}
for wl in workloads:
    ys = [cell(wl, a)["tok_per_s"] / base[wl] for a in arms]
    ax.plot(xs, ys, "-o", color=COLOR[wl], linewidth=2.6, markersize=9,
            markeredgecolor=TEXT, zorder=3, label=wl)
    for x, a in zip(xs, arms):
        if a == "ar":
            continue                                  # baseline = 1.00x (dashed line shows it)
        c = cell(wl, a)
        sp = c["tok_per_s"] / base[wl]
        ax.annotate(f"{sp:.2f}x\nMAT {c['mat']:.2f}", (x, sp), color=COLOR[wl], ha="center",
                    va="bottom" if VOFF[wl] > 0 else "top", fontsize=9.5, fontweight="bold",
                    xytext=(0, VOFF[wl]), textcoords="offset points")

ax.axhline(1.0, color=GRID, linewidth=1.0, linestyle="--", zorder=1)
ax.text(len(arms) - 1, 1.012, "PLD → Cacheback: flat\n(the LRU table buys nothing)",
        color=MUTE, ha="right", va="bottom", fontsize=9.5, style="italic")
ax.set_xticks(xs)
ax.set_xticklabels([LABELS[a] for a in arms], color=TEXT, fontsize=11)
ax.set_xlim(-0.25, len(arms) - 0.75)
ax.set_ylabel("realized speedup vs autoregressive", color=TEXT, fontsize=11)
ax.set_title("greedy · batch=1 · lossless: token-identical to AR modulo bf16 ties · speedup <= MAT, verified",
             color=MUTE, fontsize=9.5, pad=8)
ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=11, loc="center left",
          title="workload", title_fontsize=10)
for s in ax.spines.values():
    s.set_color(GRID)
ax.tick_params(colors=MUTE)
ax.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)

fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("reports/cacheback-spec-decode.png", dpi=150, facecolor=BG)
print("wrote reports/cacheback-spec-decode.png")
