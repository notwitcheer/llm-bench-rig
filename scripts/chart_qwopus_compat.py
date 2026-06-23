"""t069 Qwopus-Coder-Compat chart (Gold & Crimson v2), two panels, no bars.

The story is RECOVERY: the prior Qwopus-Coder had two measured regressions vs its
base (a degraded MTP draft head + real-SWE resolve below base). The Compat release
heals both at no capability cost.

left  = line: MTP self-speculative speedup per workload — the prior coder head sits
        flat-and-low (1.4-1.6x); the Compat head climbs back onto the base Qwen3.6
        head (1.9-2.3x).
right = dumbbell on a "% of base" axis: both regressions (draft-accept avg, SWE-12
        resolve) move from below base (prior) up to the base=100% line (Compat).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = ("#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64")

WORKLOADS = ["prose", "Q&A", "code", "JSON", "repetitive"]
BASE_HEAD   = [1.8, 1.9, 2.0, 2.2, 2.2]        # Qwen3.6-27B-MTP Q6_K (base, 2026-06-02)
PRIOR_HEAD  = [1.39, 1.36, 1.39, 1.60, 1.63]   # prior Qwopus-Coder (2026-06-12)
COMPAT_HEAD = [2.02, 1.95, 1.99, 2.28, 2.30]   # Qwopus-Coder-Compat (today)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.0), facecolor=BG,
                               gridspec_kw={"width_ratios": [1.3, 1]})
fig.suptitle("Qwopus3.6-27B-Coder-Compat: the prior tune's two regressions, healed",
             color=GOLD, fontsize=14.5, fontweight="bold", y=0.975)
fig.text(0.5, 0.925, "RTX 5090 · Q6_K · think-off · same harness as base + prior",
         color=MUTE, fontsize=9.5, ha="center")

# ---- left: MTP draft head recovery (line) ----
ax1.set_facecolor(BG)
x = np.arange(len(WORKLOADS))
ax1.plot(x, BASE_HEAD, color=MUTE, lw=1.6, ls="--", marker="o", ms=6,
         label="base Qwen3.6-27B head")
ax1.plot(x, PRIOR_HEAD, color=GOLD, lw=2.0, marker="o", ms=7,
         label="prior Qwopus-Coder head")
ax1.plot(x, COMPAT_HEAD, color=CRIMSON, lw=3.0, marker="D", ms=8,
         label="Compat head (today)")
ax1.fill_between(x, PRIOR_HEAD, COMPAT_HEAD, color=CRIMSON, alpha=0.08)
ax1.set_xticks(x); ax1.set_xticklabels(WORKLOADS, color=TEXT, fontsize=10)
ax1.set_ylim(1.2, 2.5)
ax1.set_ylabel("MTP self-speculative speedup (×)", color=TEXT, fontsize=10.5)
ax1.set_title("MTP draft head climbs back onto base", color=TEXT, fontsize=11.5, pad=8)
ax1.tick_params(colors=MUTE)
for s in ax1.spines.values(): s.set_color(GRID)
ax1.grid(True, color=GRID, lw=0.6, alpha=0.5)
ax1.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=9, loc="lower right")
ax1.annotate("flat & low\n(SFT broke the head)", (1.0, 1.36), color=GOLD, fontsize=8.5,
             ha="center", va="top", xytext=(0, -6), textcoords="offset points")

# ---- right: "% of base" dumbbell — both regressions healed ----
ax2.set_facecolor(BG)
ROWS = ["MTP draft-accept\n(avg of 5 workloads)", "SWE-bench Verified\nresolve (n=12)"]
PRIOR_PCT  = [73, 87]    # prior as % of base: 1.47/2.02 ; 7/8
COMPAT_PCT = [104, 100]  # Compat as % of base: 2.10/2.02 ; 8/8
ys = np.arange(len(ROWS))[::-1]
ax2.axvline(100, color=MUTE, lw=1.4, ls="--")
ax2.annotate("base = 100%", (100, len(ROWS) - 0.35), color=MUTE, fontsize=9,
             ha="center", va="bottom")
for y, lab, p, c in zip(ys, ROWS, PRIOR_PCT, COMPAT_PCT):
    ax2.plot([p, c], [y, y], color=CRIMSON, lw=3.0, zorder=2,
             solid_capstyle="round")
    ax2.scatter([p], [y], s=130, color=GOLD, zorder=3, marker="o")
    ax2.scatter([c], [y], s=150, color=CRIMSON, zorder=3, marker="D")
    ax2.annotate(f"{p}%", (p, y), color=GOLD, fontsize=10, va="center", ha="right",
                 xytext=(-10, 0), textcoords="offset points", fontweight="bold")
    ax2.annotate(f"{c}%", (c, y), color=CRIMSON, fontsize=10, va="center", ha="left",
                 xytext=(10, 0), textcoords="offset points", fontweight="bold")
    ax2.annotate(lab, (((p + c) / 2), y + 0.18), color=TEXT, fontsize=9.5,
                 ha="center", va="bottom")
ax2.set_xlim(60, 115)
ax2.set_ylim(-0.7, len(ROWS) + 0.1)
ax2.set_yticks([])
ax2.set_xlabel("performance as % of base Qwen3.6-27B", color=TEXT, fontsize=10.5)
ax2.set_title("prior (below base)  →  Compat (at base)", color=TEXT, fontsize=11.5, pad=8)
ax2.tick_params(colors=MUTE)
for s in ax2.spines.values(): s.set_color(GRID)

fig.text(0.99, 0.012, "WITCHEER", color=MUTE, fontsize=9, ha="right", fontweight="bold")
fig.subplots_adjust(left=0.085, right=0.965, top=0.875, bottom=0.105, wspace=0.18)
fig.savefig("reports/qwopus-coder-compat-recovery.png", dpi=150, facecolor=BG)
print("wrote reports/qwopus-coder-compat-recovery.png")
