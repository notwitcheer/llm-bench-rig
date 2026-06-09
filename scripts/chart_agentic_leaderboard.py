"""Agentic Score leaderboard (Gold & Crimson v2) — hardened 36-task suite + long-context.
(1) efficiency frontier: success % vs tokens/task. (2) Agentic Score bars.
(3) long-context reach: needle success at 32K vs 128K, with VRAM walls marked."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = ("#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64")
AMBER = "#d8902f"

# Kimi highlighted crimson = the hardened-suite find (best all-rounder + holds 128K)
MODELS = [
    ("kimi-linear-48b-a3b", "Kimi-Linear-48B-A3B",  CRIMSON),
    ("qwen3-5-35b-base",    "Qwen3.5-35B-A3B (base)", GOLD),
    ("granite-4-1-30b",     "Granite-4.1-30b",      AMBER),
    ("nex-n2-mini",         "Nex-N2-mini",          MUTE),
]
D = {s: json.loads(Path(f"results/{s}/agentic_native.json").read_text()) for s, _, _ in MODELS}


def lc(slug, tier):
    f = Path(f"results/{slug}/agentic_longctx_{tier}.json")
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return "wall" if d.get("reach") == "wall" else d["success_pct"]


# --- Chart 1: efficiency frontier ---
fig, ax = plt.subplots(figsize=(10, 6.5), facecolor=BG); ax.set_facecolor(BG)
for slug, name, color in MODELS:
    d = D[slug]
    x, y = d["tokens_per_task"], d["task_success_pct"]
    ax.scatter([x], [y], s=320, color=color, edgecolor=TEXT, zorder=4, linewidth=1.2)
    ax.annotate(f"{name}\nscore {d['score']}  ·  {d['tool_eff']:.2f} tool-eff",
                (x, y), color=TEXT, fontsize=10, ha="center", va="bottom",
                xytext=(0, 14), textcoords="offset points")
ax.set_xlabel("tokens / task  (lower = leaner)", color=TEXT)
ax.set_ylabel("task success %  (36-task hardened suite)", color=TEXT)
ax.set_xlim(40, 290); ax.set_ylim(78, 106)
ax.set_title("Hardened agentic suite: efficiency and robustness pull apart", color=GOLD, pad=16, fontsize=12.5)
for s in ax.spines.values(): s.set_color(GRID)
ax.tick_params(colors=MUTE); ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
fig.tight_layout(); fig.savefig("reports/agentic-leaderboard-frontier.png", dpi=150, facecolor=BG)
print("wrote reports/agentic-leaderboard-frontier.png")

# --- Chart 2: Agentic Score bars ---
fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG); ax.set_facecolor(BG)
order = sorted(MODELS, key=lambda m: D[m[0]]["score"], reverse=True)
xs = np.arange(len(order))
ax.bar(xs, [D[s]["score"] for s, _, _ in order], 0.6, color=[c for _, _, c in order], edgecolor=TEXT, zorder=3)
for i, (s, name, c) in enumerate(order):
    d = D[s]
    ax.annotate(f"{d['score']}", (i, d["score"]), color=TEXT, ha="center", fontsize=13, fontweight="bold",
                xytext=(0, 14), textcoords="offset points")
    ax.annotate(f"{d['task_success_pct']:.0f}% · {d['tokens_per_task']:.0f} tok",
                (i, d["score"]), color=MUTE, ha="center", fontsize=9, xytext=(0, 2), textcoords="offset points")
ax.set_xticks(xs); ax.set_xticklabels([n for _, n, _ in order], color=TEXT, fontsize=10)
ax.set_ylabel("Agentic Score (0-100)", color=TEXT); ax.set_ylim(0, 112)
ax.set_title("Agentic Score — hardened native tool-calling suite, one RTX 5090", color=GOLD, pad=16)
for s in ax.spines.values(): s.set_color(GRID)
ax.tick_params(colors=MUTE); ax.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)
fig.tight_layout(); fig.savefig("reports/agentic-leaderboard-score.png", dpi=150, facecolor=BG)
print("wrote reports/agentic-leaderboard-score.png")

# --- Chart 3: long-context reach (needle success at 32K vs 128K) ---
fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG); ax.set_facecolor(BG)
xs = np.arange(len(MODELS)); w = 0.38
v32 = [lc(s, "32k") for s, _, _ in MODELS]
v128 = [lc(s, "128k") for s, _, _ in MODELS]
ax.bar(xs - w/2, [0 if v == "wall" or v is None else v for v in v32], w, color=GOLD, edgecolor=TEXT, zorder=3, label="32K")
ax.bar(xs + w/2, [0 if v == "wall" or v is None else v for v in v128], w, color=CRIMSON, edgecolor=TEXT, zorder=3, label="128K")
for i, (v, off) in enumerate([(v, -w/2) for v in v32] + [(v, w/2) for v in v128]):
    mi = i % len(MODELS)
    val = (v32 if i < len(MODELS) else v128)[mi]
    if val == "wall":
        ax.annotate("VRAM\nwall", (mi + off, 3), color=MUTE, ha="center", va="bottom", fontsize=8.5, style="italic")
    elif val is not None:
        ax.annotate(f"{val:.0f}%", (mi + off, val), color=TEXT, ha="center", fontsize=10, xytext=(0, 3), textcoords="offset points")
ax.set_xticks(xs); ax.set_xticklabels([n for _, n, _ in MODELS], color=TEXT, fontsize=10)
ax.set_ylabel("needle success %", color=TEXT); ax.set_ylim(0, 112)
ax.set_title("Long-context reach: who finds the needle at 32K vs 128K on one 5090", color=GOLD, pad=16)
for s in ax.spines.values(): s.set_color(GRID)
ax.tick_params(colors=MUTE); ax.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)
ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, loc="lower left")
fig.tight_layout(); fig.savefig("reports/agentic-leaderboard-longctx.png", dpi=150, facecolor=BG)
print("wrote reports/agentic-leaderboard-longctx.png")
