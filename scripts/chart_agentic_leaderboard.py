"""Agentic Score leaderboard (Gold & Crimson v2) across 4 models on the native
tool-calling bench. (1) the efficiency frontier: success % vs tokens/task — the
sweet spot is top-LEFT (high success, few tokens). (2) Agentic Score bars."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = ("#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64")
AMBER = "#d8902f"

# slug, display, color  (Granite highlighted crimson = the find)
MODELS = [
    ("granite-4-1-30b",     "Granite-4.1-30b",      CRIMSON),
    ("qwen3-5-35b-base",    "Qwen3.5-35B-A3B (base)", GOLD),
    ("nex-n2-mini",         "Nex-N2-mini",          AMBER),
    ("kimi-linear-48b-a3b", "Kimi-Linear-48B-A3B",  MUTE),
]
D = {slug: json.loads(Path(f"results/{slug}/agentic_native.json").read_text()) for slug, _, _ in MODELS}

# --- Chart 1: efficiency frontier (success vs tokens/task) ---
fig, ax = plt.subplots(figsize=(10, 6.5), facecolor=BG); ax.set_facecolor(BG)
for slug, name, color in MODELS:
    d = D[slug]
    x, y = d["tokens_per_task"], d["task_success_pct"]
    ax.scatter([x], [y], s=320, color=color, edgecolor=TEXT, zorder=4, linewidth=1.2)
    dy = -1.6 if name.startswith("Granite") else 1.1
    va = "top" if dy < 0 else "bottom"
    ax.annotate(f"{name}\nscore {d['score']}  ·  {d['tool_eff']:.2f} tool-eff",
                (x, y), color=TEXT, fontsize=10, ha="center", va=va,
                xytext=(0, 14 if dy > 0 else -14), textcoords="offset points")
ax.annotate("sweet spot\n(high success, few tokens)", (78, 99.4), color=CRIMSON, fontsize=9.5,
            ha="left", va="top", style="italic")
ax.set_xlabel("tokens / task  (lower = leaner)", color=TEXT)
ax.set_ylabel("task success %", color=TEXT)
ax.set_xlim(40, 290); ax.set_ylim(82, 103)
ax.set_title("Agentic efficiency frontier: Granite-4.1-30b does near-base work at a third of the tokens",
             color=GOLD, pad=16, fontsize=12.5)
for s in ax.spines.values(): s.set_color(GRID)
ax.tick_params(colors=MUTE); ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
fig.tight_layout(); fig.savefig("reports/agentic-leaderboard-frontier.png", dpi=150, facecolor=BG)
print("wrote reports/agentic-leaderboard-frontier.png")

# --- Chart 2: Agentic Score bars ---
fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG); ax.set_facecolor(BG)
xs = np.arange(len(MODELS))
vals = [D[s]["score"] for s, _, _ in MODELS]
cols = [c for _, _, c in MODELS]
ax.bar(xs, vals, 0.6, color=cols, edgecolor=TEXT, zorder=3)
for i, (s, name, c) in enumerate(MODELS):
    d = D[s]
    ax.annotate(f"{d['score']}", (i, d["score"]), color=TEXT, ha="center", fontsize=13, fontweight="bold",
                xytext=(0, 14), textcoords="offset points")
    ax.annotate(f"{d['task_success_pct']:.0f}% · {d['tokens_per_task']:.0f} tok",
                (i, d["score"]), color=MUTE, ha="center", fontsize=9, xytext=(0, 2), textcoords="offset points")
ax.set_xticks(xs); ax.set_xticklabels([n for _, n, _ in MODELS], color=TEXT, fontsize=10)
ax.set_ylabel("Agentic Score (0-100)", color=TEXT); ax.set_ylim(0, 112)
ax.set_title("Agentic Score — native tool-calling bench, one RTX 5090", color=GOLD, pad=16)
for s in ax.spines.values(): s.set_color(GRID)
ax.tick_params(colors=MUTE); ax.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)
fig.tight_layout(); fig.savefig("reports/agentic-leaderboard-score.png", dpi=150, facecolor=BG)
print("wrote reports/agentic-leaderboard-score.png")
