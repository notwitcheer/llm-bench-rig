"""North-Mini-Code-1.0 on the agentic board — Cohere's agentic-coding MoE lands LAST (9th)
of 9 despite the tuning, tied with Nex-N2-mini. Ranked lollipop of Agentic Score; North-Mini
CRIMSON, prior leader GOLD, field MUTE. Active-param size annotated to show it's not simply
a small-active penalty (Qwen3.5-35B-A3B and Nemotron-30B-A3B sit near the top)."""
import json
import matplotlib.pyplot as plt

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = ("#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64")

rows = json.load(open("leaderboard/leaderboard.json"))
rows = sorted(rows, key=lambda r: r["agentic_score"])  # ascending -> last at bottom
names = [r["model"] for r in rows]
scores = [r["agentic_score"] for r in rows]
params = [r["params"] for r in rows]
sub = next(i for i, r in enumerate(rows) if "North-Mini" in r["model"])
top = max(range(len(rows)), key=lambda i: scores[i])
cols = [MUTE] * len(rows)
cols[top] = GOLD
cols[sub] = CRIMSON

fig, ax = plt.subplots(figsize=(10, 6.6), facecolor=BG); ax.set_facecolor(BG)
ys = range(len(rows))
ax.hlines(list(ys), min(scores) - 2, scores, color=GRID, linewidth=1.4, zorder=1)
ax.scatter(scores, list(ys), color=cols, edgecolor=TEXT, s=150, zorder=3)
for i, (s, p) in enumerate(zip(scores, params)):
    ax.annotate(f"{s:.1f}   ·   {p}", (s, i), color=TEXT, fontsize=9.5,
                xytext=(9, 0), textcoords="offset points", va="center")
ax.set_yticks(list(ys)); ax.set_yticklabels(names, color=TEXT, fontsize=9.5)
ax.set_xlabel("Agentic Score  (40-task native tool-calling harness)", color=TEXT)
ax.set_title("North-Mini-Code-1.0 lands LAST on the agentic board (RTX 5090)\n"
             "Cohere's agentic-coding MoE (90.4) ties Nex-N2-mini, below generalist A3B models",
             color=GOLD, pad=14, fontsize=12)
ax.set_xlim(min(scores) - 3, max(scores) + 9)
for s in ax.spines.values(): s.set_color(GRID)
ax.tick_params(colors=MUTE)
fig.tight_layout(); fig.savefig("reports/north-mini-board.png", dpi=150, facecolor=BG)
print("wrote reports/north-mini-board.png")
