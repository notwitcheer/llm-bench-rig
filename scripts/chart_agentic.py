"""Agentic eval harness charts (Gold & Crimson v2): Nex-N2-mini (post-train, CRIMSON)
vs base Qwen3.5-35B-A3B (GOLD) on the native tool-calling Agentic Score.
(1) the four axes + Agentic Score, grouped bars.
(2) the hero: tokens/task at equal success -> the Adaptive-Thinking ~20% verdict."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = ("#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64")

NEX = json.loads(Path("results/nex-n2-mini/agentic_native.json").read_text())
BASE = json.loads(Path("results/qwen3-5-35b-base/agentic_native.json").read_text())

# --- Chart 1: the four axes + Agentic Score (all 0-100 scale) ---
AXES = ["Task\nsuccess %", "Tool\nefficiency", "Loop\nstability %", "Agentic\nScore"]
def row(d):
    return [d["task_success_pct"], d["tool_eff"] * 100, d["stable_pct"], d["score"]]
nex_v, base_v = row(NEX), row(BASE)

fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG); ax.set_facecolor(BG)
xs = np.arange(len(AXES)); w = 0.38
ax.bar(xs - w/2, base_v, w, color=GOLD, edgecolor=TEXT, zorder=3, label="base Qwen3.5-35B-A3B")
ax.bar(xs + w/2, nex_v, w, color=CRIMSON, edgecolor=TEXT, zorder=3, label="Nex-N2-mini (agentic post-train)")
for i, (b, n) in enumerate(zip(base_v, nex_v)):
    ax.annotate(f"{b:.0f}", (i - w/2, b), color=MUTE, ha="center", fontsize=10, xytext=(0, 4), textcoords="offset points")
    ax.annotate(f"{n:.0f}", (i + w/2, n), color=TEXT, ha="center", fontsize=11, fontweight="bold", xytext=(0, 4), textcoords="offset points")
ax.set_xticks(xs); ax.set_xticklabels(AXES, color=TEXT, fontsize=11)
ax.set_ylabel("score (0-100)", color=TEXT); ax.set_ylim(0, 122)
ax.set_title("Agentic post-training, measured: Nex-N2-mini vs its base, one 5090", color=GOLD, pad=16)
for s in ax.spines.values(): s.set_color(GRID)
ax.tick_params(colors=MUTE); ax.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)
ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, loc="upper center", ncol=2, framealpha=0.95)
fig.tight_layout(); fig.savefig("reports/agentic-nex-n2-mini.png", dpi=150, facecolor=BG)
print("wrote reports/agentic-nex-n2-mini.png")

# --- Chart 2: the hero -- tokens/task at equal success (Adaptive-Thinking verdict) ---
b_tok, n_tok = BASE["tokens_per_task"], NEX["tokens_per_task"]
delta = (n_tok - b_tok) / b_tok * 100 if b_tok else 0.0
fig, ax = plt.subplots(figsize=(8.5, 6), facecolor=BG); ax.set_facecolor(BG)
pairs = [("base\nQwen3.5-35B", b_tok, GOLD, f"{BASE['task_success_pct']:.0f}% success"),
         ("Nex-N2-mini\n(Adaptive Thinking)", n_tok, CRIMSON, f"{NEX['task_success_pct']:.0f}% success")]
xs = np.arange(len(pairs))
ax.bar(xs, [p[1] for p in pairs], 0.55, color=[p[2] for p in pairs], edgecolor=TEXT, zorder=3)
for i, p in enumerate(pairs):
    ax.annotate(f"{p[1]:.0f} tok", (i, p[1]), color=TEXT, ha="center", fontsize=13, fontweight="bold", xytext=(0, 16), textcoords="offset points")
    ax.annotate(p[3], (i, p[1]), color=MUTE, ha="center", fontsize=10, xytext=(0, 3), textcoords="offset points")
ax.set_xticks(xs); ax.set_xticklabels([p[0] for p in pairs], color=TEXT, fontsize=11)
ax.set_ylabel("avg tokens / task", color=TEXT)
ax.set_ylim(0, max(b_tok, n_tok) * 1.28 + 1)
verdict = f"{delta:+.0f}% tokens" + (" -- claim holds" if delta <= -10 else " -- claim NOT seen" if delta > -5 else "")
ax.set_title(f"Adaptive Thinking verdict: Nex uses {verdict}", color=GOLD, pad=16)
for s in ax.spines.values(): s.set_color(GRID)
ax.tick_params(colors=MUTE); ax.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)
fig.tight_layout(); fig.savefig("reports/agentic-tokens.png", dpi=150, facecolor=BG)
print("wrote reports/agentic-tokens.png")
print(f"token delta: {delta:+.1f}%  (base {b_tok} -> nex {n_tok})")
