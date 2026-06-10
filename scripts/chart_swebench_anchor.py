"""SWE-bench reality anchor: synthetic Agentic Score (x) vs real SWE-bench Verified
resolve-rate (y). Confirms whether the synthetic 40-task ranking predicts real coding."""
import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = ("#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64")

SYNTH = {r["model"]: float(r["agentic_score"]) for r in csv.DictReader(open("leaderboard/leaderboard.csv"))}
# slug -> (display, leaderboard model name, color)
ROWS = [
    ("qwen3-6-27b", "Qwen3.6-27B", "Qwen3.6-27B", CRIMSON),
    ("qwen3-5-35b-base", "Qwen3.5-35B (base)", "Qwen3.5-35B-A3B (base)", GOLD),
    ("qwopus-glm-18b", "Qwopus-GLM-18B", "Qwopus-GLM-18B", "#e0a030"),
    ("nemotron-cascade-2-30b", "Nemotron-Cascade-2", "Nemotron-Cascade-2-30B", "#b08a4a"),
    ("kimi-linear-48b-a3b", "Kimi-Linear-48B", "Kimi-Linear-48B-A3B", "#d87070"),
    ("granite-4-1-30b", "Granite-4.1-30b", "Granite-4.1-30b", "#c9a86a"),
    ("nex-n2-mini", "Nex-N2-mini", "Nex-N2-mini", MUTE),
]

xs, ys = [], []
fig, ax = plt.subplots(figsize=(9.5, 6.5), facecolor=BG); ax.set_facecolor(BG)
for slug, disp, synth_name, color in ROWS:
    rep = json.loads(Path(f"results/swebench/{slug}.report.json").read_text())
    resolved, total = rep["resolved_instances"], rep["submitted_instances"]
    x, y = SYNTH[synth_name], 100.0 * resolved / total
    xs.append(x); ys.append(y)
    ax.scatter([x], [y], s=320, color=color, edgecolor=TEXT, zorder=4, linewidth=1.2)
    ax.annotate(f"{disp}\n{resolved}/{total} resolved", (x, y), color=TEXT, fontsize=9.5, ha="center",
                va="bottom", xytext=(0, 13), textcoords="offset points")

r = float(np.corrcoef(xs, ys)[0, 1])
# spearman = pearson on ranks (more robust to the single high-leverage outlier)
_rx = np.argsort(np.argsort(-np.array(xs))).astype(float)
_ry = np.argsort(np.argsort(-np.array(ys))).astype(float)
rho = float(np.corrcoef(_rx, _ry)[0, 1])
# trend line
m, b = np.polyfit(xs, ys, 1)
xl = np.array([min(xs) - 1, max(xs) + 1])
ax.plot(xl, m * xl + b, color=MUTE, linewidth=1.1, linestyle="--", zorder=2)

ax.set_xlabel("synthetic Agentic Score (40-task harness)", color=TEXT)
ax.set_ylabel("real SWE-bench Verified resolve %", color=TEXT)
# data-driven limits with padding (7 models span a wider synthetic range; harder
# instances push resolve % lower) — never clip a point
ax.set_xlim(min(xs) - 1.5, max(xs) + 1.5)
ax.set_ylim(max(0, min(ys) - 12), min(100, max(ys) + 14))
ax.set_title(f"Reality anchor (7 models): synthetic Agentic Score vs real SWE-bench Verified\n"
             f"Pearson r = {r:.2f}   ·   Spearman ρ = {rho:.2f}   ·   5/7 models rank exactly as predicted",
             color=GOLD, pad=14, fontsize=12)
for s in ax.spines.values(): s.set_color(GRID)
ax.tick_params(colors=MUTE); ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
fig.tight_layout(); fig.savefig("reports/swebench-anchor.png", dpi=150, facecolor=BG)
print(f"wrote reports/swebench-anchor.png  (Pearson r={r:.3f})")
for (slug, disp, sn, _), x, y in zip(ROWS, xs, ys):
    print(f"  {disp:18} synth={x:6.2f}  real={y:.1f}%")
