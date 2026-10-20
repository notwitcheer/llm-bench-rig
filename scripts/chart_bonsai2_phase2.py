#!/usr/bin/env python3
"""Ternary-Bonsai-2-27B phase 2 card (2026-10-20): left, AIME 2026 + HMMT Feb 2026 solved of 63 at a 32k thinking
budget (dataset/matharena_contest_32k.csv); right, GPQA Diamond think-on 16k split into finished-right, capped-right
and wrong (per-item counts read from the rig's gpqa_progress.json / gpqa_tokens.jsonl, passed in GPQA below).
writes /opt/data/tmp/bonsai2/bonsai2-phase2.png."""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, TEXT, MUTE, GRID = "#0e1420", "#eaf0f7", "#7d8ba0", "#20293a"
BASE, TERN, CAPC, WRONG = "#6fd6a8", "#7aa7e8", "#e8b64c", "#3a4458"
CSV = "/opt/data/repos/llm-bench-rig/dataset/matharena_contest_32k.csv"
ROWS = [  # slug, label, colour
    ("qwen3-8-27b-ud-iq3-xxs", "Qwen3.8-27B · UD-IQ3_XXS · 11.9 GB", BASE),
    ("qwen3-8-27b-q6-k", "Qwen3.8-27B · Q6_K · 22.9 GB", BASE),
    ("ternary-bonsai-2-27b-pq2-0", "Ternary-Bonsai-2-27B · PQ2_0 · 7.2 GB †", TERN),
    ("ternary-bonsai-2-27b-ptq1-0", "Ternary-Bonsai-2-27B · PTQ1_0 · 5.9 GB †", TERN),
]
# GPQA Diamond think-on 16k, 198 questions: (correct, capped, capped_correct), from the progress + tokens files
# (Q6_K: qwen3-8-27b-q6-k-thinkon-rerun, capped = completion >= 16,376 tokens from its sidecar)
GPQA = {
    "qwen3-8-27b-q6-k": (157, 43, 8),
    "ternary-bonsai-2-27b-pq2-0": (138, 67, 13),
    "ternary-bonsai-2-27b-ptq1-0": (136, 64, 7),
}
con = {}
for r in csv.DictReader(open(CSV)):
    con.setdefault(r["slug"], {})[r["task"]] = r
solved = {s: sum(int(con[s][t]["correct"]) for t in ("aime_2026", "hmmt_feb_2026")) for s, _, _ in ROWS}
capped = {s: sum(int(con[s][t]["capped_count"]) for t in ("aime_2026", "hmmt_feb_2026")) for s, _, _ in ROWS}
assert solved == {"qwen3-8-27b-ud-iq3-xxs": 48, "qwen3-8-27b-q6-k": 47, "ternary-bonsai-2-27b-pq2-0": 46,
                  "ternary-bonsai-2-27b-ptq1-0": 43}, solved

fig = plt.figure(figsize=(16, 7.2), dpi=110)
fig.patch.set_facecolor(BG)
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.2, left=0.235, right=0.975, top=0.70, bottom=0.14)
ax = fig.add_subplot(gs[0]); bx = fig.add_subplot(gs[1])
for a in (ax, bx):
    a.set_facecolor(BG)
    for sp in a.spines.values():
        sp.set_visible(False)
    a.tick_params(length=0, colors=MUTE)
    a.grid(axis="x", color=GRID, lw=1, zorder=0)

n = len(ROWS)
for i, (s, lab, col) in enumerate(ROWS):
    y = n - 1 - i
    ax.barh(y, solved[s], color=col, height=0.62, zorder=3)
    ax.text(solved[s] - 1, y, f"{solved[s]}", ha="right", va="center", color=BG, fontsize=15, weight="bold", zorder=4)
    ax.text(solved[s] + 0.8, y, f"{capped[s]} hit the cap", ha="left", va="center", color=MUTE, fontsize=10.5)
    ax.text(-1.5, y, lab, ha="right", va="center", color=TEXT, fontsize=11.6)
ax.set_xlim(0, 72); ax.set_ylim(-0.6, n - 0.4); ax.set_yticks([])
ax.set_xticks([0, 15, 30, 45, 63]); ax.tick_params(axis="x", labelsize=10)
ax.text(0, n - 0.1, "AIME 2026 + HMMT Feb 2026, solved of 63\nthinking budget 32k", ha="left", va="bottom", color=TEXT,
        fontsize=12.5)

for i, (s, lab, col) in enumerate(ROWS):
    y = n - 1 - i
    if s not in GPQA:
        bx.barh(y, 156, color=col, height=0.62, zorder=3, alpha=0.55)
        bx.text(154, y, "78.8", ha="right", va="center", color=BG, fontsize=14, weight="bold", zorder=4)
        bx.text(158, y, "no cap data", ha="left", va="center", color=MUTE, fontsize=10.5)
        continue
    c, cap, capc = GPQA[s]
    fin_right = c - capc
    bx.barh(y, fin_right, color=col, height=0.62, zorder=3)
    bx.barh(y, capc, left=fin_right, color=CAPC, height=0.62, zorder=3)
    bx.barh(y, cap - capc, left=c, color=CAPC, alpha=0.35, height=0.62, zorder=3, hatch="///", ec=CAPC, lw=0)
    bx.text(fin_right - 2, y, f"{c / 198 * 100:.1f}", ha="right", va="center", color=BG, fontsize=14, weight="bold",
            zorder=4)
    bx.text(c + (cap - capc) + 2, y, f"{cap} capped", ha="left", va="center", color=CAPC, fontsize=10.5)
bx.set_xlim(0, 240); bx.set_ylim(-0.6, n - 0.4); bx.set_yticks([])
bx.set_xticks([0, 50, 100, 150, 198]); bx.tick_params(axis="x", labelsize=10)
bx.text(0, n - 0.1, "GPQA Diamond, 198 questions, thinking budget 16k\nsolid: right · amber: hit the cap (solid right, hatched wrong)\nfaded: no per-item cap data",
        ha="left", va="bottom", color=TEXT, fontsize=12.5)

fig.text(0.02, 0.935, "Ternary-Bonsai-2-27B: level with the 6-bit file on this year's contests, cut short on GPQA at 16k",
         color=TEXT, fontsize=19, weight="bold")
fig.text(0.02, 0.9, "same base model (Qwen3.8-27B), one RTX 5090, greedy, zero-shot. on the GPQA questions that finished "
         "inside 16k, Bonsai is right 95.4% (PQ2_0) and 96.3% (PTQ1_0), Q6_K 96.1%.\n"
         "llama.cpp b10371 for the Qwen3.8 rows. † Bonsai rows on the PrismML llama.cpp fork (its packings are not in mainline). "
         "a GPQA pass at 32k is queued.", color=MUTE, fontsize=10.8, va="top")
fig.text(0.02, 0.025, "witcheer · llm-bench-rig dataset/matharena_contest_32k.csv + reports/sub10-qwen38-27b.md · "
         "sizes are the GGUF on disk · Bonsai legs run 2026-10-11 to 2026-10-20", color=MUTE, fontsize=9.5)
os.makedirs("/opt/data/tmp/bonsai2", exist_ok=True)
out = "/opt/data/tmp/bonsai2/bonsai2-phase2.png"
fig.savefig(out, facecolor=BG)
print(out, solved, capped)
