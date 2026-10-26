#!/usr/bin/env python3
"""Ornith 1.5 122B-A10B (UD-IQ3_XXS, experts of 24 layers in RAM) against the Ornith 1.5 35B-A3B (Q4_K_M, resident), one RTX 5090.

Left: decode speed (llama-bench tg128 at empty context and at 32k depth). Right: score per task with the Wilson 95% interval, the five-task
board plus GPQA Diamond think-off and think-on. Everything is read from the repo's own csvs:

    dataset/benchmarks.csv       tg128 / pp512 rows
    dataset/speed_at_depth.csv   tg128 at 32k depth, TTFT estimate
    dataset/board_ci.csv         score + Wilson interval per task

    python3 scripts/chart_ornith_122b.py [out.png]   # matplotlib; the rig's charts venv
"""
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "..", "dataset")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "reports", "ornith-1-5-122b-ud-iq3-xxs.png")
BG, TEXT, MUTE, GRID = "#0e1420", "#eaf0f7", "#7d8ba0", "#20293a"
SUBJ, REF = "#e8b64c", "#6fd6a8"

S122, S35, S35ON = "ornith-1-5-122b-ud-iq3-xxs", "ornith-1-5-35b-q4-k-m", "ornith-35b-thinkon"
S122ON = "ornith-1-5-122b-ud-iq3-xxs-thinkon"


def rows(name):
    return list(csv.DictReader(open(os.path.join(DS, name), newline="")))


bench = rows("benchmarks.csv")


def tps(model, test, quant):
    r = [x for x in bench if x["model"] == model and x["quant"] == quant and x["test"] == test]
    assert len(r) == 1, (model, test, len(r))
    return float(r[0]["tokens_per_sec"])


depth = {r["slug"]: r for r in rows("speed_at_depth.csv")}
ci = {(r["slug"], r["task"]): r for r in rows("board_ci.csv")}
M122, M35 = "Ornith 1.5 122B-A10B", "Ornith 1.5 35B-A3B"
tg0 = (tps(M35, "tg128", "Q4_K_M"), tps(M122, "tg128", "UD-IQ3_XXS"))
tg32 = (float(depth[S35]["tg128_d32768"]), float(depth[S122]["tg128_d32768"]))
pp = (tps(M35, "pp512", "Q4_K_M"), tps(M122, "pp512", "UD-IQ3_XXS"))
ttft = (float(depth[S35]["ttft_16k_est_s"]), float(depth[S122]["ttft_16k_est_s"]))

TASKS = [  # label, (slug35, task35), (slug122, task122)
    ("MMLU", (S35, "mmlu"), (S122, "mmlu")),
    ("ARC-Challenge", (S35, "arc_challenge"), (S122, "arc_challenge")),
    ("HellaSwag", (S35, "hellaswag"), (S122, "hellaswag")),
    ("GSM8K", (S35, "gsm8k"), (S122, "gsm8k")),
    ("HumanEval", (S35, "humaneval"), (S122, "humaneval")),
    ("GPQA Diamond, thinking off", (S35, "gpqa"), (S122, "gpqa")),
    ("GPQA Diamond, thinking on", (S35ON, "gpqa_thinkon"), (S122ON, "gpqa_thinkon")),
]


def get(key):
    r = ci[key]
    return float(r["score"]), float(r["ci_low"]), float(r["ci_high"])


fig = plt.figure(figsize=(16, 8.2), dpi=110)
fig.patch.set_facecolor(BG)
gs = fig.add_gridspec(1, 2, width_ratios=[0.78, 1.22], wspace=0.30, left=0.045, right=0.975, top=0.675, bottom=0.17)
ax = fig.add_subplot(gs[0])
bx = fig.add_subplot(gs[1])
for a in (ax, bx):
    a.set_facecolor(BG)
    for sp in a.spines.values():
        sp.set_visible(False)
    a.tick_params(length=0, colors=MUTE)
    a.grid(axis="x", color=GRID, lw=1, zorder=0)

# ---- left: decode speed
groups = [("decode, empty context", tg0), ("decode, 32k tokens deep", tg32)]
for i, (lab, (v35, v122)) in enumerate(groups):
    y = 1.6 * (1 - i)
    ax.barh(y + 0.19, v35, height=0.32, color=REF, zorder=3)
    ax.barh(y - 0.19, v122, height=0.32, color=SUBJ, zorder=3)
    ax.text(v35 + 6, y + 0.19, f"{v35:.1f}", ha="left", va="center", color=TEXT, fontsize=14, weight="bold")
    ax.text(v122 + 6, y - 0.19, f"{v122:.1f}", ha="left", va="center", color=TEXT, fontsize=14, weight="bold")
    ax.text(0, y + 0.50, lab, ha="left", va="bottom", color=TEXT, fontsize=12.5)
    ax.text(375, y - 0.19, f"{v35 / v122:.1f}x slower", ha="right", va="center", color=MUTE, fontsize=11.5)
ax.set_xlim(0, 380)
ax.set_ylim(-1.95, 2.55)
ax.set_yticks([])
ax.set_xticks([0, 100, 200, 300])
ax.tick_params(axis="x", labelsize=10)
ax.text(0, 2.62, "tokens per second (llama-bench tg128)", ha="left", va="bottom", color=TEXT, fontsize=12.5, weight="bold")
ax.scatter([], [], s=120, color=REF, label="35B-A3B  Q4_K_M, resident, 20.2 GiB")
ax.scatter([], [], s=120, color=SUBJ, label="122B-A10B  UD-IQ3_XXS, 24 layers of experts in RAM, 46.1 GiB")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.075), frameon=False, labelcolor=TEXT, fontsize=10.5, handletextpad=0.3,
          labelspacing=0.5)
ax.text(0, -0.93, f"Prompt processing is the wider gap: {pp[1]:,.0f} against {pp[0]:,.0f} tok/s (pp512),\n"
        f"so a 16k prompt is ~{ttft[1]:.0f} s to first token against ~{ttft[0]:.0f} s (derived).",
        color=TEXT, fontsize=12.2, weight="bold", va="top", ha="left")

# ---- right: scores with Wilson 95% intervals
n = len(TASKS)
SP = 1.2
for i, (lab, k35, k122) in enumerate(TASKS):
    y = (n - 1 - i) * SP + (0 if i < 5 else -0.45)
    s35, l35, h35 = get(k35)
    s122, l122, h122 = get(k122)
    for yy, s, lo, hi, col in ((y + 0.15, s35, l35, h35, REF), (y - 0.15, s122, l122, h122, SUBJ)):
        bx.plot([lo, hi], [yy, yy], color=col, lw=2.6, alpha=0.55, zorder=2, solid_capstyle="round")
        bx.scatter([s], [yy], s=150, color=col, edgecolor=BG, lw=1.4, zorder=4)
    d = s122 - s35
    clear = l122 > h35 or l35 > h122  # intervals do not touch
    bx.text(105.0, y, f"{d:+.1f}".replace("-", "\u2212"), ha="right", va="center", color=(TEXT if clear else MUTE),
            fontsize=13.5 if clear else 12, weight="bold" if clear else "normal")
    bx.text(40.6, y + 0.50, lab, ha="left", va="center", color=TEXT, fontsize=11.6)
bx.axhline((n - 5) * SP - 0.62, color=GRID, lw=1.2)
bx.set_xlim(40, 105.5)
bx.set_ylim(-1.25, n * SP - 0.05)
bx.set_yticks([])
bx.set_xticks([40, 50, 60, 70, 80, 90, 100])
bx.tick_params(axis="x", labelsize=10)
bx.text(40, n * SP - 0.05, "score, % (whiskers: Wilson 95% interval)", ha="left", va="bottom", color=TEXT, fontsize=12.5, weight="bold")
bx.text(105.0, n * SP - 0.05, "122B minus 35B", ha="right", va="bottom", color=MUTE, fontsize=11)
bx.text(105.0, -1.05, "bold = the two intervals do not touch", ha="right", va="center", color=MUTE, fontsize=10.2)

fig.text(0.02, 0.945, "Ornith 1.5 122B runs on one RTX 5090 at 47.6 tok/s. Only MMLU puts it clearly ahead of the 35B",
         color=TEXT, fontsize=19, weight="bold")
fig.text(0.02, 0.905, "Same model family and harness, greedy decoding, one run per row. The two models' intervals overlap on four of the "
         "five board tasks and on GPQA think-on\n(83.3 vs 81.8: 13 items one way, 10 the other, p = 0.68). GPQA Diamond is 198 items, so a gap under ~3 points is noise; "
         "think-off the 35B reads 4.6 points higher (p = 0.34 paired).\n"
         "Scores are greedy; Ornith's model card samples at temperature 0.6, top-k 20, top-p 0.95, so this is not the card's setting. "
         "Think-on: 16k budget, 21% (122B) and 22% (35B) of GPQA chains hit the cap.",
         color=MUTE, fontsize=10.6, va="top")
fig.text(0.02, 0.025, "witcheer \u00b7 RTX 5090 32 GB + 59 GB RAM \u00b7 llama.cpp b10371 \u00b7 llm-bench-rig dataset/board_ci.csv + reports/ornith-1-5-122b-ud-iq3-xxs.md "
         "\u00b7 all legs run 2026-10-25", color=MUTE, fontsize=9.5)
fig.savefig(OUT, facecolor=BG)
print(OUT, tg0, tg32, pp, ttft)
