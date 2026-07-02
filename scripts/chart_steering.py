"""t074 Bias-Only Steering: claimed vs measured (Gold & Crimson v2), two panels.
Top: dumbbell per benchmark — their Table-1 trajectory (base -> steering, full-FT tick)
against ours at the bounded budget (base -> steering, with LoRA + random-reward marks).
Bottom: where RLOO wall-clock actually goes (rollout/grade/update shares) — the "34s
vs 52m" efficiency claim counts ONLY the update slice; rollouts dominate and cost the
same no matter how few parameters train.
Reads results/steering/summary.json (aggregate_steering.py). Run (Mac):
  venv/bin/python scripts/chart_steering.py
"""
import json

import matplotlib.pyplot as plt

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = ("#0d0906", "#e8c44a", "#e06060", "#f5e6d0",
                                       "#3a2f25", "#8a7a64")
CLAIM = {"math500": {"base": 52.2, "steer": 79.9, "fullft": 79.3},
         "amc23": {"base": 45.8, "steer": 62.5, "fullft": 64.2}}   # their Table 1
NICE = {"math500": "MATH500 (500q)", "amc23": "AMC23 (40q)"}

S = json.load(open("results/steering/summary.json"))
table, timing = S["table"], S["timing"]


def find(labels, prefix):
    for l in labels:
        if l.startswith(prefix):
            return l
    return None


fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 8.2), facecolor=BG,
                              gridspec_kw={"height_ratios": [3.2, 1]})
for a in (ax, ax2):
    a.set_facecolor(BG)
    for s in a.spines.values():
        s.set_color(GRID)
    a.tick_params(colors=TEXT)
fig.suptitle("Bias-only steering on Qwen2.5-Math-7B: claimed vs measured on one RTX 5090",
             color=GOLD, fontsize=14.5, fontweight="bold", y=0.98)

rows, ynames = [], []
for bench in [b for b in ("math500", "amc23") if b in table]:
    labels = table[bench]
    steer = find(labels, "steer-b")
    lora = find(labels, "lora-b")
    rand = find(labels, "steer-rand")
    rows.append(("ours", bench, labels["base"]["acc"],
                 labels[steer]["acc"] if steer else None,
                 labels[lora]["acc"] if lora else None,
                 labels[rand]["acc"] if rand else None))
    ynames.append(f"{NICE[bench]}\nours (bounded budget)")
    c = CLAIM[bench]
    rows.append(("claim", bench, c["base"], c["steer"], None, None))
    ynames.append(f"{NICE[bench]}\ntheir full recipe (claim)")

for y, (kind, bench, base, steer, lora, rand) in enumerate(rows):
    ax.plot([base, steer], [y, y], color=GRID, lw=3, zorder=2)
    ax.scatter([base], [y], s=110, color=MUTE, zorder=3,
               label="base" if y == 0 else None)
    style = dict(s=130, zorder=4)
    if kind == "claim":
        ax.scatter([steer], [y], facecolors=BG, edgecolors=CRIMSON, lw=2.2,
                   label="steering (their claim)" if kind == "claim" and y <= 1 else None,
                   **style)
        ax.scatter([CLAIM[bench]["fullft"]], [y], marker="D", s=70, facecolors=BG,
                   edgecolors=GOLD, lw=1.8, zorder=4,
                   label="full-FT (their claim)" if y <= 1 else None)
    else:
        ax.scatter([steer], [y], color=CRIMSON,
                   label="steering (ours, measured)" if y == 0 else None, **style)
        if lora is not None:
            ax.scatter([lora], [y], color=GOLD, s=90, zorder=4,
                       label="LoRA r4 down_proj (ours)" if y == 0 else None)
        if rand is not None:
            ax.scatter([rand], [y], color=TEXT, marker="x", s=90, lw=2.4, zorder=5,
                       label="random-reward steering (ours)" if y == 0 else None)

ours = [y for y, r in enumerate(rows) if r[0] == "ours"]
if ours:
    y0 = ours[0]
    base0, steer0 = rows[y0][2], rows[y0][3]
    ax.annotate("20 matched steps: steering = LoRA = random-reward = base",
                xy=(max(base0, steer0), y0), xytext=(max(base0, steer0) + 1.5, y0 + 0.45),
                color=TEXT, fontsize=9, arrowprops=dict(arrowstyle="-", color=GRID))

ax.set_yticks(range(len(rows)))
ax.set_yticklabels(ynames, color=TEXT, fontsize=9.5)
ax.invert_yaxis()
ax.set_xlabel("accuracy (%)", color=TEXT)
ax.grid(axis="x", color=GRID, lw=0.6, zorder=1)
ax.legend(loc="lower right", facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=8.5)

t = list(timing.values())[0] if timing else None
if t:
    left = 0.0
    for key, col, name in (("rollout_share", CRIMSON, "rollout (generation)"),
                           ("grade_share", MUTE, "reward grading"),
                           ("update_share", GOLD, "weight update")):
        ax2.barh([0], [t[key] * 100], left=left, color=col, height=0.5, zorder=3)
        if t[key] > 0.04:
            ax2.text(left + t[key] * 50, 0, f"{name}\n{t[key]:.0%}", ha="center",
                     va="center", color=BG if key != "grade_share" else TEXT,
                     fontsize=8.5, fontweight="bold", zorder=4)
        left += t[key] * 100
    ax2.set_xlim(0, 100)
    ax2.set_yticks([])
    ax2.set_xlabel("share of RLOO training wall-clock per step (%) — one RTX 5090",
                   color=TEXT, fontsize=9.5)
    ax2.set_title('the "34s" headline counts only the optimizer sliver inside the gold '
                  "slice (which is mostly backward); neither it nor the rollouts shrink "
                  "when 245K params train instead of 7.6B",
                  color=TEXT, fontsize=9.5, pad=8)

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("reports/steering-claimed-vs-measured.png", dpi=170, facecolor=BG)
print("wrote reports/steering-claimed-vs-measured.png")
