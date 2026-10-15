import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, FG, MUT = "#0f1117", "#e8e8ea", "#9a9caa"
LFM, QW = "#7c5cff", "#3fb6a8"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": FG, "axes.labelcolor": FG,
                     "xtick.color": FG, "ytick.color": MUT})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 6.4), dpi=150, gridspec_kw={"width_ratios": [1.15, 1]})
fig.patch.set_facecolor(BG)
for a in (a1, a2):
    a.set_facecolor(BG)
    for s in ("top", "right", "left"):
        a.spines[s].set_visible(False)
    a.spines["bottom"].set_color("#333645")

# panel 1: scores
tasks = ["IFEval (strict)\n541 items", "GPQA Diamond\n198 items"]
lfm = [93.35, 68.69]
qw = [87.25, 76.77]
x = [0, 1]
w = 0.36
b1 = a1.bar([i - w / 2 for i in x], lfm, w, color=LFM, label="LFM2.5-24B-A2B  UD-Q5_K_XL")
b2 = a1.bar([i + w / 2 for i in x], qw, w, color=QW, label="Qwen3.6-35B-A3B  UD-Q5_K_M")
for bars in (b1, b2):
    for b in bars:
        a1.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f"{b.get_height():.2f}",
                ha="center", va="bottom", fontsize=13, fontweight="bold", color=FG)
a1.set_xticks(x)
a1.set_xticklabels(tasks, fontsize=12)
a1.set_ylim(0, 105)
a1.set_yticks([])
a1.set_title("score (%), same items", fontsize=14, color=FG, loc="left", pad=12)
a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=False, fontsize=11)

# panel 2: GPQA misses split
names = ["LFM2.5", "Qwen3.6"]
capped = [48, 13]
wrong = [14, 33]
y = [1, 0]
a2.barh(y, capped, 0.5, color="#5a5d6e", label="hit the 16k cap mid-thought")
a2.barh(y, wrong, 0.5, left=capped, color="#c9cbd6", label="finished and wrong")
for yi, c, wr in zip(y, capped, wrong):
    a2.text(c / 2, yi, str(c), ha="center", va="center", fontsize=13, fontweight="bold", color=FG)
    a2.text(c + wr / 2, yi, str(wr), ha="center", va="center", fontsize=13, fontweight="bold", color=BG)
    a2.text(c + wr + 1.5, yi, f"{c + wr} misses", ha="left", va="center", fontsize=12, color=MUT)
a2.set_yticks(y)
a2.set_yticklabels(names, fontsize=13)
a2.set_xlim(0, 78)
a2.set_xticks([])
a2.set_title("GPQA Diamond misses, by cause", fontsize=14, color=FG, loc="left", pad=12)
a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=False, fontsize=11)

fig.suptitle("LFM2.5-24B-A2B vs Qwen3.6-35B-A3B on one RTX 5090", fontsize=18, fontweight="bold",
             x=0.04, ha="left", y=0.98)
fig.text(0.04, 0.905, "thinking on, 16,384-token cap, zero-shot, greedy, llama.cpp", fontsize=12, color=MUT, ha="left")
fig.text(0.96, 0.02, "witcheer · RTX 5090 bench", fontsize=10, color=MUT, ha="right")
fig.subplots_adjust(left=0.04, right=0.97, top=0.82, bottom=0.26, wspace=0.18)
out = "reports/lfm25-24b-vs-qwen36-claims.png"
fig.savefig(out, facecolor=BG)
print(out)
