#!/usr/bin/env python3
"""AA-style quadrant: quality (q_avg) vs generation speed on one RTX 5090.

Reusable template for the WITCHEER benchmark lane, in the Artificial-Analysis visual grammar:
- field of models colour-coded by family, log-speed x-axis
- shaded "fast and smart" attractive quadrant (top-right)
- optional subject highlight (ring + arrow) to feature one model within the pack

House palette (deep navy, per-family colours) — this is the lane's recognisable look, the way
purple is Artificial Analysis's. Do NOT mix think-on and think-off models on one axis: pick a
regime (they are not comparable). q_avg comes from dataset/README.md leaderboard; tok/s from
results/<slug>/speed.json (tg128). Never estimate a point — every dot traces to a real json.

Usage: edit DATA + TITLE/SUBTITLE/OUT below, then run via the mercury charts venv:
  /opt/data/venvs/charts/bin/python scripts/chart_quadrant.py
(box guard trips on shell-invoked venv python — call it from execute_code subprocess on mercury.)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import matplotlib.ticker as mticker

# ---- house palette ----
BG    = "#0e1420"   # deep navy
TEXT  = "#eaf0f7"
MUTE  = "#7d8ba0"
GRID  = "#20293a"
IDEAL = "#1f9e8a"   # teal wash, attractive quadrant
HILITE= "#ffffff"   # subject ring
C = {  # per-family colours
    "qwen":   "#4c9be8", "gemma":  "#e8b64c", "ling":   "#a86fe0",
    "nemo":   "#e0645f", "gptoss": "#57c78a", "coder":  "#e88fb8",
    "mistral":"#ff8c42", "muse":   "#e0645f", "other":  "#9aa7b8",
}

# ================= EDIT PER CHART =================
TITLE    = "Quality vs speed on one RTX 5090"
SUBTITLE = "12 local models, five-task quality average vs generation throughput  ·  thinking off"
OUT      = "/opt/data/mercury-cards/quadrant-5090-quality-vs-speed.png"
# ideal-quadrant box: (x_left, y_bottom, x_right, y_top) in data coords, or None to omit
IDEAL_BOX = (150, 92, 470, 96)
XLIM = (40, 470); YLIM = (75, 96); XTICKS = [50, 100, 200, 400]
LEGEND = [("Qwen","qwen"),("Gemma","gemma"),("Ling","ling"),
          ("Nemotron","nemo"),("gpt-oss","gptoss"),("Coder","coder")]
# (label, q_avg, tok_s, family, (dx,dy,ha), show_label, subject?)
DATA = [
    ("Qwen3.6-35B-A3B",   93.3, 270.6, "qwen",  (0, 13, "center"), True,  False),
    ("Gemma 4 31B",       94.2,  55.4, "gemma", (-10, 2, "right"), True,  False),
    ("Qwen3.6-27B dense", 94.2,  61.9, "qwen",  (10, 11, "left"),  True,  False),
    ("Qwopus-27B-Coder",  94.1,  70.5, "coder", (11, 1, "left"),   True,  False),
    ("Qwable-5-27B",      93.7,  62.9, "coder", (2, -17, "center"),False, False),
    ("Qwen3-Coder-Next",  91.7, 224.7, "qwen",  (0, 13, "center"), True,  False),
    ("Ling-3.0-flash",    91.8,  46.0, "ling",  (-10, 0, "right"), True,  False),
    ("Gemma 4 12B",       87.6, 122.3, "gemma", (0, -18, "center"),True,  False),
    ("gpt-oss-20B",       87.4, 367.4, "gptoss",(0, 13, "center"), True,  False),
    ("Nemotron-3-Nano",   82.2, 369.6, "nemo",  (10, 6, "left"),   True,  False),
    ("Nemotron-Cascade-2",81.6, 350.8, "nemo",  (-10, -4, "right"),True,  False),
    ("North-Mini-Code",   77.4, 304.8, "other", (0, 12, "center"), True,  False),
]
XLABEL = "generation speed  (tokens/sec, log scale)"
YLABEL = "quality  (5-task average, %)"
# =================================================

def render():
    fig, ax = plt.subplots(figsize=(12, 7.2), dpi=110)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    fig.subplots_adjust(left=0.075, right=0.965, top=0.80, bottom=0.115)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xscale("log")
    ax.tick_params(length=0, labelsize=12, colors=MUTE)
    ax.grid(color=GRID, linewidth=1, alpha=0.7); ax.set_axisbelow(True)

    if IDEAL_BOX:
        xl, yb, xr, yt = IDEAL_BOX
        ax.add_patch(Rectangle((xl, yb), xr-xl, yt-yb, color=IDEAL, alpha=0.10, zorder=1))
        ax.text((xl*xr)**0.5, yb+0.4, "fast  and  smart", color=IDEAL, fontsize=11.5,
                fontweight="bold", ha="center", va="bottom", alpha=0.95)

    for label, q, ts, fam, off, show, subject in DATA:
        c = C.get(fam, C["other"])
        if subject:
            ax.scatter(ts, q, s=520, facecolors="none", edgecolors=HILITE,
                       linewidths=2.2, zorder=6)
        ax.scatter(ts, q, s=320, color=c, zorder=5, edgecolors=BG, linewidths=1.6)
        if show:
            dx, dy, ha = off
            ax.annotate(label, (ts, q), xytext=(dx, dy), textcoords="offset points",
                        ha=ha, va="center", color=TEXT, fontsize=11.5,
                        fontweight=("bold" if subject else "normal"))

    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_xticks(XTICKS); ax.set_xticks([], minor=True)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xticklabels([str(t) for t in XTICKS])
    ax.set_xlabel(XLABEL, fontsize=13, color=TEXT)
    ax.set_ylabel(YLABEL, fontsize=13, color=TEXT)

    fig.text(0.075, 0.925, TITLE, fontsize=25, fontweight="bold", color=TEXT)
    fig.text(0.075, 0.865, SUBTITLE, fontsize=13, color=MUTE)
    fig.text(0.965, 0.037, "WITCHEER  ·  RTX 5090 benchmarks", fontsize=11.5, color=MUTE, ha="right")

    x0 = 0.075
    for i,(name,key) in enumerate(LEGEND):
        fig.text(x0 + i*0.085 + 0.012, 0.045, name, fontsize=10, color=MUTE, va="center")
        fig.add_artist(Circle((x0 + i*0.085, 0.047), 0.006, color=C[key],
                       transform=fig.transFigure))

    fig.savefig(OUT, facecolor=BG)
    print("wrote", OUT)

if __name__ == "__main__":
    render()
