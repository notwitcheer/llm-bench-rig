#!/usr/bin/env python3
"""AA-style ranked field: a sorted bar chart of a single metric across the model field.

The Artificial-Analysis "intelligence index" bar, in the WITCHEER house palette. Sorted
descending, value printed inside each bar, per-family colour, optional subject highlight (bright
ring/edge). Use for q_avg leaderboards, single-task rankings, tok/s rankings, etc.

Do NOT mix think regimes on one chart. Values trace to dataset/README.md (q_avg) or
results/<slug>/speed.json (tok/s) — never estimate. Run via the mercury charts venv from an
execute_code subprocess (box guard trips on shell-invoked venv python).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ---- house palette (shared with chart_quadrant.py) ----
BG   = "#0e1420"; TEXT = "#eaf0f7"; MUTE = "#7d8ba0"; GRID = "#20293a"; HILITE = "#ffffff"
C = {
    "qwen":  "#4c9be8", "gemma": "#e8b64c", "ling":  "#a86fe0", "nemo": "#e0645f",
    "gptoss":"#57c78a", "coder": "#e88fb8", "mistral":"#ff8c42", "muse":"#e0645f", "other":"#9aa7b8",
}

# ================= EDIT PER CHART =================
TITLE    = "Local model quality on one RTX 5090"
SUBTITLE = "five-task average (MMLU, ARC-C, HellaSwag, GSM8K, HumanEval)  ·  thinking off"
OUT      = "/opt/data/mercury-cards/ranked-5090-quality.png"
VALUE_FMT = "{:.1f}"          # printed inside each bar
XMAX      = 100
LEGEND = [("Qwen","qwen"),("Gemma","gemma"),("Ling","ling"),
          ("Nemotron","nemo"),("gpt-oss","gptoss"),("Coder","coder")]
# (label, value, family, subject?)  — order irrelevant, sorted descending on render
DATA = [
    ("Gemma 4 31B",         94.2, "gemma", False),
    ("Qwen3.6-27B dense",   94.2, "qwen",  False),
    ("Qwopus3.6-27B-Coder", 94.1, "coder", False),
    ("Qwen3.6-35B-A3B",     93.3, "qwen",  False),
    ("Ling-3.0-flash",      91.8, "ling",  False),
    ("Qwen3-Coder-Next",    91.7, "qwen",  False),
    ("Gemma 4 12B",         87.6, "gemma", False),
    ("gpt-oss-20B",         87.4, "gptoss",False),
    ("Nemotron-3-Nano",     82.2, "nemo",  False),
    ("Nemotron-Cascade-2",  81.6, "nemo",  False),
    ("North-Mini-Code",     77.4, "other", False),
]
XLABEL = "five-task quality average (%)"
# =================================================

def render():
    rows = sorted(DATA, key=lambda d: d[1], reverse=True)
    n = len(rows)
    fig, ax = plt.subplots(figsize=(12, 0.52*n + 2.4), dpi=110)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    fig.subplots_adjust(left=0.28, right=0.965, top=1-2.0/(0.52*n+2.4), bottom=1.1/(0.52*n+2.4))
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0, labelsize=12.5, colors=TEXT)
    ax.grid(color=GRID, linewidth=1, alpha=0.6, axis="x"); ax.set_axisbelow(True)

    ys = list(range(n))[::-1]
    for y, (label, val, fam, subject) in zip(ys, rows):
        c = C.get(fam, C["other"])
        ax.barh(y, val, height=0.66, color=c, zorder=3,
                edgecolor=(HILITE if subject else BG), linewidth=(2.2 if subject else 1.5))
        ax.text(val - 1.2, y, VALUE_FMT.format(val), va="center", ha="right",
                color=BG, fontsize=12, fontweight="bold", zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=12.5)
    for tick, r in zip(ax.get_yticklabels(), rows):
        tick.set_color(TEXT if r[3] else MUTE)
        if r[3]: tick.set_fontweight("bold")
    ax.set_xlim(0, XMAX)
    ax.set_xlabel(XLABEL, fontsize=13, color=TEXT)

    H = 0.52*n + 2.4
    fig.text(0.28, 1-0.9/H, TITLE, fontsize=23, fontweight="bold", color=TEXT)
    fig.text(0.28, 1-1.45/H, SUBTITLE, fontsize=12.5, color=MUTE)
    fig.text(0.965, 0.4/H, "WITCHEER  ·  RTX 5090 benchmarks", fontsize=11.5, color=MUTE, ha="right")

    x0 = 0.28
    for i,(name,key) in enumerate(LEGEND):
        fig.text(x0 + i*0.08 + 0.012, 0.42/H, name, fontsize=9.5, color=MUTE, va="center")
        fig.add_artist(Circle((x0 + i*0.08, 0.44/H), 0.006, color=C[key], transform=fig.transFigure))

    fig.savefig(OUT, facecolor=BG)
    print("wrote", OUT)

if __name__ == "__main__":
    render()
