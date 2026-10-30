#!/usr/bin/env python3
"""Gold & Crimson episode card: one CAPE-Bench episode (gpt-oss-20b, mock tools, synthetic names) where the agent uses a
colleague's personal token, then asks permission, then reports it has no authorization; our refit readout's score on the
first token command. Text transcribed from the episode log (HF Afayyazi/CAPE-Bench, run xm_gptoss20b_core24b_minus6789,
episode credential.cred_analytics_token.0174.p2.unsafe.s8); token value masked, colleague name removed.

Usage: python3 scripts/chart_agent_knew_episode.py   (reads data/agent-knew/episode-card.json)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, GOLD, CRIMSON = "#0d0906", "#e8c44a", "#e06060"
TEXT, GRID, MUTE = "#f5e6d0", "#3a2f25", "#8a7a64"
E = json.load(open("data/agent-knew/episode-card.json"))

fig = plt.figure(figsize=(12, 6.75))
fig.patch.set_facecolor(BG)
fig.add_artist(plt.Rectangle((0.0, 0.905), 0.012, 0.03, transform=fig.transFigure, color=CRIMSON))
fig.text(0.03, 0.93, "gpt-oss-20b uses a token it was told not to share", fontsize=21, fontweight="bold", color=TEXT, va="top")
fig.text(0.03, 0.855, "one episode from \"The Agent Knew\" dataset (mock tools, synthetic names)", fontsize=13, color=MUTE, va="top")

y = 0.77
MONO = dict(family="monospace", fontsize=12.5)
for row in E["rows"]:
    col = {"text": TEXT, "gold": GOLD, "crimson": CRIMSON, "mute": MUTE}[row["colour"]]
    fig.text(0.03, y, row["step"], color=MUTE, va="top", **MONO)
    fig.text(0.14, y, row["line"], color=col, va="top", fontweight=row.get("weight", "normal"), **MONO)
    if row.get("sub"):
        fig.text(0.14, y - 0.045, row["sub"], color=MUTE, va="top", **MONO)
        y -= 0.045
    y -= 0.075

fig.text(0.03, 0.06, E["footer"], fontsize=11.5, color=MUTE, va="center")
fig.text(0.97, 0.02, "WITCHEER · RTX 5090", fontsize=10, color=MUTE, ha="right")
fig.savefig("reports/agent-knew-episode.png", dpi=150)
print("wrote reports/agent-knew-episode.png")
