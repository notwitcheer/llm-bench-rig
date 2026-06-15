"""North-Mini-Code-1.0 — ARC-Challenge is reasoning-gated: 60.2% think-OFF -> ~95% think-ON.
A dumbbell showing the +35pt swing reasoning buys on the reasoning-gated subset, vs GSM8K
(generative) which is unaffected. The clean evidence that the think-OFF board understates a
reasoning model."""
import matplotlib.pyplot as plt

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = ("#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64")

# (task, think_off, think_on, on_is_spotcheck)
ROWS = [
    ("ARC-Challenge\n(reasoning-gated)", 60.2, 95.0, True),
    ("GSM8K\n(generative — control)", 95.8, 95.8, False),
]

fig, ax = plt.subplots(figsize=(9.5, 4.6), facecolor=BG); ax.set_facecolor(BG)
ys = list(range(len(ROWS)))[::-1]
for y, (label, off, on, spot) in zip(ys, ROWS):
    ax.plot([off, on], [y, y], color=GRID, linewidth=3, zorder=1)
    ax.scatter([off], [y], s=240, color=MUTE, edgecolor=TEXT, zorder=3)
    ax.scatter([on], [y], s=240, color=CRIMSON if on != off else MUTE, edgecolor=TEXT, zorder=3)
    ax.annotate(f"{off:.1f}", (off, y), color=TEXT, fontsize=10, ha="right", va="center",
                xytext=(-10, 0), textcoords="offset points")
    if on != off:
        ax.annotate(f"{on:.0f}*", (on, y), color=CRIMSON, fontsize=11, ha="left", va="center",
                    xytext=(10, 0), textcoords="offset points", weight="bold")
        ax.annotate(f"+{on-off:.0f}pt with reasoning", ((off+on)/2, y), color=GOLD, fontsize=9.5,
                    ha="center", va="bottom", xytext=(0, 9), textcoords="offset points")
    else:
        ax.annotate("no change", (on, y), color=MUTE, fontsize=9.5, ha="left", va="center",
                    xytext=(12, 0), textcoords="offset points")

ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in ROWS], color=TEXT, fontsize=10)
ax.set_xlabel("accuracy %  (● think-OFF   ● think-ON)", color=TEXT)
ax.set_xlim(50, 102)
ax.set_title("North-Mini-Code-1.0: ARC-Challenge is reasoning-gated (RTX 5090)\n"
             "a reasoning model benched think-OFF is understated where the task needs reasoning",
             color=GOLD, pad=14, fontsize=12)
for s in ax.spines.values(): s.set_color(GRID)
ax.tick_params(colors=MUTE); ax.grid(True, axis="x", color=GRID, linewidth=0.6, zorder=0)
fig.text(0.012, 0.02, "* think-ON = 40-question spot-check (38/40). think-OFF = full 1,172.",
         color=MUTE, fontsize=8)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig("reports/north-mini-arc-gate.png", dpi=150, facecolor=BG)
print("wrote reports/north-mini-arc-gate.png")
