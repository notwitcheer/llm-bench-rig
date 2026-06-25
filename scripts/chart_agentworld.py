"""EV+ card: Qwen-AgentWorld LWM zero-FT transfer, claimed vs measured (Gold & Crimson v2).
Left  - the anchor: real SWE-bench resolve as shared + model-unique solves (a reshuffle, net -2).
Right - claimed transfer band (+3.4..12.8%) vs measured delta on synthetic (0%) and real coding (-12.5%).
The claim says up; the rig says flat on the saturated synthetic axis and down on real coding."""
import json

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BG, GOLD, CRIMSON, TEXT, GRID, MUTE = ("#0d0906", "#e8c44a", "#e06060", "#f5e6d0", "#3a2f25", "#8a7a64")
d = json.load(open("results/agentworld/transfer.json"))

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.0, 6.0), facecolor=BG)
fig.suptitle("Qwen-AgentWorld's zero-fine-tune transfer: claimed +3-13%, measured flat (synthetic) and down (real coding)",
             color=GOLD, fontsize=12.5, fontweight="bold", y=0.975)

# ---------- LEFT: real anchor solve-sets (shared + unique) ----------
axL.set_facecolor(BG)
ss = d["swe_solve_sets"]
shared, bo, ao = ss["shared"], ss["base_only"], ss["agentworld_only"]
x = [0, 1]
axL.bar(x, [shared, shared], 0.62, color=GOLD, edgecolor=TEXT, zorder=3, label="solved by both")
axL.bar(0, bo, 0.62, bottom=shared, color=MUTE, edgecolor=TEXT, zorder=3, label="base-only")
axL.bar(1, ao, 0.62, bottom=shared, color=CRIMSON, edgecolor=TEXT, zorder=3, label="AgentWorld-only")
for xi, tot, emp in [(0, d["swe_resolve_30"]["base"], d["empty_patches"]["base"]),
                     (1, d["swe_resolve_30"]["agentworld"], d["empty_patches"]["agentworld"])]:
    axL.annotate(f"{tot}/30", (xi, tot), color=TEXT, ha="center", va="bottom",
                 fontsize=14, fontweight="bold", xytext=(0, 4), textcoords="offset points")
    axL.annotate(f"{emp} give-ups", (xi, 1.2), color=BG, ha="center", va="bottom", fontsize=9.5, fontweight="bold")
axL.set_xticks(x)
axL.set_xticklabels(["Qwen3.5-35B-A3B\n(base)", "Qwen-AgentWorld-35B-A3B\n(LWM-warmed)"], color=TEXT, fontsize=10)
axL.set_ylabel("SWE-bench Verified resolved  (of 30)", color=TEXT, fontsize=11)
axL.set_ylim(0, 19)
axL.set_title("the real anchor: a reshuffle, net -2 (synthetic board flat: 97.5 = 97.5)",
              color=MUTE, fontsize=9.5, pad=8)
axL.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=9, loc="upper right")
for s in axL.spines.values():
    s.set_color(GRID)
axL.tick_params(colors=MUTE)
axL.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)

# ---------- RIGHT: claimed band vs measured delta ----------
axR.set_facecolor(BG)
lo, hi = d["claimed_transfer_pct"]
axR.axhspan(lo, hi, color=GOLD, alpha=0.22, zorder=1)
axR.axhline(0, color=TEXT, ls=":", lw=1.0, zorder=2)
axR.annotate(f"claimed transfer\n+{lo} to +{hi}%", (0.5, (lo + hi) / 2), color=GOLD, fontsize=10,
             fontweight="bold", ha="center", va="center")
syn = 0.0
real = round(100 * (d["swe_resolve_30"]["agentworld"] - d["swe_resolve_30"]["base"]) / d["swe_resolve_30"]["base"], 1)
bars = axR.bar([0, 1], [syn, real], 0.5, color=[MUTE, CRIMSON], edgecolor=TEXT, zorder=3)
for xi, v, lab in [(0, syn, "0.0%"), (1, real, f"{real}%")]:
    axR.annotate(lab, (xi, v), color=TEXT, ha="center", va="top" if v < 0 else "bottom",
                 fontsize=12, fontweight="bold", xytext=(0, -6 if v < 0 else 4), textcoords="offset points")
axR.set_xticks([0, 1])
axR.set_xticklabels(["synthetic\nagentic board", "real SWE-bench\nresolve"], color=TEXT, fontsize=10)
axR.set_ylabel("measured change vs base  (%)", color=TEXT, fontsize=11)
axR.set_ylim(-18, 15)
axR.set_title("claim vs measured: the gain lands nowhere", color=MUTE, fontsize=9.5, pad=8)
for s in axR.spines.values():
    s.set_color(GRID)
axR.tick_params(colors=MUTE)
axR.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)

fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("reports/agentworld-lwm-transfer.png", dpi=140, facecolor=BG)
print("wrote reports/agentworld-lwm-transfer.png")
