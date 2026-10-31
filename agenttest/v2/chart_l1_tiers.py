#!/usr/bin/env python3
"""chart_l1_tiers.py: the per-tier card "which model to run Hermes Agent on at N GB" from l1-tier-table.json (aggregate.py output).
five panels (8 / 12 / 16 / 24 / 32 GB): every finished row that FITS the card at the measured 64k peak, bars = tasks passed of 30, the pick (lead of 3+ tasks) in the accent colour, ties in grey-blue,
label = model + quant, peak GiB, minutes for the 30 tasks. numbers come only from the table; nothing is typed in. run with any python that has matplotlib.
usage: chart_l1_tiers.py --table l1-tier-table.json --out card.png [--watermark TEXT] [--hide-offload] [--mark SLUG=TEXT]
--hide-offload: rows whose label says "expert offload" are left out of the panels and named in a footnote (they need VRAM plus system RAM, so they are not a plain "fits N GB" file).
--mark SLUG=TEXT: adds a * to that row's label and TEXT to the footnote (e.g. a peak that a sibling process contaminated; the table value is still the one drawn)."""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CARDS = (8, 12, 16, 24, 32)
ap = argparse.ArgumentParser()
ap.add_argument("--table", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--watermark", default="")
ap.add_argument("--hide-offload", action="store_true")
ap.add_argument("--mark", action="append", default=[])
a = ap.parse_args()
rows = [r for r in json.load(open(a.table))["rows"] if not r.get("nofit")]
marks = dict(m.split("=", 1) for m in a.mark)
hidden = [r for r in rows if a.hide_offload and "offload" in r["label"]]
rows = [r for r in rows if r not in hidden]
fig, axes = plt.subplots(1, 5, figsize=(19, 6.2), sharex=True)
fig.patch.set_facecolor("#14161c")
for ax, c in zip(axes, CARDS):
    ax.set_facecolor("#14161c")
    fit = sorted([r for r in rows if r["tier"] is not None and r["tier"] <= c], key=lambda r: (-r["passed"], r["vram_peak_mib"]))[:6]
    ax.set_title(f"{c} GB card", color="white", fontsize=15, fontweight="bold")
    ax.set_xlim(0, 30)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors="#9aa0ab")
    if not fit:
        ax.text(15, 0.5, "nothing finished fits", color="#9aa0ab", ha="center")
        ax.set_yticks([])
        continue
    top = fit[0]["passed"]
    pick = len(fit) == 1 or top - fit[1]["passed"] >= 3
    for i, r in enumerate(fit):
        col = "#f5a623" if (i == 0 and pick) else ("#6c8ebf" if top - r["passed"] < 3 else "#4a4f5a")
        ax.barh(i, r["passed"], color=col, height=0.62)
        ax.text(r["passed"] + 0.4, i, f"{r['passed']}", color="white", va="center", fontsize=11, fontweight="bold")
        ax.text(0.4, i - 0.46, f"{r['label']}{'*' if r['slug'] in marks else ''}  |  {r['vram_peak_mib'] / 1024:.1f} GiB  |  {r['wall_s_total'] / 60:.0f} min", color="#d7dae0", fontsize=7.5, va="center")
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlabel("tasks passed of 30", color="#9aa0ab")
    ax.text(0.5, -0.17, "PICK" if pick else "TIE: no pick from one run", transform=ax.transAxes, ha="center", color="#f5a623" if pick else "#6c8ebf", fontsize=11, fontweight="bold")
fig.suptitle("Hermes Agent on a local model: the measured pick per VRAM tier", color="white", fontsize=19, fontweight="bold", y=0.99)
notes = [f"* {t}" for t in marks.values()] + [f"{r['label']}: {r['passed']} of 30 (needs {r['vram_peak_mib'] / 1024:.1f} GiB VRAM plus system RAM for the experts), not drawn as a plain {r['tier']} GB file." for r in hidden]
if notes:
    fig.text(0.5, 0.045, "   ".join(notes), color="#9aa0ab", ha="center", fontsize=8.5)
fig.text(0.5, 0.01, "30 own offline tasks, one run per row, peak VRAM measured at -c 65536 (f16 KV) on an RTX 5090 as a stand-in, llama.cpp, Hermes Agent as shipped. Not a public benchmark.", color="#9aa0ab", ha="center", fontsize=9)
if a.watermark:
    fig.text(0.5, 0.5, a.watermark, color="#ff3b30", alpha=0.25, ha="center", va="center", fontsize=60, rotation=20)
plt.tight_layout(rect=(0, 0.07, 1, 0.95))
fig.savefig(a.out, dpi=110, facecolor=fig.get_facecolor())
print("wrote", a.out)
