#!/usr/bin/env python3
"""hermes-settings card (2026-10-27): left, multi-turn tool calls solved of 100 per setting (toolcall-v1 S5, Qwen3.8-27B Q6_K, greedy);
right, VRAM loaded by KV type x context from the fit leg, with decode speed at 120k depth as the one side-fact the bars cannot show.

Reads ONLY two files copied from the rig results/hermes-settings/ : paired-readout.txt and fit/summary.json.

    cp ~/benchmark-rig/results/hermes-settings/paired-readout.txt  <dir>/
    cp ~/benchmark-rig/results/hermes-settings/fit/summary.json    <dir>/fit/
    python3 scripts/chart_hermes_settings.py --results <dir> --out reports/hermes-settings.png [--draft]

--draft draws legs missing from the readout as hatched 'pending' bars and tags the card DRAFT. Without --draft the script refuses to
render while any of the three late legs (c64k, c128k, q4) is missing: no placeholder number ever reaches a final card.
"""
import argparse
import json
import os
import re
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, TEXT, MUTE, GRID = "#0e1420", "#eaf0f7", "#7d8ba0", "#20293a"
BASE, GOLD, BLUE, TEAL, WARN = "#6fd6a8", "#e8b64c", "#7aa7e8", "#59c3c3", "#ff6b6b"
REF = "qwen3-8-27b-q6-k-s5-on"
CARD_MIB = 32607


def read_pairs(path):
    pairs, cur = {}, None
    for ln in open(path):
        ln = ln.strip()
        m = re.match(r"### (\S+) vs (\S+)$", ln)
        if m:
            cur = (m.group(1), m.group(2))
        elif ln.startswith("{") and cur:
            pairs[cur] = json.loads(ln)
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draft", action="store_true")
    a = ap.parse_args()
    pairs = read_pairs(os.path.join(a.results, "paired-readout.txt"))
    fit = {c["label"]: c for c in json.load(open(os.path.join(a.results, "fit", "summary.json")))["cells"]}

    # (leg dir, label, colour, note builder)
    legs = [
        ("s5-on-f16-r2", "same run again (exact repeat)", MUTE),
        ("s5-off-f16", "thinking off", GOLD),
        ("s5-on-q8", "q8_0 KV cache", BLUE),
        ("s5-on-f16-c32k", "context 32,768", TEAL),
        ("s5-on-f16-c64k", "context 65,536", TEAL),
        ("s5-on-f16-c128k", "context 131,072", TEAL),
        ("s5-on-q4", "q4_0 KV cache", BLUE),
    ]
    late = ("s5-on-f16-c64k", "s5-on-f16-c128k", "s5-on-q4")
    missing = [l for l in late if (l, REF) not in pairs]
    if missing and not a.draft:
        sys.exit(f"refusing to render a final card: no paired result yet for {missing} (use --draft for a partial)")
    ref = pairs[("s5-on-f16-r2", REF)]
    base_correct = ref["base_correct"]
    assert base_correct == 81, base_correct
    rows = [("baseline: f16 KV, thinking on, context 40,960", base_correct, BASE, "reference", False)]
    for leg, lab, col in legs:
        p = pairs.get((leg, REF))
        if p is None:
            rows.append((lab, None, col, "pending", True))
            continue
        if leg == "s5-on-f16-r2":
            note = f"{p['identical_pct']:.0f}% of runs identical"
        else:
            note = f"{p['base_only_correct']} / {p['other_only_correct']} vs baseline, p = {p['mcnemar_exact_p']:.2f}".replace("1.00", "1.0")
            if leg == "s5-off-f16":
                note += f"\n{int(p['other_completion_tokens_mean'] + 0.5):,} tokens per task against {int(p['base_completion_tokens_mean'] + 0.5):,}"
        rows.append((lab, p["other_correct"], col, note, False))

    fig = plt.figure(figsize=(16, 7.6), dpi=110)
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 0.9], wspace=0.2, left=0.25, right=0.975, top=0.70, bottom=0.21)
    ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for x in (ax, bx):
        x.set_facecolor(BG)
        for sp in x.spines.values():
            sp.set_visible(False)
        x.tick_params(length=0, colors=MUTE)
    ax.grid(axis="x", color=GRID, lw=1, zorder=0)
    bx.grid(axis="y", color=GRID, lw=1, zorder=0)
    n = len(rows)
    for i, (lab, val, col, note, pend) in enumerate(rows):
        y = n - 1 - i
        ax.text(-2.0, y, lab, ha="right", va="center", color=TEXT, fontsize=11.8)
        if pend:
            ax.barh(y, base_correct, color="none", edgecolor=MUTE, hatch="///", lw=1, height=0.62, zorder=3)
            ax.text(base_correct / 2, y, "pending", ha="center", va="center", color=MUTE, fontsize=12, style="italic", zorder=4,
                    bbox=dict(facecolor=BG, edgecolor="none", pad=2))
            continue
        ax.barh(y, val, color=col, height=0.62, zorder=3)
        ax.text(val - 1.2, y, f"{val}", ha="right", va="center", color=BG, fontsize=17, weight="bold", zorder=4)
        ax.text(val + 1.0, y, note, ha="left", va="center", color=TEXT if i != 0 else MUTE, fontsize=10.5, linespacing=1.35)
    ax.set_xlim(0, 125)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", labelsize=10)
    ax.text(0, n - 0.1, "multi-turn tool calls: tasks solved of 100\n(BFCL v4 multi_turn_base, same 100 tasks every run)", ha="left", va="bottom",
            color=TEXT, fontsize=12.5)

    # right: VRAM grouped bars
    ctxs = [32768, 65536, 131072]
    kvs = [("f16", BASE, "f16 KV (default)"), ("q8_0", BLUE, "q8_0 KV"), ("q4_0", GOLD, "q4_0 KV")]
    w = 0.26
    for j, (kv, col, lab) in enumerate(kvs):
        xs, hs = [], []
        for k, c in enumerate(ctxs):
            xs.append(k + (j - 1) * w)
            hs.append(fit[f"{kv}-c{c}"]["vram_mib_loaded"] / 1024)
        bx.bar(xs, hs, width=w * 0.92, color=col, zorder=3, label=lab)
        for x, h in zip(xs, hs):
            bx.text(x, h + 0.25, f"{h:.1f}", ha="center", va="bottom", color=TEXT, fontsize=10.5)
    card = CARD_MIB / 1024
    bx.axhline(card, color=WARN, lw=1.6, ls="--", zorder=2)
    bx.text(2.52, card + 0.25, f"card: {card:.1f} GiB", ha="right", va="bottom", color=WARN, fontsize=11, weight="bold")
    bx.set_ylim(18, 34.5)
    bx.set_yticks([20, 25, 30])
    bx.tick_params(axis="y", labelsize=10)
    bx.set_xticks(range(3))
    bx.set_xticklabels([f"{c:,}" for c in ctxs], color=TEXT, fontsize=11)
    bx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), frameon=False, labelcolor=TEXT, fontsize=10.5, ncol=3)
    bx.text(-0.45, 34.9, "GPU memory loaded, by context length (GiB)\n(the Q6_K file alone is 21.3 GiB)", ha="left", va="bottom", color=TEXT, fontsize=12.5)
    d = {kv: fit[f"{kv}-c131072"]["requests"][4]["decode_tok_s"] for kv, _, _ in kvs}
    fig.text(0.535, 0.062, f"decode with a 120k prompt: f16 {d['f16']:.1f} · q8_0 {d['q8_0']:.1f} · q4_0 {d['q4_0']:.1f} tok/s",
             color=TEXT, fontsize=13, weight="bold")

    title = "Thinking off and a q8_0 KV cache moved multi-turn tool calls by 3 tasks of 100 or fewer"
    fig.text(0.02, 0.935, title, color=TEXT, fontsize=19, weight="bold")
    fig.text(0.02, 0.895,
             "Qwen3.8-27B Q6_K on one RTX 5090, llama.cpp b10371, greedy, one setting changed per run. Pairs: tasks only the baseline solved / only the setting solved, "
             "exact McNemar test.\nThe exact repeat reproduced all 100 tasks identically, so every difference here comes from the setting. "
             "The tasks never fill a long context (longest run ~17k tokens), so context rows test fit, not long-context quality.",
             color=MUTE, fontsize=10.6, va="top")
    fig.text(0.02, 0.025, "witcheer · RTX 5090 · llm-bench-rig hermes-settings (toolcall-v1 slice) · VRAM from a separate fit run, one request per depth · 2026-10-26 to 10-27",
             color=MUTE, fontsize=9.5)
    if a.draft:
        fig.text(0.975, 0.965, f"DRAFT · {len(missing)} legs pending" if missing else "DRAFT", ha="right", color=WARN, fontsize=12, weight="bold")
    fig.savefig(a.out, facecolor=BG)
    print(a.out, [(r[0], r[1]) for r in rows])


if __name__ == "__main__":
    main()
