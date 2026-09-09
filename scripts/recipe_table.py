#!/usr/bin/env python3
"""Render the six-set table for a recipe page from results/<slug>/recipe/recipe.json.

Prints markdown to stdout. The page prose (the line, the reads) stays hand-written; only
the table is generated so numbers can never drift from the raw summaries.

Usage: python scripts/recipe_table.py results/<slug>/recipe/recipe.json
"""
from __future__ import annotations

import json
import sys

LABEL = {
    "a_base": "base (`--jinja -ngl 99 -c 32768`)",
    "b_fa": "+ `--flash-attn on`",
    "c_fa_q8kv": "+ fa + KV `q8_0/q8_0`",
    "d_fa_q4kv": "+ fa + KV `q4_0/q4_0`",
    "e_fa_np1": "+ fa + `--parallel 1`",
    "f_fa_mtp": "+ fa + MTP `--spec-type draft-mtp --spec-draft-n-max 2`",
}


def table(d: dict) -> str:
    out = ["| flag set | code tok/s | prose tok/s | 16k cold TTFT | 16k warm TTFT | tok/s on 16k prefix | VRAM peak | GPQA 40-item | MTP accept |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in d["sets"]:
        sh, g = r["short"], r["gpqa_spot"]
        acc, accp = sh["code"].get("acceptance_rate"), sh["prose"].get("acceptance_rate")
        acc_s = f"{acc:.2f} code / {accp:.2f} prose" if acc else ""
        gq = f"{g['correct']}/{g['total']}" if g else ""
        out.append(f"| {LABEL.get(r['set'], r['set'])} | {sh['code']['perceived_tps_p50']:.1f} | {sh['prose']['perceived_tps_p50']:.1f} | "
                   f"{r['long_cold']['ttft_p50_s']:.2f} s | {r['long_warm']['ttft_p50_s']:.3f} s | {r['long_warm']['perceived_tps_p50']:.1f} | "
                   f"{r['vram_peak_mib'] / 1024:.1f} GiB | {gq} | {acc_s} |")
    return "\n".join(out)


if __name__ == "__main__":
    print(table(json.load(open(sys.argv[1]))))
