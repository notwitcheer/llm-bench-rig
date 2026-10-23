#!/usr/bin/env python3
"""Summarise ~/humming-bench legs (pulled to a local dir): per-leg numbers + outputs-match per pair."""
import json, sys, os, difflib
D = sys.argv[1]
PAIRS = [("awq_marlin", "awq_humming"), ("moe_marlin", "moe_humming"), ("nv_cutlass", "nv_humming"),
         ("moe_marlin", "moe_auto"), ("awq_marlin", "awq_marlin2"), ("awq_humming", "awq_humming2"),
         ("moe_humming", "moe_humming2"), ("nv_cutlass", "nv_cutlass2"), ("nv_humming", "nv_humming2"),
         ("nv_cutlass", "nv_marlin"), ("nv_marlin", "nv_humming"), ("moe_humming", "h18_moe_humming"),
         ("nv_cutlass", "nvq_humming"), ("nv_cutlass", "nvq_marlin"), ("nv_humming", "nvq_humming")]
legs = {}
for f in sorted(os.listdir(D)):
    if f.endswith(".json") and not f.startswith(("SUMMARY", "apps")) and ".prefill" not in f:
        try:
            legs[f[:-5]] = json.load(open(os.path.join(D, f)))
        except Exception as e:
            print("skip", f, e)
out = {"legs": {}, "pairs": {}}
for k, d in legs.items():
    s = d["summary"]
    vr = open(os.path.join(D, f"vram-{k}.txt")).read().strip() if os.path.exists(os.path.join(D, f"vram-{k}.txt")) else None
    ld = open(os.path.join(D, f"load-{k}.txt")).read().strip() if os.path.exists(os.path.join(D, f"load-{k}.txt")) else None
    out["legs"][k] = {"c1": s["c1"], "c8": s["c8"], "vram": vr, "load": ld}
    print(f"{k:13s} c1 perceived p50 {s['c1']['perceived_tps_p50']:7.1f} p90 {s['c1']['perceived_tps_p90']:7.1f} ttft p50 {s['c1']['ttft_p50_s']*1000:6.1f}ms p95 {s['c1']['ttft_p95_s']*1000:6.1f}ms | "
          f"c8 perceived p50 {s['c8']['perceived_tps_p50']:6.1f} total {s['c8']['total_tps_median']:7.1f} {s['c8']['total_tps_rounds']} ttft p50 {s['c8']['ttft_p50_s']*1000:6.1f} p95 {s['c8']['ttft_p95_s']*1000:6.1f}ms | vram {vr} {ld}")


def tok_prefix(a, b):
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i

for a, b in PAIRS:
    if a not in legs or b not in legs:
        continue
    ra = {r["prompt_idx"]: r for r in legs[a]["c1_rows"]}; rb = {r["prompt_idx"]: r for r in legs[b]["c1_rows"]}
    rows = []
    for i in sorted(ra):
        ta, tb = ra[i]["text"], rb[i]["text"]
        cp = tok_prefix(ta, tb)
        ratio = difflib.SequenceMatcher(None, ta, tb, autojunk=False).ratio()
        rows.append({"idx": i, "identical": ta == tb, "common_prefix_chars": cp, "len_a": len(ta), "len_b": len(tb),
                     "prefix_frac": round(cp / max(1, max(len(ta), len(tb))), 3), "seq_ratio": round(ratio, 3)})
    ident = sum(r["identical"] for r in rows)
    s1a, s1b = legs[a]["summary"]["c1"], legs[b]["summary"]["c1"]
    s8a, s8b = legs[a]["summary"]["c8"], legs[b]["summary"]["c8"]
    pr = {"identical": f"{ident}/{len(rows)}", "median_prefix_frac": sorted(r["prefix_frac"] for r in rows)[len(rows)//2],
          "min_prefix_chars": min(r["common_prefix_chars"] for r in rows),
          "c1_perceived_ratio": round(s1b["perceived_tps_p50"] / s1a["perceived_tps_p50"], 3),
          "c8_total_ratio": round(s8b["total_tps_median"] / s8a["total_tps_median"], 3),
          "c8_perceived_ratio": round(s8b["perceived_tps_p50"] / s8a["perceived_tps_p50"], 3),
          "c1_ttft_ratio": round(s1b["ttft_p50_s"] / s1a["ttft_p50_s"], 3), "rows": rows}
    out["pairs"][f"{a}__{b}"] = pr
    print(f"\n{a} vs {b}: identical {pr['identical']}, median common-prefix frac {pr['median_prefix_frac']}, min prefix chars {pr['min_prefix_chars']}; "
          f"c1 perceived x{pr['c1_perceived_ratio']}, c8 total x{pr['c8_total_ratio']}, c8 perceived x{pr['c8_perceived_ratio']}, c1 ttft x{pr['c1_ttft_ratio']}")
    if os.environ.get("VERBOSE"):
        for r in rows:
            print("   ", r)
json.dump(out, open(os.path.join(D, "SUMMARY.json"), "w"), indent=1)
