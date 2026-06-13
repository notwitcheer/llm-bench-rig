#!/usr/bin/env python3
"""Aggregate the four spec-decode legs of one target into a comparison table + combined JSON.

Reads results/<target>/specdecode-<leg>.json (+ concurrent-<leg>.json) for
leg in {baseline,mtp,eagle3,dflash}, computes single-stream speedup vs baseline and
collects acceptance length/rate, and writes results/<target>/specdecode-summary.json
(which the chart + report consume). The speedup invariant lives in the tested
lib.specdecode.speedup (ADR-0003); this is just I/O + layout.

  python3 scripts/aggregate_specdecode.py gemma-4-26b-a4b
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.specdecode import speedup  # noqa: E402

LEGS = ["baseline", "mtp", "eagle3", "dflash"]
WORKLOADS = ["prose", "Q&A", "JSON", "code", "repetitive"]


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "gemma-4-26b-a4b"
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", target)

    sd = {leg: _load(os.path.join(root, f"specdecode-{leg}.json")) for leg in LEGS}
    cc = {leg: _load(os.path.join(root, f"concurrent-{leg}.json")) for leg in LEGS}

    def avg_tps(leg):
        d = sd.get(leg)
        if not d:
            return None
        vals = [w["decode_tps"] for w in d["workloads"].values() if w.get("decode_tps")]
        return round(sum(vals) / len(vals), 1) if vals else None

    base = avg_tps("baseline")
    summary = {"target": target, "baseline_avg_tps": base, "methods": {}}

    print(f"\n=== spec-decode three-way: {target} ===")
    print(f"{'method':9} {'avg t/s':>8} {'speedup':>8} {'accept_len':>11} {'accept_rate':>11} {'num_spec':>9}")
    for leg in LEGS:
        d = sd.get(leg)
        if not d:
            print(f"{leg:9} {'(missing)':>8}")
            continue
        a = avg_tps(leg)
        acc = d.get("acceptance", {})
        spd = speedup(a, base) if (a and base) else None
        nspec = d.get("meta", {}).get("num_spec")
        summary["methods"][leg] = {
            "avg_tps": a, "speedup": spd,
            "acceptance_length": acc.get("acceptance_length"),
            "acceptance_rate": acc.get("acceptance_rate"),
            "num_spec": nspec,
            "per_workload_tps": {k: v["decode_tps"] for k, v in d["workloads"].items()},
            "per_workload_speedup": {
                k: speedup(v["decode_tps"], sd["baseline"]["workloads"][k]["decode_tps"])
                for k, v in d["workloads"].items()
                if sd.get("baseline") and k in sd["baseline"]["workloads"]
            } if leg != "baseline" else {},
            "concurrency": {str(r["concurrency"]): r["aggregate_tps"] for r in (cc.get(leg) or [])},
        }
        m = summary["methods"][leg]
        print(f"{leg:9} {a if a else '-':>8} {(str(spd)+'x') if spd else '-':>8} "
              f"{acc.get('acceptance_length') or '-':>11} {acc.get('acceptance_rate') or '-':>11} {nspec or '-':>9}")

    # per-workload speedup table (shows the workload spread / flatness — the DFlash thesis)
    print(f"\n--- single-stream speedup by workload (vs baseline) ---")
    print(f"{'method':9} " + " ".join(f"{w:>10}" for w in WORKLOADS))
    for leg in ["mtp", "eagle3", "dflash"]:
        m = summary["methods"].get(leg)
        if not m:
            continue
        row = m["per_workload_speedup"]
        print(f"{leg:9} " + " ".join(f"{(str(row.get(w))+'x') if row.get(w) else '-':>10}" for w in WORKLOADS))

    out = os.path.join(root, "specdecode-summary.json")
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
