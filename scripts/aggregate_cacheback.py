"""Aggregate cacheback spec-decode legs into a per-(workload,arm) summary table.
Reads results/cacheback/*.json (written by bench_cacheback.py); AR is the per-workload
baseline (speedup 1.00). Speedup must stay <= MAT (asserted in T6/T7); printed here too."""
import glob
import json

rows = [json.load(open(f)) for f in sorted(glob.glob("results/cacheback/*.json"))]
base = {r["workload"]: r["tok_per_s"] for r in rows if r["arm"] == "ar"}

ARM_ORDER = {"ar": 0, "pld": 1, "cacheback": 2}
print(f"{'workload':10} {'arm':10} {'tok/s':>8} {'speedup':>8} {'MAT':>6} {'vram':>7}")
for r in sorted(rows, key=lambda r: (r["workload"], ARM_ORDER.get(r["arm"], 9))):
    sp = r["tok_per_s"] / base[r["workload"]]
    print(f"{r['workload']:10} {r['arm']:10} {r['tok_per_s']:8.1f} {sp:8.2f} "
          f"{r['mat']:6.2f} {r.get('peak_vram_gb', 0):6.1f}G")
