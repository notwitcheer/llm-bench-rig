"""Aggregate the FP4 bench: Tier-1 vLLM quant A/B with the ACTUAL kernel per arm (the
NVFP4-silently-Marlin guard), plus the Tier-2 QuTLASS MXFP4-vs-bf16 sweep if it ran."""
import glob
import json
import os

from lib.fp4 import parse_quant_kernel, speedup

vllm = {}
for f in glob.glob("results/fp4/vllm__*.json"):
    d = json.load(open(f))
    vllm[d["label"]] = d


def tps_at(d, batch):
    return next((r["tps"] for r in d["batches"] if r["batch"] == batch), None)


base = tps_at(vllm["bf16"], 1) if "bf16" in vllm else None
print(f"{'arm':7} {'declared':14} {'kernel':12} {'b1 tok/s':>9} {'vs bf16':>8} {'vram':>6}")
for lab in ["bf16", "awq", "fp8", "nvfp4"]:
    d = vllm.get(lab)
    if not d:
        continue
    logp = f"results/fp4/{lab}.initlog"
    kernel = parse_quant_kernel(open(logp).read()) if os.path.exists(logp) else "?"
    t1 = tps_at(d, 1)
    sp = f"{speedup(t1, base):.2f}x" if base and t1 else "-"
    print(f"{lab:7} {str(d['declared_quant'])[:13]:14} {kernel:12} {str(t1):>9} {sp:>8} "
          f"{d['peak_vram_gb']:>5}G")

q = {}
for f in glob.glob("results/fp4/qutlass__*.json"):
    d = json.load(open(f))
    q[d["arm"]] = d
if "bf16" in q and "mxfp4" in q:
    print("\n[QuTLASS MXFP4 vs bf16, HF Transformers]")
    qb = {r["batch"]: r["tps"] for r in q["bf16"]["batches"]}
    for r in q["mxfp4"]["batches"]:
        bb = qb.get(r["batch"])
        sp = f"{speedup(r['tps'], bb):.2f}x" if bb and r["tps"] else "-"
        print(f"  batch {r['batch']:>2}: mxfp4 {r['tps']} vs bf16 {bb}  -> {sp}")
