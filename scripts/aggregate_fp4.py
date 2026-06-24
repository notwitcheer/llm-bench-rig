"""Aggregate the FP4 bench. The DECLARED quant (awq_marlin / fp8 / modelopt_fp4) is the reliable
kernel signal — the full vLLM init log mentions many kernels (flashinfer attention, cutlass_scaled_mm
for fp8) so substring-parsing it is noisy. bf16-14B OOMs vLLM on 32GB (28GB weights, no KV room), so
AWQ is the practical baseline. The finding: native NVFP4 (modelopt_fp4, JIT-compiled cutlass FP4 on
sm_120) is SLOWER than AWQ int4 at every batch."""
import glob
import json
import os

from lib.fp4 import speedup

KERNEL = {"awq_marlin": "awq-marlin int4", "fp8": "fp8-marlin", "modelopt_fp4": "nvfp4 cutlass (JIT)"}

vllm = {}
for f in glob.glob("results/fp4/vllm__*.json"):
    d = json.load(open(f))
    vllm[d["label"]] = d


def tps_at(d, batch):
    return next((r["tps"] for r in d["batches"] if r["batch"] == batch), None)


def jit_confirmed(label):
    p = f"results/fp4/{label}.initlog"
    return os.path.exists(p) and "fp4_gemm_cutlass_sm120" in open(p, errors="ignore").read()


awq1 = tps_at(vllm["awq"], 1) if "awq" in vllm else None
print(f"{'arm':7} {'declared':14} {'kernel':22} {'b1':>7} {'b8':>7} {'b32':>7} {'b1 vs awq':>10}")
for lab in ["awq", "fp8", "nvfp4"]:
    d = vllm.get(lab)
    if not d:
        continue
    dq = str(d["declared_quant"])
    k = KERNEL.get(dq, dq)
    if lab == "nvfp4" and jit_confirmed("nvfp4"):
        k += " ✓JIT"
    t1, t8, t32 = (tps_at(d, b) for b in (1, 8, 32))
    sp = f"{speedup(t1, awq1):.2f}x" if awq1 and t1 else "-"
    print(f"{lab:7} {dq[:13]:14} {k:22} {str(t1):>7} {str(t8):>7} {str(t32):>7} {sp:>10}")
print("\nbf16: OOM — Qwen3-14B bf16 (28GB) leaves no KV room in vLLM on a 32GB card.")
print("nvfp4: did NOT run out of the box — required CUDA_HOME=/usr/local/cuda-13 + nvcc 13.x + ninja")
print("       on PATH + a flashinfer-cache clear to JIT-compile its sm_120 FP4 kernels.")
