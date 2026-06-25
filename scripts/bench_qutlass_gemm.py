"""Tier-2 kernel leg: QuTLASS MXFP4 vs torch-bf16 GEMM throughput (TFLOP/s) on sm_120a.
Reuses qutlass's own benchmarks/bench_mxfp4_sm120.py perf_report, fed Qwen3-8B layer shapes.
This is the regime the paper's ~4x lives in (compute-bound GEMM), the complement to the
end-to-end decode numbers from bench_qutlass.py.

Run (capsule, qutlass-env, from /tmp to avoid the repo-root namespace shadow):
  scp scripts/bench_qutlass_gemm.py witcheer@<capsule>:/tmp/ && \
  ssh <capsule> 'cd /tmp && MPLBACKEND=Agg ~/qutlass-env/bin/python /tmp/bench_qutlass_gemm.py'
Requires: a built qutlass (CUTLASS source, see report repro), scipy, matplotlib, triton.
"""
import sys

# `import qutlass` must resolve to the editable install, not the repo-root dir (a bare `qutlass/`
# folder with no __init__ shadows it as an empty namespace package). Strip cwd/home from the path.
sys.path = [p for p in sys.path if p not in ("", "/home/witcheer", "/home/witcheer/qutlass")]

QUTLASS_BENCH = "/home/witcheer/qutlass/benchmarks/bench_mxfp4_sm120.py"

# exec the upstream bench up to its module-level MODELS loop to grab the `benchmark` perf_report
src = open(QUTLASS_BENCH).read()
ns = {}
exec(compile(src.split("for model, layers in MODELS.items():")[0], QUTLASS_BENCH, "exec"), ns)
benchmark = ns["benchmark"]

# Qwen3-8B main GEMMs as (label, K, N): attn (qkv/o), MLP gate+up, MLP down
QWEN3_8B = [("qkv/o", 4096, 4096), ("gate+up", 4096, 24576), ("down", 12288, 4096)]
for name, K, N in QWEN3_8B:
    print(f"\n######## Qwen3-8B {name}  K={K} N={N}  had=128  (TFLOP/s, higher=better) ########", flush=True)
    benchmark.run(print_data=True, show_plots=False, N=N, K=K, had_size=128)
print("\nKERNEL_BENCH_DONE", flush=True)
