# FP4 on a consumer RTX 5090: the Blackwell headline loses to plain int4

**Rig:** one RTX 5090 32GB (sm_120a) · Qwen3-14B (dense) · vLLM 0.21 (torch 2.11+cu130) · greedy, warm-up discarded
**Question:** FP4 is the Blackwell headline. Does it actually go faster than the int4/FP8 quants a home-lab user already runs, on a *consumer* sm_120 card? Subjects: bf16 base, AWQ-int4, FP8, and NVIDIA NVFP4, same model, same harness.

## The numbers (Qwen3-14B, vLLM, decode tok/s)

| quant | kernel | batch 1 | batch 8 | batch 32 | vs AWQ (b1) | runs out of the box? |
|---|---|---|---|---|---|---|
| **AWQ int4** | awq-marlin | **150** | **1188** | **3937** | 1.00x | ✅ |
| FP8 | fp8-marlin | 90 | 711 | 2647 | 0.60x | ✅ |
| **NVFP4** | nvfp4 cutlass (JIT) | 100 | 801 | 3321 | **0.66x** | ❌ |
| bf16 | — | — | — | — | — | ❌ OOM |

Two findings, and they compound.

## Finding 1 — NVFP4 is the only quant that won't start without a compiler toolchain

AWQ and FP8 load and run with prebuilt Marlin kernels — nothing to compile. NVFP4 does not. On sm_120, vLLM hands the FP4 GEMM to FlashInfer, which **JIT-compiles native sm_120 FP4 cutlass kernels at load** (`fp4_gemm_cutlass_sm120.cu`, `-gencode=...compute_120f`). That needs three things a normal inference box doesn't have wired up:

- **`ninja`** on `PATH` — it lives in the venv's bin, which running `venv/bin/python` does not add to `PATH`; without it, `FileNotFoundError: ninja` and the engine dies.
- **A real CUDA toolkit (`nvcc`)** — the FlashInfer default resolves `CUDA_HOME=/usr/local/cuda-13.0`, which **doesn't exist on this box** (the installed toolkit is `/usr/local/cuda-13`, nvcc 13.3). Wrong path → `nvcc: not found` → dead.
- **A cache clear** — the first failed attempt writes a `build.ninja` with the bad nvcc path baked in; FlashInfer replays it and keeps failing until you `rm -rf ~/.cache/flashinfer`.

Wire up all three (`CUDA_HOME=/usr/local/cuda-13`, the toolkit + venv bins on `PATH`, clear the cache) and it finally compiles its kernels and runs. That is a meaningful amount of yak-shaving for the format that's supposed to be the easy Blackwell win. AWQ asked for none of it.

## Finding 2 — even with native FP4 kernels running, it's slower than AWQ int4

Once NVFP4 is genuinely on its native cutlass FP4 path (confirmed: it JIT-built `fp4_gemm_cutlass_sm120`, declared quant `modelopt_fp4` — not a Marlin dequant fallback), it still loses to plain AWQ int4 at **every** batch size: 100 vs 150 tok/s at batch 1 (0.66x), 801 vs 1188 at batch 8, 3321 vs 3937 at batch 32. FP8 is slowest of the three (bigger weights, more bytes to move at decode).

The mechanism: batch-1 decode is memory-bandwidth-bound, and FP4 (~10GB) and AWQ-int4 (~9.4GB) move similar bytes, so FP4's compute advantage can't show there. But even at batch 32 — where compute matters more — the FP4 path still trails the mature AWQ-Marlin kernel. The 4x-6x FP4 numbers in the headlines are B200 (sm_100) tensor-core throughput; on consumer sm_120 the JIT'd cutlass FP4 kernels don't have it, and a well-tuned int4-Marlin kernel wins.

## bf16 doesn't even fit

Qwen3-14B in bf16 is 28GB of weights; on a 32GB card vLLM has no room left for a KV cache and refuses to start ("No available memory for the cache blocks"). So the bf16 baseline isn't a consumer option for a 14B at all — quantization isn't optional here, the only question is which one, and the answer is AWQ.

## Finding 3 — the real-FP4 path (QuTLASS MXFP4): 4x is real at the GEMM, lost at the token

vLLM's NVFP4 dequants on sm_120, so it never tests the format itself. The path that *does* run native FP4 weights through native FP4 kernels is **QuTLASS / MR-GPTQ** (arXiv 2509.23202), which reports a genuine **~4x on an RTX 5090**. Built it from source on sm_120a and ran it two ways — at the GEMM, and end to end — on Qwen3-8B (the paper's headline model). The two regimes give opposite answers.

**At the GEMM, the 4x is real — and then some.** QuTLASS's own sm_120 benchmark, on Qwen3-8B's `gate+up` MLP shape (MXFP4 vs torch bf16, TFLOP/s):

| batch | 1 | 32 | 128 | 512 | 2048 |
|---|---|---|---|---|---|
| MXFP4 / bf16 (on-the-fly act quant) | 1.6x | 2.9x | **4.7x** | 5.3x | **5.7x** |
| MXFP4 / bf16 (pre-quantized act) | 1.8x | 3.3x | 5.4x | 5.8x | 6.1x |

It crosses the claimed 4x by batch ~128 and peaks near 6x. So on a *consumer* 5090 the academic FP4 kernels deliver — the opposite of the vLLM/NVFP4 result, because this path runs real MXFP4 matmuls instead of dequantizing to Marlin.

**End to end at decode, it loses to bf16 outright** (HF Transformers, batch 1/8/32 decode tok/s):

| | batch 1 | batch 8 | batch 32 | peak VRAM |
|---|---|---|---|---|
| MXFP4 | 20.4 | 162.2 | 642.1 | 32.4 GB |
| bf16 | 78.6 | 575.9 | 2027.3 | 17.1 GB |
| ratio | **0.26x** | 0.28x | 0.32x | — |

MXFP4 is ~3-4x *slower* than bf16 at the thing a single user actually does, and uses ~2x the VRAM (a 4-bit model heavier than bf16). The mechanism is the same memory-bound logic as Finding 2, sharper: autoregressive decode is M=1 per step, so the GEMM is a small slice of step time and its 4x can't show; meanwhile every layer pays a fixed online Hadamard-rotation + activation-quant cost (`fusedQuantizeMx`) that the tiny decode matmul never amortizes, and the unoptimized HF integration appears to hold a bf16 weight copy alongside the 4-bit weights (the ~+15GB). The paper's end-to-end 4x is a **prefill / large-batch** number, exactly where the GEMM dominates; at batch-1 serving even the real-FP4 path is the wrong choice.

**The build is its own consumer wall.** QuTLASS wants torch 2.8 + CUDA 12.8 + a CUTLASS source compile. The rig's torch 2.11/cu130 fails on the torch headers; CUDA 12.8's nvcc then rejects the box's GCC 15 (`unsupported GNU version`), fixed by forcing `-ccbin g++-14`. The "real FP4" path is gated behind an exact, older toolchain an inference box won't have wired up — which is itself part of the datacenter-vs-consumer story.

## Caveats

- **Snapshot of vLLM 0.21 / FlashInfer 0.6.8 on 2026-06-24.** Kernels improve; the FP4 path may close the gap later. This is the state today.
- **NVFP4 MoE is worse — it's broken, not just slow** (not run here; autopsy from the trackers): negative-scale Marlin + TMA-WS failures on sm_120, with the cutlass block-scaled FP4 kernels gated to `sm_100a` (CUTLASS #2800). Dense is the case that works at all.
- **The academic "real FP4" path (QuTLASS MXFP4) is now measured — see Finding 3.** Its 4x is real at the GEMM (peaks ~6x) but its end-to-end decode loses to bf16; the high-batch regime where it would win is prefill/serving-many, not single-stream.

## Worth it if / not if

- **For a 14B on one 32GB card, use AWQ int4.** It's fastest, it fits, and it loads with zero toolchain fuss.
- **Skip NVFP4 on consumer Blackwell.** It's the only quant that needs a CUDA toolkit + ninja + the right `CUDA_HOME` to even start, and after all that it's slower than AWQ. FP4's speed story is a datacenter (B200) story today.
- **FP8** if you specifically want its accuracy profile and have the VRAM; it's the slowest of the three at decode.
- **QuTLASS MXFP4 only makes sense for compute-bound, high-batch work** (prefill, batched serving), where its real 4-6x GEMM speedup shows. For single-stream decode it's slower than bf16 and far slower than AWQ, and it costs an exact-stack source build to get running. The format is genuine on consumer Blackwell; the *serving* regime where you'd feel it is not the home-lab single-user one.

## Repro

- `lib/fp4.py` (throughput stats + a vLLM kernel-name parser — note: real vLLM logs name many kernels, so the **declared quant** `awq_marlin`/`fp8`/`modelopt_fp4` is the reliable signal). Driver `scripts/bench_fp4_vllm.py` (vLLM, `--gpu-mem-util` knob; bf16-14B needs 0.95 and still OOMs). Aggregate/chart: `scripts/{aggregate,chart}_fp4.py`.
- NVFP4-on-sm_120 recipe that finally worked: `CUDA_HOME=/usr/local/cuda-13` (the box's real toolkit, not the default `cuda-13.0`), `PATH=/usr/local/cuda-13/bin:<venv>/bin:$PATH` (nvcc + ninja), `FLASHINFER_CUDA_ARCH_LIST=12.0f`, `VLLM_USE_FLASHINFER_SAMPLER=0`, and `rm -rf ~/.cache/flashinfer` after any failed attempt.
- QuTLASS (Finding 3) build on sm_120a: torch 2.8 + CUDA 12.8 + `pip install --no-build-isolation -e .` (CUTLASS is a git submodule, clone `--recursive`). The box's GCC 15 trips CUDA 12.8's host-compiler guard (`unsupported GNU version`) — fix with `CC=gcc-14 CXX=g++-14 NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-14"`. Loading the MR-GPTQ checkpoint through Transformers needs the `fp_quant` package (the bridge to the qutlass kernels). Kernel bench: `qutlass/benchmarks/bench_mxfp4_sm120.py`; model decode: `scripts/bench_qutlass.py`.
