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

## Caveats

- **Snapshot of vLLM 0.21 / FlashInfer 0.6.8 on 2026-06-24.** Kernels improve; the FP4 path may close the gap later. This is the state today.
- **NVFP4 MoE is worse — it's broken, not just slow** (not run here; autopsy from the trackers): negative-scale Marlin + TMA-WS failures on sm_120, with the cutlass block-scaled FP4 kernels gated to `sm_100a` (CUTLASS #2800). Dense is the case that works at all.
- **The academic "real FP4" path is the open follow-up.** QuTLASS / MR-GPTQ (arXiv 2509.23202) reports a genuine **4x on a 5090** — but for Qwen3-8B via HF Transformers at high batch vs bf16, and it needs a CUTLASS 4.2.1 source build. Testing whether *that* path beats AWQ on the rig is parked for a dedicated session (the build is its own ~hour).

## Worth it if / not if

- **For a 14B on one 32GB card, use AWQ int4.** It's fastest, it fits, and it loads with zero toolchain fuss.
- **Skip NVFP4 on consumer Blackwell.** It's the only quant that needs a CUDA toolkit + ninja + the right `CUDA_HOME` to even start, and after all that it's slower than AWQ. FP4's speed story is a datacenter (B200) story today.
- **FP8** if you specifically want its accuracy profile and have the VRAM; it's the slowest of the three at decode.

## Repro

- `lib/fp4.py` (throughput stats + a vLLM kernel-name parser — note: real vLLM logs name many kernels, so the **declared quant** `awq_marlin`/`fp8`/`modelopt_fp4` is the reliable signal). Driver `scripts/bench_fp4_vllm.py` (vLLM, `--gpu-mem-util` knob; bf16-14B needs 0.95 and still OOMs). Aggregate/chart: `scripts/{aggregate,chart}_fp4.py`.
- NVFP4-on-sm_120 recipe that finally worked: `CUDA_HOME=/usr/local/cuda-13` (the box's real toolkit, not the default `cuda-13.0`), `PATH=/usr/local/cuda-13/bin:<venv>/bin:$PATH` (nvcc + ninja), `FLASHINFER_CUDA_ARCH_LIST=12.0f`, `VLLM_USE_FLASHINFER_SAMPLER=0`, and `rm -rf ~/.cache/flashinfer` after any failed attempt.
