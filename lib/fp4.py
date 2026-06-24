"""FP4-bench pure helpers (ADR-0003: tested logic; the GPU drivers only orchestrate).
parse_quant_kernel is the load-bearing guard — on sm_120 an NVFP4 checkpoint usually runs the
Marlin DEQUANT kernel (FP4->bf16, no FP4 FLOPS), so 'which kernel' decides whether a tok/s number
measures the format or vLLM's routing fallback."""
from __future__ import annotations

import statistics


def median_tps(samples: list[float]) -> float:
    return statistics.median(samples)


def speedup(tps: float, base_tps: float) -> float:
    return tps / base_tps


def parse_quant_kernel(vllm_log: str) -> str:
    """Classify the low-bit kernel vLLM chose, from its init log (case-insensitive)."""
    t = vllm_log.lower()
    if "cutlass" in t and ("fp4" in t or "nvfp4" in t or "scaled_mm" in t):
        return "cutlass-fp4"
    if "flashinfer" in t:
        return "flashinfer"
    if "machete" in t:
        return "machete"
    if "marlin" in t and ("fp8" in t or "w8a" in t):
        return "fp8-marlin"
    if "marlin" in t:
        return "marlin"
    return "unknown"
