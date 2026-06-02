"""Resolve a per-model MoE-offload override from config.

Absent slug or absent `offload:` section -> {} (caller falls back to full-VRAM defaults)."""


def resolve_offload(slug: str, cfg: dict) -> dict:
    section = (cfg or {}).get("offload") or {}
    entry = section.get(slug)
    if not entry:
        return {}
    out = {}
    if "n_cpu_moe" in entry:
        out["n_cpu_moe"] = entry["n_cpu_moe"]
    if "n_gpu_layers" in entry:
        out["n_gpu_layers"] = entry["n_gpu_layers"]
    return out
