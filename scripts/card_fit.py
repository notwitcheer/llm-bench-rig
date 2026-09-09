#!/usr/bin/env python3
"""What your card can run: a fit table from measured VRAM peaks on the RTX 5090 board.

For every results dir with quality.json + speed.json (llama.cpp rows only, vLLM sizes
its own pool so its peak is not a fit number), take:
  - `vram_peak_mib`: the peak sampled during the llama-bench sweep, whose largest
    prompt is 16,384 tokens, so the peak is weights + KV cache for a 16k prompt +
    compute buffers. That is the number a 16k-context agent session needs.
  - `model_size_gib`: the GGUF file size.
  - q_avg (five-task board mean) and GPQA-diamond think-off when present.
and derive, per card size (8, 12, 16, 24, 32 GB):
  - "yes"     peak + HEADROOM fits the card: runs resident with a 16k context on this recipe
  - "tight"   the file fits but the 16k peak does not: shorter context, or KV cache quantised
              (q8_0 KV roughly halves the cache), or a smaller quant of the same model
  - "offload" the file itself is larger than the card: experts in RAM if it is a MoE
              (see the Flash-Next and Ling-3 offload ladders), otherwise no
Rows that were themselves measured under expert offload (`n_cpu_moe` in speed.json) are
listed with their offload point and marked as such.

This is a 5090 measurement applied as a budget, not a measurement on those cards: peak
VRAM is the same bytes on any CUDA card at the same context and flags, but decode speed
scales with the card's memory bandwidth and is not shown here. HEADROOM is 768 MiB for
the driver and display; a headless card can shave it.

Usage: python scripts/card_fit.py results/ [-o dataset/card_fit.md] [--csv dataset/card_fit.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.board import quality_average, think_mode  # noqa: E402

CARDS_GB = (8, 12, 16, 24, 32)
HEADROOM_MIB = 768
MIB_PER_GB = 1024  # card "GB" labels are GiB in practice (a 24 GB card exposes 24,576 MiB)

# board display names + quant by slug where meta.json is inconsistent (ladder rungs carry
# quant "unknown", hand-run treatments carry no name). Anything not listed uses meta.json.
NAMES = {
    "gemma-4-31b-it-q6-k": ("Gemma 4 31B-it", "Q6_K"),
    "gemma-4-31b-q4-0-it": ("Gemma 4 31B-it (unsloth cut)", "Q4_0"),
    "google-gemma-4-31b-it-q4-0": ("Gemma 4 31B-it (Google QAT)", "Q4_0"),
    "gemma-4-12b-it-q6-k": ("Gemma 4 12B-it", "Q6_K"),
    "qwen3-8-27b-ud-iq3-xxs": ("Qwen3.8-27B", "UD-IQ3_XXS"),
    "qwen3-8-27b-ud-iq2-m": ("Qwen3.8-27B", "UD-IQ2_M"),
    "qwen3-8-27b-ud-iq2-xxs": ("Qwen3.8-27B", "UD-IQ2_XXS"),
    "qwen3-8-flash-next-ud-q2-k-xl": ("Qwen3.8-Flash-Next 125B-A10B", "UD-Q2_K_XL"),
    "qwopus3-8-27b-flash-q6-k": ("Qwopus3.8-27B-Flash", "Q6_K"),
    "qwopus-coder-compat": ("Qwopus3.6-27B-Coder-Compat-MTP", "Q6_K"),
    "qwable-5-27b-coder": ("Qwable-5-27B-Coder", "Q6_K"),
    "qwable-27b-q4-k-m": ("Qwable-27B", "Q4_K_M"),
    "qwen3-6-27b-nvfp4-mtp": ("Qwen3.6-27B-NVFP4-MTP (gguf)", "NVFP4"),
    "north-mini-code": ("North-Mini-Code 1.0", "UD-Q6_K"),
    "ling-3-0-flash-q3-k-m": ("Ling-3.0-flash 127B-A5B", "Q3_K_M"),
    "ling-3-0-flash-iq3-xxs": ("Ling-3.0-flash 127B-A5B", "IQ3_XXS"),
    "ling-3-0-flash-iq2-m": ("Ling-3.0-flash 127B-A5B", "IQ2_M"),
    "nvidia-nemotron-3-5-lightning-30b-a3b-iq3-xxs": ("Nemotron 3.5 Lightning 30B-A3B", "IQ3_XXS"),
    "nvidia-nemotron-3-5-lightning-30b-a3b-iq2-m": ("Nemotron 3.5 Lightning 30B-A3B", "IQ2_M"),
    "nvidia-nemotron-3-5-lightning-30b-a3b-iq4-xs": ("Nemotron 3.5 Lightning 30B-A3B", "IQ4_XS"),
    "nvidia-nemotron-3-5-lightning-30b-a3b-iq2-xxs": ("Nemotron 3.5 Lightning 30B-A3B", "IQ2_XXS"),
    "nvidia-nemotron-3-5-lightning-30b-a3b-q4-k-m": ("Nemotron 3.5 Lightning 30B-A3B", "Q4_K_M"),
    "nvidia-nemotron-3-5-lightning-30b-a3b-q5-k-m": ("Nemotron 3.5 Lightning 30B-A3B", "Q5_K_M"),
    "nvidia-nemotron-3-5-lightning-30b-a3b-q3-k-m": ("Nemotron 3.5 Lightning 30B-A3B", "Q3_K_M"),
    "nvidia-nemotron-cascade-2-30b-a3b-q4-k-m": ("Nemotron Cascade 2 30B-A3B", "Q4_K_M"),
    "ornith-1-5-35b-q4-k-m": ("Ornith 1.5 35B-A3B", "Q4_K_M"),
}
# superseded duplicate dirs (same model+quant re-run under the current slug scheme)
EXCLUDE = {"ornith-35b"}
# rows whose speed.json lost the vram sample (sampler read 10 MiB) or never had one:
# measured peak supplied from the report text, in MiB
PEAK_OVERRIDE = {
    "nvidia-nemotron-cascade-2-30b-a3b-q4-k-m": None,   # no usable sample on disk
    "qwopus3-8-27b-flash-q6-k": None,                   # hand-run treatment, no sampler; not estimated
}
SIZE_OVERRIDE = {"qwopus3-8-27b-flash-q6-k": 20.9, "qwen3-8-flash-next-ud-q2-k-xl": 73.45}


def _score(v):
    return v.get("score") if isinstance(v, dict) else v


def fit(peak_mib: int | None, size_gib: float | None, card_gb: int, offloaded: bool = False) -> str:
    """offloaded rows: the measured peak is an offload point, so a card that holds it can run
    the same recipe ("offload"); a card that cannot gets "no" (a smaller n_cpu_moe would be
    needed and is not measured here)."""
    cap = card_gb * MIB_PER_GB
    if offloaded:
        if peak_mib is None:
            return "?"
        return "offload" if peak_mib + HEADROOM_MIB <= cap else "no"
    if size_gib is not None and size_gib * 1024 > cap:
        return "no"
    if peak_mib is None:
        return "?"
    return "yes" if peak_mib + HEADROOM_MIB <= cap else "tight"


def collect(results_root: Path) -> list[dict]:
    rows = []
    for d in sorted(results_root.iterdir()):
        if d.name in EXCLUDE:
            continue
        qf, sf, mf = d / "quality.json", d / "speed.json", d / "meta.json"
        if not (qf.exists() and sf.exists()):
            continue
        speed = json.loads(sf.read_text())
        if not isinstance(speed, dict):
            continue
        meta = json.loads(mf.read_text()) if mf.exists() else {}
        engine = speed.get("engine", meta.get("engine", "llama.cpp"))
        if engine != "llama.cpp":
            continue
        quality = json.loads(qf.read_text())
        qavg = quality_average(quality)
        if qavg is None:
            continue
        peak = speed.get("vram_peak_mib")
        peak = int(peak) if isinstance(peak, (int, float)) and peak > 100 else None
        if d.name in PEAK_OVERRIDE:
            peak = PEAK_OVERRIDE[d.name] and int(PEAK_OVERRIDE[d.name])
        size = SIZE_OVERRIDE.get(d.name, speed.get("model_size_gib", meta.get("size_gib")))
        size = float(size) if isinstance(size, (int, float)) else None
        name, quant = NAMES.get(d.name, (meta.get("name", d.name), meta.get("quant", "")))
        if quant in (None, "unknown"):
            quant = ""
        gp = d / "gpqa.json"
        gpqa = None
        if gp.exists():
            g = json.loads(gp.read_text())
            if isinstance(g, dict) and isinstance(g.get("score"), (int, float)):
                gpqa = g["score"]
        offload_n = speed.get("n_cpu_moe")
        # a row whose file is far larger than its measured peak was run with experts in RAM
        offloaded = offload_n is not None or (size is not None and peak is not None and size * 1024 > peak * 1.5)
        if offloaded and offload_n is None:
            offload_n = "all"
        rows.append({
            "slug": d.name,
            "model": name,
            "quant": quant,
            "think": think_mode(meta, d.name),
            "size_gib": size,
            "vram_peak_mib": peak,
            "offload_n_cpu_moe": offload_n,
            "q_avg": qavg,
            "gpqa": gpqa,
            **{f"fits_{c}gb": fit(peak, size, c, offloaded) for c in CARDS_GB},
        })
    rows.sort(key=lambda r: (-r["q_avg"], r["vram_peak_mib"] or 0))
    return rows


def _gb(mib):
    return f"{mib / 1024:.1f}" if mib else "?"


def markdown(rows: list[dict]) -> str:
    out = ["| Model | Quant | file GiB | VRAM @16k GiB | q_avg | GPQA | 8 GB | 12 GB | 16 GB | 24 GB | 32 GB |",
           "|-------|-------|---------:|--------------:|------:|-----:|:----:|:-----:|:-----:|:-----:|:-----:|"]
    sym = {"yes": "✅", "tight": "🟡", "offload": "🟠", "no": "⬜", "?": "?"}
    for r in rows:
        if r["think"] is True:
            continue  # think-on rows are a different quality axis; keep the table on the think-off board
        name = r["model"]
        if r["offload_n_cpu_moe"] is not None:
            name += f" (`--n-cpu-moe {r['offload_n_cpu_moe']}`)"
        cells = [sym.get(r[f"fits_{c}gb"], r[f"fits_{c}gb"]) for c in CARDS_GB]
        gp = f"{r['gpqa']:.1f}" if r["gpqa"] is not None else ""
        size = f"{r['size_gib']:.1f}" if r["size_gib"] is not None else "?"
        out.append(f"| {name} | {r['quant']} | {size} | {_gb(r['vram_peak_mib'])} | {r['q_avg']:.1f} | {gp} | " + " | ".join(cells) + " |")
    return "\n".join(out)


LEGEND = ("✅ runs resident with a 16k context on the board recipe (measured peak + 768 MiB headroom fits). "
          "🟡 the file fits but the 16k peak does not: use a shorter context, quantise the KV cache (q8_0 roughly halves it), or take a smaller quant. "
          "⬜ does not fit: the file is larger than the card. "
          "🟠 runs with experts in system RAM: this row was measured under expert offload at the shown `--n-cpu-moe` (its VRAM column is that point, and it needs system RAM for the rest of the file, 59 GB on the measuring box); "
          "a card that cannot hold that point would need a larger `--n-cpu-moe`, not measured here (see the offload ladders in the Flash-Next and Ling-3 reports).")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("results_root")
    ap.add_argument("-o", "--out", default="dataset/card_fit.md")
    ap.add_argument("--csv", default="dataset/card_fit.csv")
    a = ap.parse_args(argv)
    rows = collect(Path(a.results_root))
    Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})
    md = ("# What your card can run\n\n"
          "Every row is a model + quant measured on the RTX 5090 board (think-off quality table). "
          "**VRAM @16k** is the peak sampled during the speed sweep, whose largest prompt is 16,384 tokens: weights, KV cache for a 16k prompt, compute buffers. "
          "The card columns apply that measured peak as a budget; they are not measurements on those cards, and decode speed on a smaller card scales with its memory bandwidth. "
          "Regenerate with `python scripts/card_fit.py results/`.\n\n"
          + LEGEND + "\n\n" + markdown(rows) + "\n")
    Path(a.out).write_text(md)
    print(f"{len(rows)} rows -> {a.out}, {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
