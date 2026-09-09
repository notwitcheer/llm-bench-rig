#!/usr/bin/env python3
"""Fold llama-bench depth sweeps into speed.json and build the speed-at-depth table.

Two jobs, both idempotent:

1. merge: for every results/<slug>/bench.depth.json (raw `llama-bench -o json` output of a
   `-p 512 -n 128 -d 0,8192,32768` sweep) whose speed.json lacks `@d` keys, add
   `pp512@d8192`, `tg128@d8192`, `pp512@d32768`, `tg128@d32768` entries
   ({tokens_per_sec, stddev}) plus `depth_source` (file + build) and leave the rest of
   speed.json untouched. The d=0 rows are NOT written over the existing pp512/tg128 (those
   came from the original treatment run and are the board's speed row).

2. table: read every speed.json that has depth keys and emit dataset/speed_at_depth.csv and
   a markdown table (stdout, or --md <path>) with the two agent-shaped columns the
   local-frustrations harvest asked for (2026-09-08):
     - decode at 32k: tg128@d32768 and its retention vs tg128 at d=0
     - TTFT for a 16k prompt: DERIVED as 16384 / pp512@d8192 seconds. pp512 measured at
       8192 tokens of depth is the closest measured prefill rate for a mid-size prompt; the
       real number is a little worse (prefill slows as the prompt grows), so the column is
       labelled "est." and the served lane's measured TTFT (speed.json `ttft_16k_s` when a
       treatment recorded one) is preferred when present.

Usage:
  python scripts/depth_table.py merge results/
  python scripts/depth_table.py table results/ [-o dataset/speed_at_depth.csv] [--md -]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DEPTHS = (8192, 32768)
PROMPT_16K = 16384

# meta.json `name`/`quant` are inconsistent for ladder rungs and hand-run treatments
# (run_treatment only recognises Q*_K_* names). Board display names by slug win.
NAMES = {
    "qwen3-8-27b-ud-iq3-xxs": ("Qwen3.8-27B", "UD-IQ3_XXS"),
    "qwen3-8-27b-q6-k": ("Qwen3.8-27B", "Q6_K"),
    "qwen3-6-27b-q6-k": ("Qwen3.6-27B", "Q6_K"),
    "qwen3-6-27b-q4-k-m": ("Qwen3.6-27B", "Q4_K_M"),
    "gemma-4-31b-q4-0-it": ("Gemma 4 31B-it", "Q4_0"),
    "ornith-1-5-35b-q4-k-m": ("Ornith 1.5 35B-A3B", "Q4_K_M"),
    "nvidia-nemotron-3-5-lightning-30b-a3b-q4-k-m": ("Nemotron 3.5 Lightning 30B-A3B", "Q4_K_M"),
    "qwopus3-8-27b-flash-q6-k": ("Qwopus3.8-27B-Flash", "Q6_K"),
    "qwen3-8-flash-next-ud-q2-k-xl": ("Qwen3.8-Flash-Next 125B-A10B", "UD-Q2_K_XL"),
}


def _rows_from_raw(raw: list[dict]) -> tuple[dict[str, dict], str | None]:
    out = {}
    build = None
    for r in raw:
        build = build or r.get("build_commit")
        depth = int(r.get("n_depth", 0) or 0)
        if depth not in DEPTHS:
            continue
        if r.get("n_prompt"):
            key = f"pp{r['n_prompt']}@d{depth}"
        elif r.get("n_gen"):
            key = f"tg{r['n_gen']}@d{depth}"
        else:
            continue
        out[key] = {"tokens_per_sec": round(float(r["avg_ts"]), 2),
                    "stddev": round(float(r.get("stddev_ts", 0.0)), 2)}
    return out, build


def merge(results_root: Path) -> list[str]:
    merged = []
    for d in sorted(results_root.iterdir()):
        raw_p, sp_p = d / "bench.depth.json", d / "speed.json"
        if not (raw_p.exists() and sp_p.exists()):
            continue
        speed = json.loads(sp_p.read_text())
        if not isinstance(speed, dict) or any(k.endswith("@d32768") for k in speed):
            continue
        rows, build = _rows_from_raw(json.loads(raw_p.read_text()))
        if not rows:
            continue
        speed.update(rows)
        speed["depth_source"] = {"file": "bench.depth.json", "build_commit": build}
        sp_p.write_text(json.dumps(speed, indent=2) + "\n")
        merged.append(d.name)
    return merged


def _tps(speed: dict, key: str):
    v = speed.get(key)
    if isinstance(v, dict):
        return v.get("tokens_per_sec")
    return None


def collect(results_root: Path) -> list[dict]:
    rows = []
    for d in sorted(results_root.iterdir()):
        sp_p, meta_p = d / "speed.json", d / "meta.json"
        if not sp_p.exists():
            continue
        speed = json.loads(sp_p.read_text())
        if not isinstance(speed, dict) or _tps(speed, "tg128@d32768") is None:
            continue
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        name, quant = NAMES.get(d.name, (meta.get("name", d.name), meta.get("quant", speed.get("quant", ""))))
        if quant == "unknown":
            quant = ""
        tg0, tg32 = _tps(speed, "tg128"), _tps(speed, "tg128@d32768")
        pp8 = _tps(speed, "pp512@d8192")
        measured = speed.get("ttft_16k_s")
        rows.append({
            "slug": d.name,
            "model": name,
            "quant": quant,
            "engine": speed.get("engine", meta.get("engine", "llama.cpp")),
            "n_cpu_moe": speed.get("n_cpu_moe", ""),
            "tg128": tg0,
            "tg128_d8192": _tps(speed, "tg128@d8192"),
            "tg128_d32768": tg32,
            "decode_retention_32k": round(100 * tg32 / tg0, 1) if tg0 and tg32 else None,
            "pp512_d8192": pp8,
            "ttft_16k_est_s": round(PROMPT_16K / pp8, 1) if pp8 else None,
            "ttft_16k_measured_s": measured if isinstance(measured, (int, float)) else None,
        })
    rows.sort(key=lambda r: -(r["tg128_d32768"] or 0))
    return rows


FIELDS = ["slug", "model", "quant", "engine", "n_cpu_moe", "tg128", "tg128_d8192", "tg128_d32768",
          "decode_retention_32k", "pp512_d8192", "ttft_16k_est_s", "ttft_16k_measured_s"]


def write_csv(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r[k]) for k in FIELDS})


def markdown(rows: list[dict]) -> str:
    lines = ["| Model | Quant | tg128 | tg128 @32k | held | pp512 @8k | TTFT, 16k prompt (est.) |",
             "|-------|-------|------:|-----------:|-----:|----------:|------------------------:|"]
    for r in rows:
        name = r["model"] + (f" (`--n-cpu-moe {r['n_cpu_moe']}`)" if r["n_cpu_moe"] not in ("", None) else "")
        ttft = f"{r['ttft_16k_measured_s']:.1f} s" if r["ttft_16k_measured_s"] is not None else (
            f"~{r['ttft_16k_est_s']:.1f} s" if r["ttft_16k_est_s"] is not None else "not measured")
        pp8 = f"{r['pp512_d8192']:,.0f}" if r["pp512_d8192"] else "not measured"
        lines.append(f"| {name} | {r['quant']} | {r['tg128']:.1f} | {r['tg128_d32768']:.1f} | "
                     f"{r['decode_retention_32k']:.0f}% | {pp8} | {ttft} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("cmd", choices=["merge", "table"])
    ap.add_argument("results_root")
    ap.add_argument("-o", "--out", default="dataset/speed_at_depth.csv")
    ap.add_argument("--md", default=None, help="write the markdown table here ('-' = stdout)")
    a = ap.parse_args(argv)
    root = Path(a.results_root)
    if a.cmd == "merge":
        merged = merge(root)
        print(f"merged {len(merged)}: {' '.join(merged)}" if merged else "nothing to merge")
        return 0
    rows = collect(root)
    write_csv(rows, Path(a.out))
    md = markdown(rows)
    if a.md == "-":
        print(md)
    elif a.md:
        Path(a.md).write_text(md + "\n")
    print(f"{len(rows)} rows -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
