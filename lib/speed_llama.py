import subprocess
import re
from pathlib import Path
from lib.config import get

def build_bench_command(model_path: str, n_cpu_moe: int | None = None,
                        n_gpu_layers: int | None = None) -> list[str]:
    bench_bin = str(Path(get("llama_cpp.bench_bin")).expanduser())
    ngl = n_gpu_layers if n_gpu_layers is not None else get("speed.n_gpu_layers", 99)
    ctx_lengths = get("speed.context_lengths", [128, 512, 2048])
    gen_length = get("speed.generation_length", 128)

    pp_args = []
    for cl in ctx_lengths:
        pp_args.extend(["-p", str(cl)])

    cmd = [
        bench_bin, "-m", model_path,
        "-ngl", str(ngl),
        *pp_args,
        "-n", str(gen_length),
    ]
    if n_cpu_moe is not None:
        cmd += ["--n-cpu-moe", str(n_cpu_moe)]
    return cmd

def parse_llama_bench_output(raw: str) -> dict:
    results = {}
    model_size_gib = None
    params = None
    backend = None

    # Column indices resolved from the header row; -1 means not found.
    col = {"size": -1, "params": -1, "backend": -1, "test": -1, "t/s": -1}

    lines = raw.split("\n")
    for line in lines:
        if not line.startswith("|"):
            continue
        if "---" in line:
            continue

        cells = [c.strip() for c in line.split("|")[1:-1]]
        if not cells:
            continue

        # Detect the header row by presence of both "model" and "test" cells.
        lowered = [c.lower() for c in cells]
        if "model" in lowered and "test" in lowered:
            for key in col:
                if key in lowered:
                    col[key] = lowered.index(key)
            continue

        # Skip rows where we haven't resolved the header yet.
        if col["test"] == -1 or col["t/s"] == -1:
            continue

        # Skip rows that don't have enough columns.
        max_needed = max(col[k] for k in col if col[k] != -1)
        if len(cells) <= max_needed:
            continue

        if model_size_gib is None:
            if col["size"] != -1:
                size_match = re.search(r"([\d.]+)\s*GiB", cells[col["size"]])
                if size_match:
                    model_size_gib = float(size_match.group(1))
            if col["params"] != -1:
                params = cells[col["params"]].strip()
            if col["backend"] != -1:
                backend = cells[col["backend"]].strip()

        test_name = cells[col["test"]].strip()
        ts_match = re.match(r"([\d.]+)\s*±\s*([\d.]+)", cells[col["t/s"]])
        if ts_match:
            results[test_name] = {
                "tokens_per_sec": float(ts_match.group(1)),
                "stddev": float(ts_match.group(2)),
            }

    results["model_size_gib"] = model_size_gib
    results["params"] = params
    results["backend"] = backend
    return results

# NOTE: bench.py wires per-model offload into run_llama_bench in Task 4 of the plan.
def run_llama_bench(model_path: str, n_cpu_moe: int | None = None,
                    n_gpu_layers: int | None = None) -> dict:
    cmd = build_bench_command(model_path, n_cpu_moe=n_cpu_moe, n_gpu_layers=n_gpu_layers)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        raise RuntimeError(f"llama-bench failed: {result.stderr}")
    return parse_llama_bench_output(result.stdout)
