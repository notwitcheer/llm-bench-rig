import subprocess
import re

QUERY_FIELDS = "name,memory.used,memory.total,utilization.gpu,temperature.gpu"
SMI_CMD = ["nvidia-smi", "--query-gpu=" + QUERY_FIELDS, "--format=csv,noheader"]

def parse_nvidia_smi(raw: str) -> dict:
    line = raw.strip().split("\n")[-1]
    parts = [p.strip() for p in line.split(",")]
    return {
        "name": parts[0],
        "vram_used_mib": int(re.search(r"(\d+)", parts[1]).group(1)),
        "vram_total_mib": int(re.search(r"(\d+)", parts[2]).group(1)),
        "gpu_util_pct": int(re.search(r"(\d+)", parts[3]).group(1)),
        "temp_c": int(parts[4].strip()),
    }

def get_gpu_stats() -> dict:
    result = subprocess.run(SMI_CMD, capture_output=True, text=True, timeout=5)
    return parse_nvidia_smi(result.stdout)

def get_vram_used_mib() -> int:
    return get_gpu_stats()["vram_used_mib"]
