import json
import os
import re
from pathlib import Path

def detect_engine(model_path: Path) -> str:
    if str(model_path).endswith(".gguf"):
        return "llama.cpp"
    return "vllm"

def parse_model_name(filename: str) -> dict:
    name = filename.replace(".gguf", "")
    # Strip trailing shard suffix e.g. -00001-of-00003
    name = re.sub(r"-\d{3,}-of-\d{3,}$", "", name)
    quant_pattern = r"[-.]((?:UD-)?(?:[QqFf]\d[\w_]*(?:-[A-Z]+)*|[Mm][Xx][Ff][Pp]\d+))$"
    match = re.search(quant_pattern, name)
    if match:
        quant = match.group(1)
        base_name = name[:match.start()]
    else:
        quant = "unknown"
        base_name = name
    return {"base_name": base_name, "quant": quant}

def get_file_size_gib(path: Path) -> float:
    size_bytes = path.stat().st_size if path.is_file() else sum(
        f.stat().st_size for f in path.rglob("*") if f.is_file()
    )
    return round(size_bytes / (1024 ** 3), 2)

def extract_metadata(model_path: Path) -> dict:
    model_path = Path(model_path)
    engine = detect_engine(model_path)
    filename = model_path.name
    name_info = parse_model_name(filename)
    size_gib = get_file_size_gib(model_path)
    slug = re.sub(r"[^a-z0-9]+", "-", name_info["base_name"].lower()).strip("-")
    if name_info["quant"] != "unknown":
        slug += "-" + re.sub(r"[^a-z0-9]+", "-", name_info["quant"].lower()).strip("-")

    return {
        "name": name_info["base_name"],
        "filename": filename,
        "slug": slug,
        "engine": engine,
        "quant": name_info["quant"],
        "size_gib": size_gib,
        "path": str(model_path),
    }

def save_metadata(meta: dict, results_dir: Path):
    out_dir = results_dir / meta["slug"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return out_dir
