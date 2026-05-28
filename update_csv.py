import json, csv
from pathlib import Path
from datetime import date

RESULTS = Path("results")
CSV_PATH = Path("dataset/benchmarks.csv")

MODELS_TO_ADD = {
    "nemotron-3-nano-30b-a3b-q4-k-m": {"arch": "MoE (3B active)"},
    "gpt-oss-20b-q4-k-m":             {"arch": "Dense"},
    "qwen3-6-27b-mtp-q4-k-m":         {"arch": "Dense (MTP)", "display_name": "Qwen3.6-27B-MTP"},
    "qwen3-coder-next-ud-q2-k-xl":    {"arch": "MoE"},
}

existing_models = set()
with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        existing_models.add(row["model"])

today = date.today().isoformat()
new_rows = []

for slug, info in MODELS_TO_ADD.items():
    meta = json.loads((RESULTS / slug / "meta.json").read_text())
    speed = json.loads((RESULTS / slug / "speed.json").read_text())

    name = info.get("display_name", meta["name"])
    if name in existing_models:
        print(f"  skip {name} (already in CSV)")
        continue

    params = speed.get("params", "").replace(" B", "")
    size = speed.get("model_size_gib", meta.get("size_gib"))

    for key, val in speed.items():
        if not isinstance(val, dict) or "tokens_per_sec" not in val:
            continue
        new_rows.append([
            name, info["arch"], params, meta["quant"], size,
            "llama.cpp", speed.get("backend", "CUDA"),
            "RTX 5090", 32, key,
            val["tokens_per_sec"], val["stddev"], today,
        ])

with open(CSV_PATH, "a") as f:
    writer = csv.writer(f)
    for row in new_rows:
        writer.writerow(row)

print(f"Added {len(new_rows)} rows ({len(MODELS_TO_ADD)} models) to {CSV_PATH}")
