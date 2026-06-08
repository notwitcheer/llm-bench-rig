#!/usr/bin/env bash
# Run the embedding bench over all arms x datasets, Donald-safe.
# Small models usually co-reside with Donald; we only restart Donald on EXIT as insurance.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

# Reduce CUDA fragmentation so the encodes fit in the ~12GB left free by Donald.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASETS=("${@:-scifact}")
ARMS=(e5-small qwen3-text qwen3-vl)

restore_donald() {
  echo "[run_embed_bench] ensuring Donald (llama-server) is up"
  systemctl is-active --quiet llama-server.service || sudo systemctl start llama-server.service || true
}
trap restore_donald EXIT

for ds in "${DATASETS[@]}"; do
  for arm in "${ARMS[@]}"; do
    echo "=== $arm x $ds ==="
    .venv/bin/python -m scripts.embed_bench.run "$arm" "$ds" || echo "[WARN] $arm x $ds failed, continuing"
  done
done
echo "[run_embed_bench] done"
