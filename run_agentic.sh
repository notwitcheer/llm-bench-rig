#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
echo "Hermes Pairing Benchmark — starting $(date)"
python3 -m lib.agentic.run 2>&1 | tee "results/agentic_run_$(date +%Y%m%d_%H%M).log"
echo "Done $(date)"
