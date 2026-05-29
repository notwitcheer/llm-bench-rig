#!/bin/bash
# Run quality benchmarks on all cached GGUF models.
# Usage: ./run_quality.sh        (runs all)
#        ./run_quality.sh 3      (starts from model #3)

set -euo pipefail
cd ~/benchmark-rig
source venv/bin/activate
export PYTHONUNBUFFERED=1

MODELS=(
  ~/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-GGUF/snapshots/82d411acf4a06cfb8d9b073a5211bf410bfc29bf/Qwen3.6-27B-Q6_K.gguf
  ~/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
  ~/.cache/huggingface/hub/models--unsloth--Qwen3-Coder-Next-GGUF/snapshots/ce09c67b53bc8739eef83fe67b2f5d293c270632/Qwen3-Coder-Next-UD-Q2_K_XL.gguf
  ~/.cache/huggingface/hub/models--unsloth--gemma-4-31B-it-GGUF/snapshots/3f07b20fc8e73cec677713305971e534fe8c4ce3/gemma-4-31B-it-Q6_K.gguf
  ~/.cache/huggingface/hub/models--bartowski--nvidia_Nemotron-Cascade-2-30B-A3B-GGUF/snapshots/931b595fc71b7ca14fb9d935af011f69f7c0434c/nvidia_Nemotron-Cascade-2-30B-A3B-Q4_K_M.gguf
)

START=${1:-1}
TOTAL=${#MODELS[@]}

echo "╔══════════════════════════════════════════════╗"
echo "║  Quality benchmark run — $TOTAL models            ║"
echo "║  Starting from model #$START                      ║"
echo "║  $(date '+%Y-%m-%d %H:%M')                            ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

for i in "${!MODELS[@]}"; do
  NUM=$((i + 1))
  if [ "$NUM" -lt "$START" ]; then
    continue
  fi

  MODEL="${MODELS[$i]}"
  NAME=$(basename "$MODEL" .gguf)

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  [$NUM/$TOTAL] $NAME"
  echo "  Started: $(date '+%H:%M:%S')"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  if python3 bench.py "$MODEL"; then
    echo "  ✓ $NAME complete at $(date '+%H:%M:%S')"
  else
    echo "  ✗ $NAME FAILED at $(date '+%H:%M:%S') — continuing"
  fi
done

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  All done — $(date '+%Y-%m-%d %H:%M')              ║"
echo "╚══════════════════════════════════════════════╝"
