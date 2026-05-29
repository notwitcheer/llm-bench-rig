#!/bin/bash
# Extended context length speed benchmarks (32K, 64K, 128K).
# Runs each context length independently — OOM at 128K won't kill 32K/64K results.
# Results merge into existing results/<slug>/speed.json.
#
# Usage: ./run_extended_ctx.sh        (runs all)
#        ./run_extended_ctx.sh 3      (starts from model #3)

set -euo pipefail
cd ~/benchmark-rig
source venv/bin/activate
export PYTHONUNBUFFERED=1

BENCH_BIN=~/llama.cpp/build/bin/llama-bench

MODELS=(
  ~/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-GGUF/snapshots/82d411acf4a06cfb8d9b073a5211bf410bfc29bf/Qwen3.6-27B-Q6_K.gguf
  ~/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
  ~/.cache/huggingface/hub/models--unsloth--Qwen3-Coder-Next-GGUF/snapshots/ce09c67b53bc8739eef83fe67b2f5d293c270632/Qwen3-Coder-Next-UD-Q2_K_XL.gguf
  ~/.cache/huggingface/hub/models--unsloth--gemma-4-31B-it-GGUF/snapshots/3f07b20fc8e73cec677713305971e534fe8c4ce3/gemma-4-31B-it-Q6_K.gguf
  ~/.cache/huggingface/hub/models--bartowski--nvidia_Nemotron-Cascade-2-30B-A3B-GGUF/snapshots/931b595fc71b7ca14fb9d935af011f69f7c0434c/nvidia_Nemotron-Cascade-2-30B-A3B-Q4_K_M.gguf
)

SLUGS=(
  qwen3-6-27b-q6-k
  qwen3-6-35b-a3b-ud-q4-k-m
  qwen3-coder-next-ud-q2-k-xl
  gemma-4-31b-it-q6-k
  nvidia-nemotron-cascade-2-30b-a3b-q4-k-m
)

CTX_LENGTHS=(32768 65536 131072)

START=${1:-1}
TOTAL=${#MODELS[@]}

echo "╔══════════════════════════════════════════════╗"
echo "║  Extended context speed sweep — $TOTAL models     ║"
echo "║  Context: 32K, 64K, 128K                    ║"
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
  SLUG="${SLUGS[$i]}"
  NAME=$(basename "$MODEL" .gguf)

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  [$NUM/$TOTAL] $NAME"
  echo "  Started: $(date '+%H:%M:%S')"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  for CTX in "${CTX_LENGTHS[@]}"; do
    CTX_K=$((CTX / 1024))
    echo -n "  pp${CTX} (${CTX_K}K): "

    # Run llama-bench for this single context length
    OUTPUT=$($BENCH_BIN -m "$MODEL" -ngl 99 -p "$CTX" -n 0 2>&1) || {
      echo "OOM or failed — skipping"
      continue
    }

    # Parse output and merge into speed.json via Python
    python3 -c "
import json, re, sys

output = '''$OUTPUT'''
f = 'results/$SLUG/speed.json'

for line in output.split('\n'):
    if not line.startswith('|') or '---' in line or 'model' in line:
        continue
    cells = [c.strip() for c in line.split('|')[1:-1]]
    if len(cells) < 7:
        continue
    test_name = cells[5] if len(cells) > 5 else ''
    ts_str = cells[6] if len(cells) > 6 else ''
    m = re.match(r'([\d.]+)\s*±\s*([\d.]+)', ts_str)
    if m and 'pp' in test_name:
        d = json.load(open(f))
        d[test_name] = {'tokens_per_sec': float(m.group(1)), 'stddev': float(m.group(2))}
        json.dump(d, open(f, 'w'), indent=2)
        print(f'{m.group(1)} ± {m.group(2)} t/s')
        sys.exit(0)

print('no data parsed')
" || echo "parse failed"
  done

  echo "  ✓ $NAME extended ctx done at $(date '+%H:%M:%S')"
done

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  All done — $(date '+%Y-%m-%d %H:%M')              ║"
echo "╚══════════════════════════════════════════════╝"
