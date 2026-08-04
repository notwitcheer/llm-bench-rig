#!/bin/bash
# t090: run the whole quant ladder. NEEDS THE GPU — Donald must be down first.
#
# Server flags are identical for every rung. The only variable is the GGUF file:
#   -c 8192           context budget, held constant (Q8_0 is 27.1 GB of 32.6 GB,
#                     so the KV budget is the tight resource and must not float)
#   -ctk/-ctv f16     KV cache dtype pinned, so KV quant never confounds weight quant
#   -ngl 999          fully resident, no offload confound (the point of the 5090)
#   --no-cont-batching batch 1, single consumer, which is the consumer case measured
#
# Run (capsule): bash scripts/quant_tax/run_ladder.sh
set -euo pipefail

RIG=/home/witcheer/benchmark-rig
GGUF=/home/witcheer/t090/gguf
BIN=/home/witcheer/t090/llama.cpp/build/bin
PY=/home/witcheer/sglang-env/bin/python
PORT=8090
LOGS=/home/witcheer/t090/logs
LADDER="${LADDER:-Q8_0 Q6_K Q5_K_M Q4_K_M Q3_K_M}"
TASKS="${TASKS:-gsm8k mmlu humaneval}"

mkdir -p "$LOGS"
cd "$RIG"
export PYTHONPATH="$RIG"

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$used" -gt 2000 ]; then
  echo "ABORT: ${used} MiB already resident on the GPU."
  echo "The ladder needs the card to itself. Stop Donald first, then re-run."
  exit 1
fi

for Q in $LADDER; do
  F="$GGUF/Qwen3.6-27B-$Q.gguf"
  [ -f "$F" ] || { echo "missing $F"; exit 1; }

  echo "=== $Q: serving $(date -Is) ==="
  "$BIN/llama-server" -m "$F" --port "$PORT" \
    -c 8192 -ngl 999 -ctk f16 -ctv f16 --no-cont-batching \
    > "$LOGS/server-$Q.log" 2>&1 &
  SRV=$!
  trap 'kill $SRV 2>/dev/null || true' EXIT

  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
    sleep 5
  done
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "$Q: server never came up"; exit 1; }

  echo "=== $Q: speed ==="
  $PY scripts/quant_tax/speed.py --quant "$Q" --port "$PORT"

  for T in $TASKS; do
    echo "=== $Q: $T ==="
    $PY scripts/quant_tax/gen.py --task "$T" --quant "$Q" --port "$PORT"
  done

  kill $SRV 2>/dev/null || true
  wait $SRV 2>/dev/null || true
  trap - EXIT
  sleep 5

  echo "=== $Q: perplexity ==="
  "$BIN/llama-perplexity" -m "$F" -f dataset/quant_tax/wikitext2_test.txt \
    -c 4096 -ngl 999 > "$LOGS/ppl-$Q.log" 2>&1 || echo "  ppl failed for $Q, see $LOGS/ppl-$Q.log"
  grep -E "^Final estimate|PPL" "$LOGS/ppl-$Q.log" | tail -2 || true
done

echo "=== grading $(date -Is) ==="
$PY scripts/quant_tax/grade.py results/quant_tax/*.gens.json

echo "T090-LADDER-EVAL-DONE $(date -Is)"
