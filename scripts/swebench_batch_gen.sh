#!/usr/bin/env bash
# SWE-bench anchor — BATCH generation (phase 1) for all 7 leaderboard models over a
# pinned instance set. Donald (llama-server.service) is stopped ONCE for the whole batch
# and restored ONCE on exit (no per-model flapping). GPU-only; grading is phase 2.
#
# Usage:  ./scripts/swebench_batch_gen.sh [IDS_FILE]   (default: swebench_ids_30.txt)
# Run inside a detached tmux:  tmux new -d -s swgen "./scripts/swebench_batch_gen.sh 2>&1 | tee ~/swgen.log"
set -uo pipefail
cd ~/benchmark-rig
IDS="${1:-swebench_ids_30.txt}"
BIN=~/llama.cpp/build/bin/llama-server
PY=~/swebench-env/bin/python
[ -f "$IDS" ] || { echo "FATAL: ids file $IDS missing"; exit 2; }

# slug<TAB>abs-gguf-path — order: top synthetic score -> bottom (cosmetic only)
MODELS="
qwen3-6-27b	/home/witcheer/models/Qwen3.6-27B-Q6_K.gguf
qwopus-glm-18b	/home/witcheer/models/qwopus-glm-18b.gguf
qwen3-5-35b-base	/home/witcheer/models/qwen35-base/Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf
nemotron-cascade-2-30b	/home/witcheer/models/nemotron-cascade-2-30b.gguf
kimi-linear-48b-a3b	/home/witcheer/models/kimi-linear-48b/moonshotai_Kimi-Linear-48B-A3B-Instruct-Q4_K_M.gguf
granite-4-1-30b	/home/witcheer/models/granite-4.1-30b/granite-4.1-30b-UD-Q4_K_XL.gguf
nex-n2-mini	/home/witcheer/models/nex-n2-mini/Nex-N2-mini.Q4_K_M.gguf
"

# pre-flight: every gguf exists before we take Donald down
miss=0
while IFS=$'\t' read -r slug gguf; do
  [ -z "${slug:-}" ] && continue
  [ -f "$gguf" ] || { echo "MISSING GGUF: $slug -> $gguf"; miss=1; }
done <<< "$MODELS"
[ "$miss" = 0 ] || { echo "FATAL: missing gguf(s), aborting before touching Donald"; exit 3; }

restore() {
  echo "[batch] restoring Donald (llama-server.service)"
  systemctl is-active --quiet llama-server.service || sudo -n systemctl start llama-server.service || true
}
trap restore EXIT

echo "[batch] stopping Donald for the full batch"
sudo -n systemctl stop llama-server.service; sleep 3

while IFS=$'\t' read -r slug gguf; do
  [ -z "${slug:-}" ] && continue
  echo "==================== [$slug] $(date +%H:%M:%S) ===================="
  "$BIN" -m "$gguf" -ngl 999 -fa on --jinja -c 32768 --port 8090 --no-warmup \
    > /tmp/sweb_$slug.log 2>&1 &
  SRV=$!
  ok=0
  for i in $(seq 1 120); do curl -sf http://127.0.0.1:8090/health >/dev/null 2>&1 && { ok=1; break; }; sleep 3; done
  if [ "$ok" != 1 ]; then
    echo "[$slug] server did not come healthy — skipping (see /tmp/sweb_$slug.log)"
    kill -9 $SRV 2>/dev/null; pkill -9 -x llama-server 2>/dev/null; sleep 4; continue
  fi
  "$PY" -m lib.agentic.native.run_swebench "$slug" "$IDS" || echo "[$slug] WARN generation returned nonzero"
  kill -9 $SRV 2>/dev/null; pkill -9 -x llama-server 2>/dev/null; sleep 4
  echo "[$slug] done $(date +%H:%M:%S)  -> predictions/$slug.jsonl"
done <<< "$MODELS"

echo "SWEB_BATCH_GEN_DONE $(date +%H:%M:%S)"
