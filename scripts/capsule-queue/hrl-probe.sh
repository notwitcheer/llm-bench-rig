#!/usr/bin/env bash
# hermes-harness transfer probe (personal GPU lane, opened 2026-10-05).
# Serves one LFM2.5-2.6B checkpoint on vLLM (FineEnvs serve_model.py flags, DP1), starts the
# SmolDataEnvs Harbor server with a docker sandbox + direct capture, then runs the FineEnvs
# evaluator with the hermes harness only. Drains Donald first, restores him on every exit path.
#
#   N=3 C=3 TAG=base            ~/harness-rl/hrl-probe.sh
#   N=250 C=8 TAG=mhrl CKPT=FineEnvs/LFM2.5-2.6B-multiharness-RL ~/harness-rl/hrl-probe.sh
#
# short probes are allowed any time; full 250-task legs belong in heures-creuses blocks.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd ~/harness-rl/FineEnvs/05-multi-harness-rl || exit 1
source .venv/bin/activate
N=${N:-3}; C=${C:-3}; TAG=${TAG:-base}; CKPT=${CKPT:-}
BASE=LiquidAI/LFM2.5-2.6B; REV=654f9463ce32b05d0429d76fe1f580b27d4c1ac0
HERMES_TAG=${HERMES_TAG:-2026.9.24}  # numeric: OpenEnv requires it; patched adapter adds the v
OUT=${OUT:-$HOME/harness-rl/runs/${TAG}-n${N}}; mkdir -p "$OUT"
exec > >(tee -a "$OUT/probe.log") 2>&1
echo "== probe start $(date) tag=$TAG n=$N c=$C ckpt=${CKPT:-base@$REV} hermes=$HERMES_TAG"
VPORT=8000; SPORT=8200; CPORT=8100
# task.toml forwards ${HF_TOKEN} into the sandbox, where the agent runs with --yolo. the data bucket
# (AdithyaSK/jupyter-agent-kaggle-all) is public, so pass an EMPTY token: anonymous pull, no secret in reach.
export HF_TOKEN=""
PIDS=()
restore() {
  for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done
  sleep 10
  for p in "${PIDS[@]}"; do kill -9 "$p" 2>/dev/null; done
  sudo -n systemctl start llama-server.service; echo "== Donald restore rc=$? $(date)"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
}
trap restore EXIT
sudo -n systemctl stop llama-server.service || { echo "drain failed"; exit 1; }
sleep 3

python - "$OUT" "$BASE" <<'PY'
import sys
sys.path.insert(0, ".")
from train.whitebox import tokenizer_for
open(sys.argv[1] + "/chat_template.jinja", "w").write(tokenizer_for(sys.argv[2]).chat_template)
PY

MODEL_ARGS=("$BASE" --revision "$REV")
[ -n "$CKPT" ] && MODEL_ARGS=("$CKPT" --safetensors-load-strategy eager)
VLLM_USE_DEEP_GEMM=0 VLLM_DEEP_GEMM_WARMUP=skip VLLM_USE_FLASHINFER_SAMPLER=0 \
vllm serve "${MODEL_ARGS[@]}" --host 127.0.0.1 --port $VPORT --served-model-name "$BASE" \
  --tensor-parallel-size 1 --data-parallel-size 1 --dtype bfloat16 --max-model-len 131072 \
  --gpu-memory-utilization 0.85 --enable-auto-tool-choice --tool-call-parser lfm2 \
  --chat-template "$OUT/chat_template.jinja" \
  --default-chat-template-kwargs '{"enable_thinking":false,"preserve_thinking":true}' \
  --generation-config vllm --logprobs-mode processed_logprobs --return-tokens-as-token-ids \
  --enforce-eager --no-enable-prefix-caching ${VLLM_EXTRA:-} > "$OUT/vllm.log" 2>&1 &
PIDS+=($!)
for i in $(seq 1 180); do curl -sf http://127.0.0.1:$VPORT/health >/dev/null && break; sleep 2; done
curl -sf http://127.0.0.1:$VPORT/health >/dev/null || { echo "vllm never healthy"; tail -30 "$OUT/vllm.log"; exit 1; }
echo "== vllm up $(date)"

OPENENV_DATASETS="$(pwd)/prepared/datasets/train,$(pwd)/prepared/datasets/test" \
OPENENV_LLM_URL=http://127.0.0.1:$VPORT OPENENV_MODEL="$BASE" \
OPENENV_HARBOR_TRIALS_DIR="$OUT/trials" OPENENV_MAX_OUTPUT_TOKENS=4096 \
OPENENV_CAPTURE_PORT=$CPORT OPENENV_EXPOSE=direct OPENENV_DIRECT_HOST=172.17.0.1 \
MAX_CONCURRENT_ENVS=$(( C + 8 )) ENABLE_WEB_INTERFACE=false \
OPENENV_HARBOR_AGENT_VERSIONS="{\"hermes\":\"$HERMES_TAG\"}" \
python -m uvicorn smoldataenv_harbor.server:app --host 127.0.0.1 --port $SPORT \
  --ws-ping-timeout 1800 > "$OUT/harbor.log" 2>&1 &
PIDS+=($!)
for i in $(seq 1 90); do curl -sf http://127.0.0.1:$SPORT/health >/dev/null && break; sleep 2; done
curl -sf http://127.0.0.1:$SPORT/health >/dev/null || { echo "harbor never healthy"; tail -30 "$OUT/harbor.log"; exit 1; }
echo "== harbor up $(date)"

HRL_HARNESSES=hermes HRL_SANDBOX=docker \
python -m eval.evaluate_hrl --mode blackbox --model "$BASE" \
  --checkpoint "${CKPT:-$BASE@$REV}" --server http://127.0.0.1:$SPORT \
  --vllm-url http://127.0.0.1:$VPORT --tasks "$N" --concurrency "$C" --output "$OUT"
echo "== eval rc=$? $(date)"
[ -f "$OUT/summary.json" ] && python -c "import json,sys;d=json.load(open('$OUT/summary.json'));print(json.dumps({k:v for k,v in d.items() if not isinstance(v,dict)},indent=1))"
