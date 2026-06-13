#!/usr/bin/env bash
# run_specdecode_leg.sh <target_slug> <model> <leg> [num_spec]
# ONE leg of the t036 vLLM spec-decode three-way, on capsule. Donald (llama-server) must
# already be stopped (this script does NOT touch Donald — the operator owns that, so a
# failed leg never leaves the agent down). Mirrors bench_mtp_workload.sh's slate-gate
# discipline, but for `vllm serve` instead of llama-server.
#
#   leg ∈ {baseline, mtp, eagle3, dflash}.  Writes results/<target>/specdecode-<leg>.json
#   + concurrent-<leg>.json. Launches vLLM in tmux session "vllm", kills it at the end.
set -uo pipefail

TARGET="${1:?target slug}"; MODEL="${2:?model id/path}"; LEG="${3:?leg}"; NSPEC_IN="${4:-}"
HOST=127.0.0.1; PORT=8000; BASE="http://$HOST:$PORT/v1"
VLLM="$HOME/vllm-env/bin/vllm"
PY="$HOME/benchmark-rig/.venv/bin/python"
OUTDIR="$HOME/benchmark-rig/results/$TARGET"; mkdir -p "$OUTDIR"
LOG="$HOME/vllm-$LEG.log"
LAUNCH="/tmp/vllm_launch_$LEG.sh"

# Drafter repos default to gemma-4-26B-A4B; override via env for other targets (e.g. 31B). num_spec via arg 4.
MTP_DRAFTER="${MTP_DRAFTER:-google/gemma-4-26B-A4B-it-assistant}"
EAGLE3_DRAFTER="${EAGLE3_DRAFTER:-RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3}"
DFLASH_DRAFTER="${DFLASH_DRAFTER:-z-lab/gemma-4-26B-A4B-it-DFlash}"
case "$LEG" in
  baseline) SPEC=""; METHOD=""; NSPEC="" ;;
  mtp)    NSPEC="${NSPEC_IN:-4}";  METHOD=mtp;    SPEC="{\"method\":\"mtp\",\"model\":\"$MTP_DRAFTER\",\"num_speculative_tokens\":$NSPEC}" ;;
  eagle3) NSPEC="${NSPEC_IN:-3}";  METHOD=eagle3; SPEC="{\"method\":\"eagle3\",\"model\":\"$EAGLE3_DRAFTER\",\"num_speculative_tokens\":$NSPEC}" ;;
  # FLEX_ATTENTION (not flash_attn): gemma-4 is multimodal (target needs TRITON_ATTN) AND DFlash's
  # block draft needs non-causal — only FLEX_ATTENTION's arbitrary-mask backend satisfies both (#42068).
  dflash) NSPEC="${NSPEC_IN:-15}"; METHOD=dflash; SPEC="{\"method\":\"dflash\",\"model\":\"$DFLASH_DRAFTER\",\"num_speculative_tokens\":$NSPEC,\"attention_backend\":\"FLEX_ATTENTION\"}" ;;
  *) echo "[leg] unknown leg '$LEG' (want baseline|mtp|eagle3|dflash)"; exit 2 ;;
esac

cleanup(){ tmux kill-session -t vllm 2>/dev/null || true; }
trap cleanup EXIT INT TERM

slate_clear(){  # port free AND VRAM drained — Donald is down, so expect near-0
  for _ in $(seq 1 60); do
    if ! curl -s -m2 "http://$HOST:$PORT/health" >/dev/null 2>&1; then
      local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
      [ "${u:-999999}" -lt 2000 ] && return 0
    fi
    sleep 2
  done
  echo "[leg:$LEG] slate never cleared (port busy or VRAM held — is Donald still up?)"; return 1
}

# Build the launcher in a file so the speculative-config JSON survives shell quoting cleanly.
{
  # sm_120 env: give flashinfer the arch explicitly + skip its sampler JIT (greedy temp=0) + force the
  # NVFP4 GEMM onto MARLIN (flashinfer's FP4 kernel won't compile for sm_120; MARLIN is the consumer path).
  printf 'exec env TORCH_CUDA_ARCH_LIST=12.0 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NVFP4_GEMM_BACKEND=marlin %q serve %q --host %s --port %s --max-model-len %s ' "$VLLM" "$MODEL" "$HOST" "$PORT" "${MAXLEN:-8192}"
  printf -- '--gpu-memory-utilization %s --max-num-batched-tokens 8192 --trust-remote-code ' "${GPUUTIL:-0.92}"
  [ -n "$SPEC" ] && printf -- "--speculative-config '%s' " "$SPEC"
  printf '\n'
} > "$LAUNCH"

cleanup
slate_clear || exit 1
: > "$LOG"   # truncate any stale log first, else the error-grep can false-positive on a prior run
echo "[leg:$LEG] launching: $(cat "$LAUNCH")"
tmux new -d -s vllm "bash $LAUNCH > $LOG 2>&1"
sleep 3   # let the redirect truncate + first lines land before we start grepping

# vLLM model load + (for spec legs) drafter load can take a few minutes on sm_120.
healthy=0
for _ in $(seq 1 200); do
  if curl -s -m3 "http://$HOST:$PORT/health" >/dev/null 2>&1; then healthy=1; echo "[leg:$LEG] healthy"; break; fi
  if grep -qE "Traceback \(most recent call last\)|error: unrecognized arguments|RuntimeError|ValueError:|AssertionError|CUDA (error|out of memory)|OutOfMemoryError|EngineDeadError|EngineCore.*(failed|died)|not valid for this configuration" "$LOG" 2>/dev/null; then
    echo "[leg:$LEG] LAUNCH ERROR:"; tail -25 "$LOG"; exit 1
  fi
  sleep 3
done
[ "$healthy" = 1 ] || { echo "[leg:$LEG] never healthy; tail:"; tail -25 "$LOG"; exit 1; }

echo "[leg:$LEG] benching (workload spread + acceptance)..."
"$PY" "$HOME/benchmark-rig/scripts/bench_specdecode.py" --base "$BASE" --model "$MODEL" \
  --target "$TARGET" --label "$LEG" ${METHOD:+--method "$METHOD"} ${NSPEC:+--num-spec "$NSPEC"} \
  --reps "${REPS:-3}" --out "$OUTDIR/specdecode-$LEG.json" || echo "[leg:$LEG] WARN bench_specdecode nonzero"

echo "[leg:$LEG] concurrency sweep..."
"$PY" "$HOME/benchmark-rig/scripts/bench_concurrent.py" --base "http://$HOST:$PORT" --model "$MODEL" \
  --levels 1,4,8,16 --out "$OUTDIR/concurrent-$LEG.json" || echo "[leg:$LEG] WARN bench_concurrent nonzero"

echo "[leg:$LEG] DONE -> $OUTDIR/specdecode-$LEG.json"
