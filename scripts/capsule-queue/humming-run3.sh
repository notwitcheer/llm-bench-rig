#!/usr/bin/env bash
# Humming vs Marlin (and the stock SM120 defaults) on one RTX 5090, stock PyPI vLLM 0.32.0 + humming-kernels 0.1.17.
# same checkpoint per pair, only the kernel backend flag differs. short daytime legs.
# chase run 3 (second server start of the 0.1.17 Humming legs + NVFP4 cutlass), waits for tmux hmrun2 to end.
# booking gate: ~/sparkagent-DONE (or no ~/sparkagent-BUSY after 11:30), no vx-*/voxel-*/prz*/dgz* tmux, no idleq,
# no GPU compute app except Donald's llama-server. no leg starts at/after 13:30 CEST, everything killed at 13:55 CEST.
# safety (2026-10-21 OOM): MAX_JOBS=4, humming build workers 4, nice 19, RAM watchdog kills vLLM under 6 GiB available.
set -u
GATE_FROM_EPOCH=1792661400     # 11:30 CEST: BUSY-absent fallback
LAST_LEG_EPOCH=1792668600      # 13:30 CEST
HARD_STOP_EPOCH=1792670100     # 13:55 CEST
ENV=/home/witcheer/humming-env
OUT=/home/witcheer/humming-bench
CLIENT=/home/witcheer/hm_client.py
PORT=8013
LEGS="${LEGS:-moe_humming2 awq_humming2 nv_humming2 nv_cutlass2}"
ENV18=/home/witcheer/humming018-env   # same vLLM 0.32.0, humming-kernels overridden to 0.1.18 (SM120 grid_m=1 evict_first, humming#151)
M_AWQ=/home/witcheer/models/qwen3.6-27b-awq
M_MOE=/home/witcheer/models/ornith-15-35b-nvfp4
M_NV=/home/witcheer/models/qwen38-27b-nvfp4
mkdir -p "$OUT"
say(){ echo "[hm] $(date +%T) $*"; }
now(){ date +%s; }

for p in "$ENV18/bin/python" "$ENV/bin/python" "$CLIENT" "$M_AWQ/config.json" "$M_MOE/config.json" "$M_NV/config.json"; do
  [ -e "$p" ] || { say "ABORT missing $p"; exit 2; }
done
HELP=$("$ENV/bin/python" -m vllm.entrypoints.cli.main serve --help=all 2>&1 || true)
for want in "--linear-backend" "--moe-backend" "--no-enable-prefix-caching"; do
  case "$HELP" in *"$want"*) ;; *) say "ABORT flag missing $want"; exit 2;; esac
done
say "preflight ok; legs: $LEGS"
[ "${DRYRUN:-0}" = "1" ] && { say "DRYRUN done"; exit 0; }

if [ "${DRYRUN:-0}" != "1" ]; then while tmux has-session -t =hmrun2 2>/dev/null; do sleep 20; done; sleep 20; fi
gate(){
  local why=""
  if [ -f /home/witcheer/sparkagent-DONE ]; then :;
  elif [ -f /home/witcheer/sparkagent-BUSY ]; then why="sparkagent BUSY";
  elif [ "$(now)" -lt "$GATE_FROM_EPOCH" ]; then why="no sparkagent marker before 11:30"; fi
  local t; t=$(tmux ls -F '#{session_name}' 2>/dev/null | grep -E '^(vx-|voxel-|prz|dgz|idleq$)' | tr '\n' ' ')
  [ -n "$t" ] && why="$why tmux:[$t]"
  local apps; apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -v 'llama.cpp-donald' | tr '\n' ' ')
  [ -n "$apps" ] && why="$why gpu:[$apps]"
  echo "$why"
}
last=""
while :; do
  w=$(gate)
  [ -z "$w" ] && break
  [ "$(now)" -ge "$LAST_LEG_EPOCH" ] && { say "ABORT: card never free before 13:30 ($w)"; exit 4; }
  [ "$w" != "$last" ] && say "waiting: $w"; last="$w"
  sleep 30
done
say "gate open"

export FLASHINFER_CUDA_ARCH_LIST=12.0f CUDA_HOME=/usr/local/cuda-13
export PATH="$ENV/bin:/usr/local/cuda-13/bin:$PATH"
export MAX_JOBS=4 NVCC_THREADS=1 HUMMING_MAX_PARALLEL_BUILD_WORKERS=4 HUMMING_CACHE_DIR=/home/witcheer/humming-cache
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
SRV=""; WD=""
restore(){
  [ -n "$WD" ] && kill "$WD" 2>/dev/null
  [ -n "$SRV" ] && kill "$SRV" 2>/dev/null; sleep 5; [ -n "$SRV" ] && kill -9 "$SRV" 2>/dev/null
  pkill -9 -f "vllm.entrypoints.cli.main serve" 2>/dev/null; pkill -9 -f "VLLM::EngineCore" 2>/dev/null; pkill -9 -x "VLLM::EngineCor" 2>/dev/null
  sleep 3
  sudo -n systemctl start llama-server.service
  for i in $(seq 1 30); do curl -s -m 3 localhost:8090/health | grep -qi ok && break; sleep 2; done
  say "EXIT: Donald health=$(curl -s -m 3 localhost:8090/health) VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  say "HM_DONE"
}
trap restore EXIT
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tee "$OUT/apps-preflight.txt"
say "drain Donald"
sudo -n systemctl stop llama-server.service || { say "ABORT: cannot stop llama-server"; exit 2; }
for i in $(seq 1 45); do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '); [ "$U" -lt 2000 ] && break; sleep 2; done
say "VRAM after drain ${U} MiB"
[ "$U" -lt 2000 ] || { say "ABORT: VRAM not free after drain"; exit 2; }

# watchdog: RAM floor + hard stop. logs RAM/VRAM every 5 s.
(
  while :; do
    A=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    V=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    echo "$(date +%T) avail_mib=$A vram_mib=$V" >> "$OUT/watch.log"
    if [ "$A" -lt 6144 ]; then echo "$(date +%T) RAM FLOOR HIT avail=$A, killing vllm" >> "$OUT/watch.log"; pkill -9 -f "vllm.entrypoints.cli.main serve"; pkill -9 -f "VLLM::EngineCore"; pkill -9 -x "VLLM::EngineCor"; fi
    if [ "$(date +%s)" -ge "$HARD_STOP_EPOCH" ]; then echo "$(date +%T) HARD STOP 13:55" >> "$OUT/watch.log"; pkill -9 -f "vllm.entrypoints.cli.main serve"; pkill -9 -f "VLLM::EngineCore"; pkill -9 -x "VLLM::EngineCor"; pkill -f hm_client.py; fi
    sleep 5
  done
) &
WD=$!

serve(){ # leg model extra-args...
  local leg=$1 model=$2; shift 2
  SENV=$ENV; export HUMMING_CACHE_DIR=/home/witcheer/humming-cache
  case $leg in h18_*) SENV=$ENV18; export HUMMING_CACHE_DIR=/home/witcheer/humming-cache-018;; esac
  say "env $SENV humming=$("$SENV/bin/python" -c "import importlib.metadata as m; print(m.version('humming-kernels'))")"
  local log="$OUT/serve-$leg.log"
  say "serve $leg: $(basename $model) $*"
  local t0; t0=$(now)
  nice -n 19 "$SENV/bin/python" -m vllm.entrypoints.cli.main serve "$model" --served-model-name hm \
    --max-model-len 4096 --max-num-seqs 16 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.92 \
    --seed 721 --host 127.0.0.1 --port $PORT --no-enable-prefix-caching --trust-remote-code \
    --language-model-only "$@" > "$log" 2>&1 &
  SRV=$!
  for i in $(seq 1 180); do
    if ! kill -0 $SRV 2>/dev/null; then say "serve $leg DIED after $(( $(now)-t0 ))s"; grep -a -E "Error|error|Traceback" "$log" | tail -12; SRV=""; return 1; fi
    if curl -s -m 3 localhost:$PORT/v1/models | grep -q '"hm"'; then
      say "serve $leg up after $(( $(now)-t0 ))s VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader) RAMavail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)G"
      echo "load_s=$(( $(now)-t0 ))" > "$OUT/load-$leg.txt"
      grep -a -i -E "Using .*(kernel|backend)|humming|marlin|NvFp4|MoE backend|linear backend|falling back" "$log" | sed 's/^.*\] //' | sort -u | head -25 > "$OUT/kernels-$leg.txt"
      cat "$OUT/kernels-$leg.txt"
      return 0
    fi
    [ "$(now)" -ge "$HARD_STOP_EPOCH" ] && { say "hard stop during load"; return 1; }
    sleep 5
  done
  say "serve $leg TIMEOUT"; tail -25 "$log"; return 1
}
killall_vllm(){ pkill -9 -f "vllm.entrypoints.cli.main serve" 2>/dev/null; pkill -9 -f "VLLM::EngineCore" 2>/dev/null; pkill -9 -x "VLLM::EngineCor" 2>/dev/null; }
stop_srv(){ [ -n "$SRV" ] && kill $SRV 2>/dev/null; for i in $(seq 1 30); do kill -0 $SRV 2>/dev/null || break; sleep 2; done; kill -9 $SRV 2>/dev/null; SRV=""; sleep 3; killall_vllm; sleep 3
  for i in $(seq 1 30); do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '); [ "$U" -lt 2000 ] && break; sleep 2; done; say "VRAM after stop ${U} MiB"; }
bench(){ # leg
  nvidia-smi --query-gpu=memory.used --format=csv,noheader > "$OUT/vram-$1.txt"
  timeout 900 "$ENV/bin/python" "$CLIENT" $PORT "$1" "$OUT/$1.json" 2>&1 | tail -20
  say "bench $1 rc=${PIPESTATUS[0]}"
}
for leg in $LEGS; do
  [ "$(now)" -ge "$LAST_LEG_EPOCH" ] && { say "SKIP $leg: past 13:30"; continue; }
  case $leg in
    h18_awq_humming) serve $leg "$M_AWQ" --linear-backend humming && bench $leg ;;
    h18_moe_humming) serve $leg "$M_MOE" --linear-backend cutlass --moe-backend humming && bench $leg ;;
    h18_nv_humming)  serve $leg "$M_NV" --linear-backend humming && bench $leg ;;
    awq_marlin|awq_marlin2) serve $leg "$M_AWQ" && bench $leg ;;
    awq_humming|awq_humming2) serve $leg "$M_AWQ" --linear-backend humming && bench $leg ;;
    moe_marlin)  serve $leg "$M_MOE" --linear-backend cutlass --moe-backend marlin && bench $leg ;;
    moe_humming|moe_humming2) serve $leg "$M_MOE" --linear-backend cutlass --moe-backend humming && bench $leg ;;
    moe_auto)    serve $leg "$M_MOE" --linear-backend cutlass && bench $leg ;;
    nv_cutlass|nv_cutlass2)  serve $leg "$M_NV" --linear-backend cutlass && bench $leg ;;
    nv_humming|nv_humming2)  serve $leg "$M_NV" --linear-backend humming && bench $leg ;;
    nv_marlin)   serve $leg "$M_NV" --linear-backend marlin && bench $leg ;;
  esac
  stop_srv
done
say "all legs walked"
