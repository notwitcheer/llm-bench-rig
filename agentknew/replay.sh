#!/bin/bash
# nerve-gate replay: the paper's own scripts/56 extractor on gpt-oss-20b (MXFP4, 5090), conf sets first, then dev sets,
# resumable (skips a set whose features exist), hard stop before the 01:10 idle-queue block.
# EV: "my GPU felt the agent decide to break the rules before it wrote the call": our held-out AUROC for gpt-oss-20b vs the
#     paper's 0.884 (results/prereg_v57/summary.json per_model.gptoss20b.all.I). protects any nerve-gate post; esoteric lane.
set -uo pipefail
source "$HOME/night-lib.sh"; NL_TAG=nerve-replay
R=$HOME/esoteric/Honesty; V=$HOME/esoteric/.venv; W=$HOME/esoteric/nerve-gate
STOP_EPOCH=${STOP_EPOCH:-$(date -d "tomorrow 00:50" +%s)}; [ "$(date +%H)" -lt 1 ] && STOP_EPOCH=$(date -d "today 00:50" +%s)
log(){ echo "[replay] $(date '+%F %H:%M:%S') $*"; }
if tmux ls 2>/dev/null | grep -E '^(idleq|vx-|flint|prz|hm)' ; then log "ABORT sibling session live"; exit 10; fi
apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -v -i llama || true)
[ -n "$apps" ] && { log "ABORT foreign GPU pid: $apps"; exit 10; }
# set|bench|extra args  (conf x- sets restricted to the v57 conf episodes, as slurm/v57.sh)
JOBS=(
 "xm_gptoss20b_core24b_minus6789|core24b|episodes=outputs/v57/conf_episodes.parquet"
 "xm_gptoss20b_core24b_minus10111213|core24b|episodes=outputs/v57/conf_episodes.parquet"
 "xm_gptoss20b_core24b_reasoning_minus6789|core24b_reasoning|episodes=outputs/v57/conf_episodes.parquet"
 "xm_gptoss20b_core24b_obfuscate_minus10111213|core24b_obfuscate|episodes=outputs/v57/conf_episodes.parquet"
 "xm_gptoss20b_core24b_plus|core24b|"
 "xm_gptoss20b_core24b_reasoning_plus|core24b_reasoning|"
 "xm_gptoss20b_core24b_obfuscate_plus|core24b_obfuscate|"
 "xm_gptoss20b_core24b_minus|core24b|"
 "xm_gptoss20b_core24b_reasoning_minus|core24b_reasoning|"
 "xm_gptoss20b_core24b_obfuscate_minus|core24b_obfuscate|"
 "xm_gptoss20b_core24_minus|core24|"
 "xm_gptoss20b_core24_plus|core24|"
 "xm_gptoss20b_v2fresh_minus|v2fresh|"
 "xm_gptoss20b_v2fresh_plus|v2fresh|"
 "xm_gptoss20b_v2fresh_plus23|v2fresh|"
 "xm_gptoss20b_core24b_minus2345|core24b|"
 "xm_gptoss20b_core24b_reasoning_minus2345|core24b_reasoning|"
 "xm_gptoss20b_core24b_obfuscate_minus2345|core24b_obfuscate|"
)
if [ "${DRYRUN:-0}" = "1" ]; then
  for j in "${JOBS[@]}"; do IFS='|' read -r st b ex <<<"$j"; [ -f "$R/outputs/rollouts/$st.parquet" ] && [ -d "$R/outputs/capebench/$b" ] && echo "ok $st ($b)" || echo "MISSING $st ($b)"; done
  log "DRYRUN stop at $(date -d @$STOP_EPOCH '+%F %H:%M')"; exit 0
fi
trap 'nl_donald_restore; log "donald restored"' EXIT
nl_donald_stop
for i in $(seq 1 24); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$used" -lt 1024 ] && break; sleep 5; done
[ "$used" -ge 1024 ] && { log "ABORT VRAM never drained (${used} MiB)"; exit 11; }
cd "$R"
export PYTHONPATH=$R/src HF_HUB_OFFLINE=0 HF_HUB_CACHE=$HOME/.cache/huggingface/hub PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for j in "${JOBS[@]}"; do
  IFS='|' read -r st b ex <<<"$j"
  [ -f "outputs/xm_feats/${st}_meta.parquet" ] && { log "skip $st (done)"; continue; }
  left=$(( STOP_EPOCH - $(date +%s) )); [ $left -lt 300 ] && { log "HARD STOP reached before $st"; break; }
  log "start $st (bench $b) ${left}s left"
  timeout ${left}s "$V/bin/python" scripts/56_xm_replay_extract.py actor=gptoss20b bench=$b max_tokens=6000 "sets=[$st]" $ex
  log "end $st rc=$?"
done
log "REPLAY DONE"; ls outputs/xm_feats/ | grep -c _meta.parquet
