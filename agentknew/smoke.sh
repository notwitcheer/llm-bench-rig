#!/bin/bash
# nerve-gate GPU smoke (<5 min card time): does gpt-oss-20b load as MXFP4 on the 5090 through the paper's own loader,
# and what do a 5-layer residual capture + a 1.6k-token prefill cost? Guarded: aborts if the card is not ours.
# EV: protects a possible "The Agent Knew" reproduction post.
set -uo pipefail
source "$HOME/night-lib.sh"; NL_TAG=nerve-smoke
W=$HOME/esoteric/nerve-gate; V=$HOME/esoteric/.venv
log(){ echo "[smoke] $(date +%H:%M:%S) $*"; }
# ownership: no sibling run sessions, no non-Donald GPU pid, not within 30 min of an HC block start
if tmux ls 2>/dev/null | grep -E '^(idleq|vx-|flint|prz|hm)' ; then log "ABORT sibling session live"; exit 10; fi
apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -v -i llama || true)
[ -n "$apps" ] && { log "ABORT foreign GPU pid: $apps"; exit 10; }
hm=$(date +%H%M); if { [ "$hm" -ge 0040 ] && [ "$hm" -lt 0115 ]; } || { [ "$hm" -ge 1340 ] && [ "$hm" -lt 1415 ]; }; then log "ABORT too close to an HC block start"; exit 10; fi
[ "${DRYRUN:-0}" = "1" ] && { log "DRYRUN ok: guards pass"; exit 0; }

trap 'nl_donald_restore; log "donald restored"' EXIT
nl_donald_stop
for i in $(seq 1 24); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$used" -lt 1024 ] && break; sleep 5; done
[ "$used" -ge 1024 ] && { log "ABORT VRAM never drained (${used} MiB)"; exit 11; }
log "card drained (${used} MiB), running smoke"
cd "$W" && HF_HUB_OFFLINE=0 timeout 20m "$V/bin/python" "$W/smoke.py"; rc=$?
log "smoke rc=$rc"; exit $rc
