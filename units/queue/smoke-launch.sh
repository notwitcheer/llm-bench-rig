#!/bin/bash
# GPU smoke for 88-units-orbit (rule-3 check done by the launcher at 10:03 CET): 3 tasks x 4 forms on Qwen3.5-9B Q4 + a 5-per-view bridge check. hard stop = now + 24 min (< 30 min, outside HC allowed).
export NL_HARD_STOP_EPOCH=$(( $(date +%s) + 24*60 ))
export UNITS_SMOKE=1 UNITS_MODELS=qwen35-9b-q4km UNITS_TASKS=t01,p01,u01 UNITS_BRIDGE=1 UNITS_BRIDGE_N=5 UNITS_NO_SELFTEST=1
DRYRUN=1 bash ~/voxel-stage/88-units-orbit.sh && bash ~/voxel-stage/88-units-orbit.sh
echo "SMOKE_RC=$?"
