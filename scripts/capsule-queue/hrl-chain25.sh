#!/usr/bin/env bash
# wait for the hermes bake, point the 250 test task Dockerfiles at it (backup kept), run 25 base tasks.
# guard: never start after 23:30 CEST (bench 59 owns the GPU from 01:10).
LOG=~/harness-rl/chain25.log; exec >>"$LOG" 2>&1
echo "== chain start $(date)"
until grep -q BAKE_RC= ~/harness-rl/bake.log; do sleep 30; done
grep -q BAKE_RC=0 ~/harness-rl/bake.log || { echo "bake failed"; tail -20 ~/harness-rl/bake.log; exit 1; }
docker run --rm hrl/env-data-agent-hermes:2026.9.24 bash -c 'export PATH="$HOME/.local/bin:$PATH"; hermes --version' || { echo "baked hermes missing"; exit 1; }
[ "$(date +%H%M)" -lt 2330 ] || { echo "too late, not starting"; exit 0; }
cd ~/harness-rl/FineEnvs/05-multi-harness-rl/prepared/datasets/test/tasks
n=0; for d in */; do f="$d/environment/Dockerfile"; [ -f "$f.orig" ] || cp "$f" "$f.orig"
  printf 'FROM hrl/env-data-agent-hermes:2026.9.24\nWORKDIR /workdir\n' > "$f"; n=$((n+1)); done
echo "rewrote $n Dockerfiles"
N=25 C=6 TAG=base ~/harness-rl/hrl-probe.sh
echo "== chain done $(date)"
