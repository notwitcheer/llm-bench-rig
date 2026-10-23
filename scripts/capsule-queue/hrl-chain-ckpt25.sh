#!/usr/bin/env bash
# same 25 test tasks, the three FineEnvs checkpoints, inside Hermes v2026.9.24. ~10 min each.
LOG=~/harness-rl/chain-ckpt25.log; exec >>"$LOG" 2>&1
echo "== chain start $(date)"
for spec in mhrl:FineEnvs/LFM2.5-2.6B-multiharness-RL ocrl:FineEnvs/LFM2.5-2.6B-opencode-RL mhsft:FineEnvs/LFM2.5-2.6B-multiharness-SFT; do
  [ "$(date +%H%M)" -lt 2330 ] || { echo "too late, stopping before ${spec%%:*}"; break; }
  N=25 C=6 TAG=${spec%%:*} CKPT=${spec#*:} ~/harness-rl/hrl-probe.sh
  echo "-- ${spec%%:*} done $(date)"
done
echo "== chain done $(date)"
