#!/bin/bash
# waits for the replay to end (REPLAY DONE or tmux gone), then runs the CPU analysis. tmux: nerve-watch
W=$HOME/esoteric/nerve-gate
while tmux has-session -t nerve-replay 2>/dev/null; do sleep 60; done
grep -q "REPLAY DONE" $W/replay.log || echo "[watch] replay ended WITHOUT 'REPLAY DONE' (hard stop or crash); analysing what exists" >> $W/analysis.log
bash $W/analysis.sh >> $W/analysis.log 2>&1
echo "[watch] $(date '+%F %H:%M:%S') rc=$?" >> $W/analysis.log
