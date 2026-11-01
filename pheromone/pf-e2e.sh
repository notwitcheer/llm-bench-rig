#!/bin/bash
# pf-e2e.sh: forced short end-to-end test of 05-pheromone-pf.sh, outside HC (< 15 min card time), staged 2026-10-30 by mercury.
# waits until NOT_BEFORE (17:15 CET, after the 14:10-17:10 block), refuses while the idle queue or any GPU process is live, then runs 5 cells x 1 trial.
NOT_BEFORE=1793376900
LOG=/home/witcheer/esoteric/pf-e2e.log
while [ "$(date +%s)" -lt "$NOT_BEFORE" ]; do sleep 30; done
for i in $(seq 1 30); do
  if tmux has-session -t idleq 2>/dev/null || [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then sleep 30; else break; fi
done
if tmux has-session -t idleq 2>/dev/null || [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then echo "E2E ABORT: card still owned at $(date +%H:%M)" >> $LOG; exit 2; fi
rm -rf /home/witcheer/esoteric/pheromone-e2e
echo "E2E START $(date +%H:%M:%S)" >> $LOG
PH_RES=/home/witcheer/esoteric/pheromone-e2e PH_MANIFEST=/tmp/pheromone-e2e.json PH_RUNLOG=/home/witcheer/esoteric/pheromone-e2e-run.log \
PH_CELLS="pe-pf4 me-pf4 me-cv4 me-cvM pe-decay" PH_TRIALS=1 PH_WIN_START=0 PH_WIN_END=$(( $(date +%s) + 1500 )) \
  bash /home/witcheer/idle-queue/05-pheromone-pf.sh >> $LOG 2>&1
echo "E2E RC=$? $(date +%H:%M:%S) card=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader) donald=$(curl -s localhost:8090/health)" >> $LOG
