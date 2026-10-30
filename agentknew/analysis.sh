#!/bin/bash
# nerve-gate analysis (CPU only): paper's 154 fit (dev) + 155 v57 analysis (fresh conf) for gpt-oss-20b, then compare to the paper.
set -uo pipefail
R=$HOME/esoteric/Honesty; V=$HOME/esoteric/.venv; W=$HOME/esoteric/nerve-gate
cd "$R"; export PYTHONPATH=$R/src HF_HUB_OFFLINE=1
mkdir -p $W/paper; [ -f $W/paper/scores_v57.parquet ] || cp outputs/v57/scores.parquet $W/paper/scores_v57.parquet
echo "[analysis] $(date '+%F %H:%M:%S') fit"
$V/bin/python scripts/154_v54_gate.py stage=fit actor=gptoss20b > $W/fit.log 2>&1 || { echo "FIT FAILED"; tail -20 $W/fit.log; exit 1; }
tail -3 $W/fit.log
echo "[analysis] $(date '+%F %H:%M:%S') v57"
$V/bin/python scripts/155_v57_analysis.py "models=[gptoss20b]" B=2000 > $W/v57.log 2>&1 || { echo "V57 FAILED"; tail -30 $W/v57.log; exit 1; }
cp results/prereg_v57/summary.json $W/ours_v57_summary.json; cp results/prereg_v54/gate_fit_gptoss20b.json $W/ours_gate_fit.json
cp outputs/v57/scores.parquet $W/ours_scores_v57.parquet; cp outputs/v54/gate_gptoss20b.npz $W/ours_gate_gptoss20b.npz
git checkout -- results/ 2>/dev/null; cp $W/paper/scores_v57.parquet outputs/v57/scores.parquet
$V/bin/python $W/compare.py
echo "[analysis] $(date '+%F %H:%M:%S') ANALYSIS DONE"
