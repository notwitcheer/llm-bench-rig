#!/usr/bin/env bash
# SWE-bench anchor — BATCH grading (phase 2). Runs the OFFICIAL swebench harness over each
# model's predictions (apply patch -> FAIL_TO_PASS + PASS_TO_PASS in the instance container).
# CPU/Docker only — safe to run with Donald back up. Normalizes each run to
# results/swebench/<slug>.report.json (+ copies telemetry) for the Mac chart/report.
#
# Usage:  ./scripts/swebench_batch_grade.sh   (grades every predictions/<slug>.jsonl present)
set -uo pipefail
cd ~/benchmark-rig
PY=~/swebench-env/bin/python
DATASET=princeton-nlp/SWE-bench_Verified
WORKERS="${WORKERS:-6}"
mkdir -p results/swebench

SLUGS="qwen3-6-27b qwopus-glm-18b qwen3-5-35b-base nemotron-cascade-2-30b kimi-linear-48b-a3b granite-4-1-30b nex-n2-mini"

for slug in $SLUGS; do
  pred="predictions/$slug.jsonl"
  [ -f "$pred" ] || { echo "[$slug] no predictions, skip"; continue; }
  echo "==================== grade [$slug] $(date +%H:%M:%S) ===================="
  rm -f "$slug.$slug.json"
  "$PY" -m swebench.harness.run_evaluation \
    --dataset_name "$DATASET" \
    --predictions_path "$pred" \
    --run_id "$slug" \
    --namespace swebench \
    --max_workers "$WORKERS" || echo "[$slug] WARN run_evaluation returned nonzero"

  # locate the report json the harness wrote (<model>.<run_id>.json == <slug>.<slug>.json),
  # fall back to any json carrying resolved_instances for this run_id
  rep=""
  [ -f "$slug.$slug.json" ] && rep="$slug.$slug.json"
  if [ -z "$rep" ]; then
    rep=$(grep -lE '"resolved_instances"' ./*"$slug"*.json 2>/dev/null | head -1)
  fi
  if [ -n "$rep" ]; then
    cp "$rep" "results/swebench/$slug.report.json"
    echo "[$slug] report -> results/swebench/$slug.report.json :: $($PY -c "import json;d=json.load(open('$rep'));print('resolved',d['resolved_instances'],'/',d['submitted_instances'],'submitted; empty',d['empty_patch_instances'],'errors',d['error_instances'])")"
  else
    echo "[$slug] ERROR: no report json found after grading"
  fi
  # surface generation telemetry next to the report for the Mac report.md
  [ -f "results/swebench/$slug/telemetry.json" ] && cp "results/swebench/$slug/telemetry.json" "results/swebench/$slug.telemetry.json"
done

echo "SWEB_BATCH_GRADE_DONE $(date +%H:%M:%S)"
echo "--- next: copy results/swebench/*.report.json (+ *.telemetry.json) to your publishing host, then run scripts/chart_swebench_anchor.py ---"
