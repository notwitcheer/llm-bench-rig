#!/usr/bin/env bash
# SWE-bench anchor — grade one slug (official harness, cached Docker images). Arg: SLUG
set -uo pipefail
cd ~/benchmark-rig
PY=~/swebench-env/bin/python
DATASET=princeton-nlp/SWE-bench_Verified
WORKERS="${WORKERS:-4}"
mkdir -p results/swebench
slug="$1"
pred="predictions/$slug.jsonl"
[ -f "$pred" ] || { echo "[$slug] no predictions"; exit 2; }
echo "==================== grade [$slug] $(date +%H:%M:%S) ===================="
rm -f "$slug.$slug.json"
"$PY" -m swebench.harness.run_evaluation --dataset_name "$DATASET" --predictions_path "$pred" \
  --run_id "$slug" --namespace swebench --max_workers "$WORKERS" || echo "[$slug] WARN nonzero"
rep=""; [ -f "$slug.$slug.json" ] && rep="$slug.$slug.json"
[ -z "$rep" ] && rep=$(grep -lE '"resolved_instances"' ./*"$slug"*.json 2>/dev/null | head -1)
if [ -n "$rep" ]; then
  cp "$rep" "results/swebench/$slug.report.json"
  echo "[$slug] :: $($PY -c "import json;d=json.load(open('$rep'));print('resolved',d['resolved_instances'],'/',d['submitted_instances'],'submitted; empty',d['empty_patch_instances'],'errors',d['error_instances'])")"
else echo "[$slug] ERROR no report"; fi
echo "SWE_GRADE_ONE_DONE $slug $(date +%H:%M:%S)"
