#!/bin/bash
# tests/test-flock-gates.sh: CPU-only proof of the episode-2 gates (quick gate = agent-visible, final gate = ~/.vxfinal). Run on capsule (playwright chromium, --disable-gpu).
# usage: bash test-flock-gates.sh <ref.html template> [workdir]   (template = reference/flock/hovm_ref.html with MODE = 'ok')
# PASS path: MODE ok must pass both gates. RED path: each wrong mode must FAIL the intended check(s) and exit 1:
#   hovm_is_svm    -> triadic panel behaves like the pairwise one (hovm_collapsed, gap)
#   self_included  -> the paper's own text rule (i inside its neighbour pairs): collapse comes ~0.4 of eta too late (hovm_collapsed)
#   noise_double   -> noise 2*eta (svm_ordered_low)
#   one_panel      -> right panel empty (two_panels)
#   nohook         -> no window.flockRun / flockStats (hooks)
#   throws         -> flockRun throws (hooks)
#   freeze_hot     -> flockRun(eta >= 3) never returns (main_thread_ok / alive_at_other_noise)
set -u
T=${1:?template}; W=${2:-$HOME/voxel-stage/ep2/stubs}; mkdir -p "$W"
PY=${PY:-$HOME/benchmark-rig/.venv/bin/python}
QG=${QG:-$HOME/voxelbench/scripts/quick_gate_flock.py}; FG=${FG:-$HOME/.vxfinal/gate_flock.py}
printf '%-14s %-8s %-8s %s\n' mode quick final failed-checks
for M in ok hovm_is_svm self_included noise_double one_panel nohook throws freeze_hot; do
  sed "s/^const MODE = 'ok';/const MODE = '$M';/" "$T" > "$W/stub-$M.html"
  q=$(timeout 600 $PY $QG "$W/stub-$M.html" "$W/q-$M" 2>&1 | tail -1); qrc=$?
  f=$(timeout 900 $PY $FG "$W/stub-$M.html" "$W/f-$M" --no-extra 2>&1 | tail -1)
  qp=$($PY -c "import json,sys;d=json.loads(sys.argv[1]);print('pass' if d.get('pass') else 'FAIL')" "$q" 2>/dev/null || echo ERR)
  fp=$($PY -c "import json,sys;d=json.loads(sys.argv[1]);print('pass' if d.get('pass') else 'FAIL')" "$f" 2>/dev/null || echo ERR)
  bad=$($PY -c "import json,sys;d=json.loads(sys.argv[1]);print(','.join(k for k,v in (d.get('checks') or {}).items() if not v))" "$f" 2>/dev/null)
  qbad=$($PY -c "import json,sys;d=json.loads(sys.argv[1]);print(','.join(k for k,v in (d.get('checks') or {}).items() if not v))" "$q" 2>/dev/null)
  printf '%-14s %-8s %-8s final:[%s] quick:[%s]\n' "$M" "$qp" "$fp" "$bad" "$qbad"
done
echo "chromium left over: $(pgrep -c -x chrome || true)"
