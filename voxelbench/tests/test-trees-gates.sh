#!/bin/bash
# CPU-only (SwiftShader) test matrix for quick_gate_trees.py and final-gate/gate_trees.py on stub pages (HARDENING L32-L36).
# usage (capsule): bash test-trees-gates.sh <workdir>   expects ~/voxelbench/scripts/quick_gate_trees.py and ~/.vxfinal/gate_trees.py
# never touches the GPU: --swiftshader. prints one line per case: name, quick rc, final rc, key facts.
set -u
W=${1:-$HOME/.vxfinal/test-out}; mkdir -p "$W"
T=$(dirname "$(readlink -f "$0")")
GPY=${GPY:-$HOME/benchmark-rig/.venv/bin/python}
QUICK=${QUICK:-$HOME/voxelbench/scripts/quick_gate_trees.py}; FINAL=${FINAL:-$HOME/.vxfinal/gate_trees.py}
for m in pass nohook nogrow hang slow wronglambda; do sed "s/__MODE__/$m/" "$T/trees-stub-template.html" > "$W/stub-$m.html"; done
for m in pass nohook nogrow hang slow; do
  s=$(date +%s)
  timeout 400 nice -n 19 $GPY $QUICK "$W/stub-$m.html" "$W/q-$m" --swiftshader --video-seconds 0 > "$W/q-$m.txt" 2>&1; qrc=$?
  echo "quick  $m rc=$qrc ($(( $(date +%s)-s )) s): $(tail -c 420 "$W/q-$m.txt" | tr '\n' ' ')"
done
# missing html: path feedback
mkdir -p "$W/runs/slug-a/work" "$W/runs/slug-b/work"; cp "$W/stub-pass.html" "$W/runs/slugb/work/index.html" 2>/dev/null || { mkdir -p "$W/runs/slugb/work"; cp "$W/stub-pass.html" "$W/runs/slugb/work/index.html"; }
timeout 100 $GPY $QUICK "$W/runs/slug-a/work/index.html" "$W/runs/slug-a/qgate" --swiftshader --video-seconds 0 > "$W/q-pathmiss.txt" 2>&1; echo "quick  pathmiss rc=$? : $(tail -c 500 "$W/q-pathmiss.txt" | tr '\n' ' ')"
for m in pass hang slow; do
  s=$(date +%s)
  timeout 900 nice -n 19 $GPY $FINAL "$W/stub-$m.html" "$W/f-$m" --swiftshader --video-seconds 12 --no-extra-seeds > "$W/f-$m.txt" 2>&1; frc=$?
  echo "final  $m rc=$frc ($(( $(date +%s)-s )) s): $(tail -c 600 "$W/f-$m.txt" | tr '\n' ' ')"
done
