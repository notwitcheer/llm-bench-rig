#!/bin/bash
# render.sh: CPU-only readout step of item 96-arete-eval, run ON capsule AFTER the judge step has copied grades + summary.json into run/arete96/. Not part of the queue item (the judge needs the judge API key).
# usage: bash ~/voxel-stage/arete96/render.sh [RUNDIR]      (default RUNDIR = ~/voxel-stage/arete96/run/arete96)
# writes RUNDIR/out/arete-eval.mp4 (8 s result card, then the two-pane replay of the pre-registered prompt), out/stills/*.png, out/contact-*.png, out/render.log. Refuses BROKEN/INCOMPLETE (exit 4 from the renderer).
set -uo pipefail
S=/home/witcheer/voxel-stage/arete96
RUN=${1:-$S/run/arete96}
RPY=/home/witcheer/scratch/race/venv/bin/python
FF=/home/witcheer/voxelbench/ffmpeg
[ -s "$RUN/summary.json" ] || { echo "render.sh: $RUN/summary.json missing (the judge step has not written it)"; exit 2; }
mkdir -p "$RUN/out"
cp -f "$S/data/items.jsonl" "$RUN/items.jsonl"
"$RPY" "$S/code/render_arete.py" "$RUN" "$RUN/out/arete-eval.mp4" --ffmpeg "$FF" --stills "$RUN/out/stills" > "$RUN/out/render.log" 2>&1; rc=$?
tail -2 "$RUN/out/render.log"
[ "$rc" = 0 ] || { echo "render.sh: renderer rc=$rc"; exit "$rc"; }
"$FF" -nostdin -hide_banner -loglevel error -y -ss 0.5 -i "$RUN/out/arete-eval.mp4" -frames:v 1 "$RUN/out/contact-1.png" </dev/null
"$FF" -nostdin -hide_banner -loglevel error -y -ss 14 -i "$RUN/out/arete-eval.mp4" -frames:v 1 "$RUN/out/contact-2.png" </dev/null
ls -la "$RUN/out/arete-eval.mp4"
