#!/bin/bash
# renders a 4-task MOCK teaser from scripted results and checks the mp4 is 1920x1080 25 fps with the expected duration. run on capsule (numpy/PIL/ffmpeg there).
# usage: selftest_render.sh [mock-results-dir mock-models.json]   (defaults: ./mock/results ./mock/mock-models.json, built by make_mock_results.py)
set -euo pipefail
cd "$(dirname "$0")"
RES=${1:-mock/results}; MODS=${2:-mock/mock-models.json}
OUT=$(mktemp -d)
python3 render_orbit.py --res "$RES" --models "$MODS" --mock --out "$OUT/t.mp4" --max-tasks 3 --seconds-per-task 0.6 --finale-seconds 1.5 > "$OUT/log" 2>&1 || { cat "$OUT/log"; exit 1; }
FF=$(python3 -c "import render_orbit as r;print(r.ffmpeg_bin())")
INFO=$("$FF" -nostdin -hide_banner -i "$OUT/t.mp4" </dev/null 2>&1 || true)
case "$INFO" in *"1920x1080"*"25 fps"*) ;; *) echo "bad video: $INFO"; exit 1;; esac
case "$INFO" in *"Duration: 00:00:03"*) ;; *) echo "unexpected duration: $INFO"; exit 1;; esac
echo "RENDER_SELFTEST_OK $(stat -c %s "$OUT/t.mp4") bytes"
rm -rf "$OUT"
