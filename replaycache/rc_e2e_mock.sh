#!/bin/bash
# rc_e2e_mock.sh: CPU end-to-end proof of the replay-cache pipeline with a MOCK source/builder (rc_mockserver.py, hand-written solve() for 3 templates) through the REAL
# 1B extraction, the REAL embedder and the REAL scorer, including near-miss (p1) items. Never counted; nothing here touches the card or :8090.
# usage: rc_e2e_mock.sh <out_dir>
set -uo pipefail
OUT=${1:?out dir}
RC=/home/witcheer/esoteric/replay-cache; CODE=$RC/code
export RC_DATA=$RC/data CUDA_VISIBLE_DEVICES=
mkdir -p "$OUT"; rm -rf "$OUT/run"; mkdir -p "$OUT/run/out"
say() { echo "[e2e] $(date +%H:%M:%S) $*"; }
python3 "$CODE/rc_select.py" "$OUT/sel.json" --tids 365,269,636 --held 6 --near 6 || exit 1
python3 "$CODE/rc_mockserver.py" 8314 > "$OUT/mock.log" 2>&1 & MP=$!
trap 'kill $MP 2>/dev/null' EXIT
for i in $(seq 1 20); do [ "$(curl -s -m 2 http://127.0.0.1:8314/health)" != "" ] && break; sleep 0.5; done
DL=$(( $(date +%s) + 1200 ))
say "build (mock builder, one template needs a rewrite)"
python3 "$CODE/rc_gpu.py" build --base http://127.0.0.1:8314 --out "$OUT/run" --sel "$OUT/sel.json" --deadline $DL --slots 3 2>&1 | tail -8
say "source (mock)"
python3 "$CODE/rc_gpu.py" source --base http://127.0.0.1:8314 --out "$OUT/run" --sel "$OUT/sel.json" --deadline $DL --slots 3 2>&1 | tail -2
kill $MP 2>/dev/null
say "CPU legs: real embedder + real 1B (ALL = embed, B, C, summary)"
RC_NO_RENDER=1 bash "$CODE/rc_chain.sh" --nogpu ALL "$OUT/run" "$OUT/sel.json" "$OUT/run/entries" 2>&1 | tail -14
say "summary:"; head -4 "$OUT/run/out/summary.txt" | cut -c1-300
say "mock render"
"$RC/.venv/bin/python" "$CODE/rc_render.py" "$OUT/run" "$OUT/run/out/mock.mp4" --mock --ffmpeg /home/witcheer/voxelbench/ffmpeg 2>&1 | tail -2
ls -la "$OUT/run/out"
