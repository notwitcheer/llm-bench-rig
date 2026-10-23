#!/bin/bash
# 06-bolt-gpqa32k.sh: idle-queue item, STAGED 2026-10-16 ~21:30 CEST by mercury (operator "following leads, please do for the better").
# follow-up to 05-bolt-74b's gpqa leg: Bolt capped 59/198 at the board's 16k budget (50 of its 56 misses). re-run ONLY those 59
# items at max_tokens 32768 / ctx 40960, same prompts, parser, greedy, server build. answers "how much of the gap to Qwen is the
# budget" before any public GPQA number. sorts after 05 (runs once Bolt's board legs are done), before the duck items.
# artefact: results/nemotron-3-5-bolt-74b-a9b-24gb-iq2-thinkon-capped32k/gpqa_capped32k.json. resumable per item (progress.json).
# not a board row: the 32k board is a separate pass; this is a capped-item diagnostic, reported as such.
set -uo pipefail
NL_TAG=boltg32
MANIFEST=/home/witcheer/boltg32-status.json
RUNLOG=/home/witcheer/boltg32-run.log
export MANIFEST
. /home/witcheer/night-lib.sh
RIG=/home/witcheer/benchmark-rig; PY=$RIG/.venv/bin/python
export BOLT_SRV=/home/witcheer/llama.cpp-bolt/build/bin/llama-server
GGUF=/home/witcheer/models/nemotron-35-bolt/main/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf
SRC=$RIG/results/nemotron-3-5-bolt-74b-a9b-24gb-iq2-thinkon
OUT=$RIG/results/nemotron-3-5-bolt-74b-a9b-24gb-iq2-thinkon-capped32k
ART=$OUT/gpqa_capped32k.json
nl_say "PLAN phase (DRYRUN=${DRYRUN:-0})"
nl_check_path "$PY" "$BOLT_SRV" "$GGUF" "$SRC/gpqa_progress.json" /home/witcheer/bolt-gpqa32k.py
[ "$(stat -c %s "$GGUF")" = 22018326528 ] || { nl_say "  gguf size wrong"; exit 2; }
if [ -f "$ART" ]; then nl_say "artefact present, item complete"; exit 0; fi
# wait for the board item: never take the card while 05-bolt-74b still has legs (lexical order already does this; belt and braces)
[ -f /home/witcheer/idle-queue/05-bolt-74b.sh ] && { nl_say "05-bolt-74b still queued, yielding (exit 3)"; exit 3; }
nl_check_disk 20
DL=$(nl_deadline_epoch); left=$(( DL - $(date +%s) )); nl_say "  hard stop $(date -d "@$DL" '+%a %H:%M') local, $((left/60)) min away"
[ "$left" -gt 900 ] || { nl_say "under 15 min left, resumable"; exit 3; }
if [ "${DRYRUN:-0}" = "1" ] || [ ! -f "$MANIFEST" ]; then nl_manifest_init gpqa_capped32k; fi
nl_gate
restore() { pkill -9 -x llama-server 2>/dev/null; sleep 5; nl_say "EXIT trap: restoring Donald"; systemctl is-active --quiet llama-server.service || sudo -n systemctl start llama-server.service || true; }
trap restore EXIT
nl_say "draining Donald"; sudo -n systemctl stop llama-server.service; sleep 3
for i in $(seq 1 120); do v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${v:-99999}" -lt 4000 ] && { nl_say "drain clear (${v}MiB)"; break; }; sleep 2; [ "$i" -eq 120 ] && { nl_say "FATAL drain timeout"; exit 1; }; done
cd "$RIG"; export PYTHONPATH="$RIG"
t0=$(date +%s); nl_mark gpqa_capped32k running
timeout $(( DL - $(date +%s) - 60 )) $PY -u /home/witcheer/bolt-gpqa32k.py "$GGUF" "$SRC" "$OUT" >> "$RUNLOG" 2>&1; rc=$?
pkill -9 -x llama-server 2>/dev/null
if nl_judge_artifact gpqa_capped32k $rc "$ART" "$t0" "$(( ($(date +%s)-t0)/60 )) min"; then nl_say "  $(grep -a SUMMARY "$RUNLOG" | tail -1 | cut -c1-300)"; exit 0; fi
nl_say "  rc=$rc (124 at the hard stop = resumable), $(python3 -c "import json;print(len(json.load(open('$OUT/progress.json'))))" 2>/dev/null || echo 0)/59 done"; exit 3
