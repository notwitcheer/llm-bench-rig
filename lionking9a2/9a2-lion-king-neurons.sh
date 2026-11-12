#!/bin/bash
# 9a2-lion-king-neurons.sh: idle-queue bench item (STAGED in ~/voxel-stage; built 2026-11-12, esoteric lane).
# WHAT: arXiv 2611.13885 (causal neuron circuits, repo omtekyav/causal-neuron-circuits pinned d09f034) on Qwen/Qwen3-1.7B-Base, the paper's model (rev ea980cb0, cached 2026-11-12).
#   HEADLINE (counted): the repo's own run_chains.py with its own 1.0 margin gate, no forcing; summarize.py applies PREREG.md to the lk chain.
#   EXTENSION (labelled, never the headline): all 20 chains with the gate forced off (run_chains_probe.py, SMOKE_FORCE_CHAINS), margin split post hoc.
# MOJO: a 12-neuron group from another question flips the lion king's answer from Scar to Simba, the answer text changing live | ahead + funny | "[N] of 4,096 subsets flip it, [D] disjoint pairs flip it too. one chain on qwen3-1.7b-base, the paper's model" (line 1 is built by PREREG.md from the summary, not typed)
# EV: private (projected post angle and reach bracket live in the ops vault status note)
# PRE-REGISTRATION: PREREG.md + FROZEN.sha256 in ~/esoteric/lionking9a2/; PLAN refuses to run on any hash mismatch.
# CARD: the 2026-11-12 forced-lk smoke ran in 17 s (load ~8 s, 1 chain). the 20-chain extension is UNVERIFIED on the 5090; expect minutes, not 45.
# RESUMABLE: the headline leg is short. the extension is the leg that can exit 3 when the hard stop arrives; the next block resumes at the extension.
# WINDOW: NOT_BEFORE_EPOCH=1794528600 (Fri 2026-11-13 01:10 CET; moved from Sat 00:10 on 2026-11-12, the item is ~15 min) waits inside the script; the runner may fire the item earlier and it sleeps until then.
MANIFEST=/home/witcheer/9a2-lionking-status.json
set -uo pipefail
NL_TAG=9a2
[ -n "${MANIFEST_OVERRIDE:-}" ] && MANIFEST=$MANIFEST_OVERRIDE
export MANIFEST
. /home/witcheer/night-lib.sh
ROOT=/home/witcheer/esoteric/lionking9a2
REPO=$ROOT/repo
PY=$ROOT/venv/bin/python
NOT_BEFORE_EPOCH=${NOT_BEFORE_EPOCH:-1794528600}
SMOKE=${LK_SMOKE:-0}
SLUG=9a2-lion-king-neurons
[ "$SMOKE" = 1 ] && SLUG=$SLUG-smoke
RUN=$ROOT/runs/$SLUG
OUT=$RUN/out
HEAD_DIR=$RUN/headline
EXT_DIR=$RUN/ext20
SUMMARY=$OUT/summary.json
CLIP=$OUT/9a2-lion-king-neurons.mp4
MODEL_DIR=/home/witcheer/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B-Base/snapshots/ea980cb0a6c2ae4b936e82123acc929f1cec04c1
FROZEN=$ROOT/FROZEN.sha256
export NL_HARD_STOP_EPOCH=${NL_HARD_STOP_EPOCH:-}

# ---- PLAN: preflight (refuses to run on drift). DRYRUN exits here. ----
nl_say "PLAN phase (DRYRUN=${DRYRUN:-0} smoke=$SMOKE)"
nl_manifest_init plan headline ext summarize render
nl_check_path "$REPO/run_chains.py" "$REPO/run_chains_probe.py" "$REPO/src/intervene.py" "$ROOT/summarize.py" "$ROOT/PREREG.md" "$FROZEN" "$ROOT/render_lk.py" "$REPO/configs/chains-card5090-base.yaml" "$PY" "$MODEL_DIR/model.safetensors" "/home/witcheer/night-lib.sh" "/home/witcheer/voxelbench/ffmpeg"
( cd "$ROOT" && sha256sum -c FROZEN.sha256 >/dev/null 2>&1 ) && nl_ok "PREREG.md, summarize.py, render_lk.py, configs, repo probe match FROZEN.sha256" || nl_fail "a frozen file changed since FROZEN.sha256 was written: refusing to run"
HEADSHA=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo none)
[ "$HEADSHA" = "d09f034" ] && nl_ok "repo HEAD d09f034" || nl_fail "repo HEAD $HEADSHA, want d09f034"
SZ=$(stat -L -c %s "$MODEL_DIR/model.safetensors" 2>/dev/null || echo 0)
[ "$SZ" = 3441185608 ] && nl_ok "Qwen3-1.7B-Base model.safetensors $SZ bytes (HF API size)" || nl_fail "model size $SZ != 3441185608"
nl_check_disk 122
[ -x /home/witcheer/voxelbench/ffmpeg ] && nl_ok "ffmpeg static present" || nl_fail "ffmpeg missing"
nl_gate
nl_say "  PLAN ok"

# ---- wait for the window (sleeps inside the script, never the launch line) ----
now=$(date +%s)
[ "$now" -lt "$NOT_BEFORE_EPOCH" ] && [ "$SMOKE" != 1 ] && { nl_say "not before $(date -d "@$NOT_BEFORE_EPOCH" '+%a %F %H:%M %Z'): exit 3, stays queued (no sleep inside the queue slot)"; nl_mark plan paused "not before window"; exit 3; }

DEADLINE=$(nl_deadline_epoch)
STOPAT=$(( DEADLINE - 420 ))
left_min=$(( (STOPAT - $(date +%s)) / 60 ))
[ "$SMOKE" != 1 ] && [ "$left_min" -lt 12 ] && { nl_say "under 12 min left before the stop: not starting (exit 3, resumable)"; nl_mark plan paused "under 12 min left"; exit 3; }
nl_say "  $left_min min left before the stop"
mkdir -p "$RUN" "$OUT"
t0=$(date +%s)

donald_back=1
donald_restore() {
  [ "$donald_back" = 1 ] && return 0
  donald_back=1
  nl_say "restoring Donald (systemctl start, no pkill)"
  sudo -n systemctl start llama-server.service
  for i in $(seq 1 60); do H=$(curl -s -m 5 localhost:8090/health); [[ "${H,,}" == *ok* ]] && { nl_say "Donald healthy again"; return 0; }; sleep 5; done
  nl_say "WARNING: Donald /health did not answer within 5 min"
}
trap donald_restore EXIT

# ---- the GPU lock is shared with the other staging children: busy means retry later, not fail ----
flock -n /home/witcheer/.gpu-smoke.lock true || { nl_say "gpu-smoke lock busy: exit 3 (retry next block)"; nl_mark plan paused "gpu lock busy"; exit 3; }

# ---- drain Donald and check the card is clear ----
donald_back=0
sudo -n systemctl stop llama-server.service; sleep 5
mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
[ "${mem:-99999}" -lt 1024 ] || { nl_mark headline failed "card not clear: ${mem} MiB"; nl_say "FAIL: card not clear"; exit 1; }

# ---- HEADLINE: the repo's gate, no forcing ----
nl_mark headline running
before=$(ls -d "$REPO"/results/chains_Qwen3-1.7B-Base_* 2>/dev/null | sort | tail -1)
( cd "$REPO" && flock -n /home/witcheer/.gpu-smoke.lock "$PY" run_chains.py configs/chains-card5090-base.yaml > "$RUN/headline.log" 2>&1 ); rc=$?
after=$(ls -d "$REPO"/results/chains_Qwen3-1.7B-Base_* 2>/dev/null | sort | tail -1)
if [ "$rc" = 0 ] && [ -n "$after" ] && [ "$after" != "$before" ] && [ -f "$after/lk.npz" ] && [ -f "$after/results.json" ]; then
  rm -rf "$HEAD_DIR"; mkdir -p "$HEAD_DIR"; cp "$after"/* "$HEAD_DIR"/
  nl_mark headline done "$(basename "$after") rc=0"
else
  nl_mark headline failed "rc=$rc see $RUN/headline.log"; nl_say "FAIL: headline rc=$rc"; exit 1
fi

# ---- EXTENSION: all 20 chains forced (labelled), resumable at the leg boundary ----
if [ ! -f "$EXT_DIR/results.json" ]; then
  left_min=$(( (STOPAT - $(date +%s)) / 60 ))
  if [ "$SMOKE" != 1 ] && [ "$left_min" -lt 8 ]; then
    nl_mark ext paused "headline done; extension not started"; nl_say "extension: under 8 min left, exit 3 (resumes next block)"
    exit 3
  fi
  nl_mark ext running
  ALL=$(python3 -c "import json;print(','.join(sorted({json.loads(l)['chain'] for l in open('$REPO/data/chains.jsonl')})))")
  before=$(ls -d "$REPO"/results/chains_Qwen3-1.7B-Base_* 2>/dev/null | sort | tail -1)
  ( cd "$REPO" && SMOKE_FORCE_CHAINS="$ALL" flock -n /home/witcheer/.gpu-smoke.lock "$PY" run_chains_probe.py configs/chains-card5090-base.yaml > "$RUN/ext.log" 2>&1 ); rc=$?
  after=$(ls -d "$REPO"/results/chains_Qwen3-1.7B-Base_* 2>/dev/null | sort | tail -1)
  if [ "$rc" = 0 ] && [ -n "$after" ] && [ "$after" != "$before" ] && [ -f "$after/results.json" ]; then
    rm -rf "$EXT_DIR"; mkdir -p "$EXT_DIR"; cp "$after"/* "$EXT_DIR"/
    python3 -c "import json;json.dump({'forced':'$ALL'.split(',')},open('$EXT_DIR/forced.json','w'))"
    nl_mark ext done "20 chains forced"
  else
    nl_mark ext failed "rc=$rc see $RUN/ext.log"; nl_say "FAIL: extension rc=$rc"; exit 1
  fi
fi

# ---- Donald back before the CPU stages ----
donald_restore
trap - EXIT

# ---- SUMMARIZE + RENDER (CPU) ----
# summarize.py reads one headline dir and an ext dir of run dirs; ext20 is one run dir, so it sits in a parent with a single child.
EXT_PARENT=$RUN/ext-parent; rm -rf "$EXT_PARENT"; mkdir -p "$EXT_PARENT"; ln -s "$EXT_DIR" "$EXT_PARENT/ext20"
nl_mark summarize running
"$PY" "$ROOT/summarize.py" --headline-dir "$HEAD_DIR" --ext-dir "$EXT_PARENT" --headline lk --out "$SUMMARY" > "$OUT/summarize.log" 2>&1 || { nl_mark summarize failed "see $OUT/summarize.log"; nl_say "FAIL: summarize"; exit 1; }
nl_mark summarize done
nl_say "  $(tail -n 1 "$OUT/summarize.log" | cut -c1-200)"
nl_mark render running
nice "$PY" "$ROOT/render_lk.py" --summary "$SUMMARY" --out "$CLIP" --stills 0.5,9,20,40,53 --stills-dir "$OUT/stills" > "$OUT/render.log" 2>&1 && nl_mark render done "$(stat -c %s "$CLIP") bytes" || { nl_mark render failed "see $OUT/render.log"; nl_say "WARNING: render failed"; }
nl_say "DONE $SLUG in $(( ($(date +%s)-t0)/60 )) min. verdict: $(python3 -c "import json;print(json.load(open('$SUMMARY'))['headline']['verdict'])" 2>/dev/null)"
exit 0
