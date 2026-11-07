#!/bin/bash
# 91-glb-3d-coder.sh: idle-queue item (STAGED in ~/voxel-stage; built 2026-11-05 by a mercury subagent for the GPU thread, mojo-scan pick C, the filler: sorts last).
# WHAT: ccore/Qwen3.5-4B-GLB-3D-Coder (4B, final release 2026-11-05, apache-2.0, writes ONE self-contained Three.js page that shows an asset and has an Export GLB button) against two local models we hold,
#   Qwen3.8-27B UD-Q4_K_M and Ornith-1.5-35B Q4_K_M, on 12 absurd prompts x 2 seeds = 24 cells per model, 72 in all. Same system prompt (the card's, verbatim) and the card's sampler for all three,
#   4B thinking at its template default (on, the card's regime), the 27B/35B with thinking OFF (enable_thinking=false; with it on and a 20k cap they never reached a page in the smoke, PREREG v2/v3), 32,768-token cap. Script checks only, no aesthetic scoring: page loads in headless chromium without a JS error, renders a non-empty scene (pixel variance), Export GLB returns a
#   GLB that parses (magic, JSON chunk, mesh + POSITION, buffer views inside BIN). Design + verdict rule: glb-3d-coder/PREREG.md (frozen with sha256 before any output). The card's DeepSeek comparison is NOT ours and is never repeated.
# MOJO: a 4B model builds a coral-armoured lighthouse keeper (and a pancake pirate ship, a haunted vending machine) in 3D, then the 27B and the 35B get the same prompt: three turntables side by side, each labelled with its size and token count | never seen + funny (hook 3 and 1) | [line 1 from the data: "on 12 absurd prompts a 4B model's Three.js page loaded, drew a scene and exported a valid GLB in N/24 tries; Qwen3.8-27B M/24, Ornith-1.5-35B K/24"]
# EV: never seen + funny (a tiny specialist writes 3D assets from absurd prompts next to our big local models; the turntables carry it). comparables: Muse Glimmer 30B 3,901 views as the nearest brand-pull-free niche model, Qwen day-0 first numbers 8,995, which-file post 11,438 (gpu-scout-2026-11-03.md); the r/LocalLLaMA thread is at 677. projected 2-5k (UNVERIFIED).
#   rules: gpu-ev-gate-2026-10-25 section 2 rule 3 (a finding first, never "I ran"/"my") + the mojo rule. call: X only if the turntables read at phone size and the run is not BROKEN/INVALID; a LEVEL or BEHIND result is still a post ("a 4B that fits in 5 GB vs a 27B"), said plainly; else TG + repo.
# serves: content (never-seen picture) + experiment (does a task-specific 4B hold against 7-9x larger general models on its own task, and where does each fail).
#
# PRE-REGISTERED RULE (PREREG.md; score.py applies it, selftest.sh proves PROVISIONAL / BROKEN / AHEAD branches):
#   BROKEN = the 4B server does not come up or the 4B has 0 page_valid cells over its 24 | INVALID = checker fault rate > 20% for a model | PROVISIONAL while any model has < 24 scored cells (exit 3, resumable)
#   else d = full_pass(4B) - full_pass(X) per comparator: d >= 3 AHEAD, d <= -3 BEHIND, else LEVEL; overall AHEAD OF BOTH / BEHIND BOTH / LEVEL WITH BOTH / MIXED. full_pass = page_valid AND valid GLB export.
# POSTABLE ARTEFACT (CPU only on capsule after the card is released; render_glb.py): per prompt out/render/turntable-<prompt>.mp4 (16:9 1920x1080, 4 s, three models side by side, each labelled size + completion tokens,
#   subject turning from frame 0, the model's OWN exported GLB in one fixed shared viewer) + contact-<prompt>.png + sheet-all.png + tally.png/tally.mp4 + render-summary.json; out/aggregate.json + out/readout.md.
# HONEST LIMITS: 24 cells per model on one author's absurd prompts; "ran and exported" is not "good"; the card sampler + system prompt are tuned for the 4B only; a 32,768-token cap cuts a long thinker off (counted as a fail and reported);
#   27B/35B are Q4_K_M, the 4B is Q8_0 (the card's evaluation file); software-GL page check (animation speed not judged); a seed is a label under --parallel 4; no Hermes involved; the card says its dataset release is still pending.
# regime (label every number): llama.cpp 839a572a4 (build 11777), --jinja (each model's own embedded template; comparators enable_thinking=false), --parallel 4 --ctx-size 139264 (34,816/slot), flash-attn on, f16 KV, temp 0.8 top_k 40 top_p 0.95 min_p 0 repeat 1.0, seeds 20261105/20261106, max_tokens 20480, port 8196.
# resumable PER cell (results/<model>.jsonl and checks.jsonl rows are skipped). needs >= 40 min left in the window to start (exit 3, stays queued, otherwise); a leg is only started with its own minimum left; exit 3 while legs remain.
# card time: ESTIMATE ~60-90 min typical (worst case ~2 h at the 32,768 cap on every cell; resumable) incl. 3 loads (smoke 2026-11-05, par 2-4: 4B Q8 ~15k tokens/page at ~150-210 tok/s per slot, 27B thinking-off ~9-10k tokens at ~50-65 tok/s per slot, 35B ~150-220 tok/s per slot; full-run totals are not measured). CPU checks run in the background while the next model loads; render ~2-4 min CPU after the card is released.
#   block: any heures-creuses block, it sorts last and is the filler; needs >= 40 min left to start, a 27B/35B leg needs >= 20 min.
# test knobs (empty in production): GLB_SMOKE=1 (slug glb91-smoke, no minimum rule, no summary/render), GLB_MODELS=key,key, GLB_PROMPTS=id,id, GLB_SEEDS=n,n, GLB_MAXTOK=n, GLB_PARALLEL=n, GLB_NO_DRAIN=1 (smoke only: leave Donald alone, needs a rule-3 pass and free VRAM),
#   GLB_STUB_UPSTREAM=http://127.0.0.1:PORT (CPU stub: no drain, no server start; DRYRUN-safe), GLB_NO_SELFTEST=1, GLB_SLUG_SUFFIX.
MANIFEST=/home/witcheer/glb-3d-coder-status.json
set -uo pipefail
NL_TAG=glb91
[ -n "${MANIFEST_OVERRIDE:-}" ] && MANIFEST=$MANIFEST_OVERRIDE
export MANIFEST
. /home/witcheer/night-lib.sh
CODE=/home/witcheer/glb-3d-coder
RUNS=/home/witcheer/glb-3d-coder/runs
BIN=/home/witcheer/llama.cpp-donald/build/bin/llama-server          # read-only use: never modified
PORT=8196                                                           # own port: never :8090 (llama-swap / Donald)
SMOKE=${GLB_SMOKE:-0}; STUB=${GLB_STUB_UPSTREAM:-}; NODRAIN=${GLB_NO_DRAIN:-0}
SLUG=glb91; [ "$SMOKE" = 1 ] && SLUG=$SLUG-smoke; SLUG=$SLUG${GLB_SLUG_SUFFIX:-}
RUN=$RUNS/$SLUG; RES=$RUN/res; OUT=$RUN/out; ART=$RUN/summary.json
PYX=/home/witcheer/benchmark-rig/.venv/bin/python                   # playwright + PIL + numpy
WANT=${GLB_MODELS:-all}; PROMPTS=${GLB_PROMPTS:-all}; SEEDS=${GLB_SEEDS:-all}; PAR=${GLB_PARALLEL:-4}; MAXTOK=${GLB_MAXTOK:-0}
CTX_SLOT=34816; CTX=$(( CTX_SLOT * PAR ))
mapfile -t LEGS < <(python3 - "$CODE/config.json" "$WANT" <<'PYEOF'
import json, sys
c = json.load(open(sys.argv[1])); want = None if sys.argv[2] == "all" else set(sys.argv[2].split(","))
for m in c["models"]:
    if want is None or m["key"] in want: print("|".join([m["key"], m["file"], str(m["bytes"]), m["sha256"], m["label"]]))
PYEOF
)
NP=$(python3 -c "import json,sys;p=json.load(open('$CODE/prompts.json'))['prompts'];w='$PROMPTS';print(len(p) if w=='all' else len(w.split(',')))")
NS=$(python3 -c "import json;c=json.load(open('$CODE/config.json'));w='$SEEDS';print(len(c['seeds']) if w=='all' else len(w.split(',')))")
CELLS_PER_LEG=$(( NP * NS )); EXPECT_N=$(( ${#LEGS[@]} * CELLS_PER_LEG ))

[ "$SMOKE" = 1 ] || [ "${#LEGS[@]}" = 3 ] || nl_fail "production run needs all 3 legs, got ${#LEGS[@]} (GLB_MODELS is a smoke knob)"
[ "$SMOKE" = 1 ] || [ "$NODRAIN" = 0 ] || nl_fail "GLB_NO_DRAIN is a smoke knob"
nl_say "PLAN phase (DRYRUN=${DRYRUN:-0} smoke=$SMOKE stub=${STUB:-none} legs=${#LEGS[@]} prompts=$NP seeds=$NS cells_per_leg=$CELLS_PER_LEG expect=$EXPECT_N parallel=$PAR)"
nl_check_path "$BIN" "$CODE/glbcommon.py" "$CODE/run_glb.py" "$CODE/check_cells.py" "$CODE/score.py" "$CODE/render_glb.py" "$CODE/mock_glb.py" "$CODE/selftest.sh" "$CODE/prompts.json" "$CODE/system.txt" "$CODE/config.json" "$CODE/PREREG.md" "$CODE/FROZEN.sha256" \
  "$CODE/vendor/package/build/three.module.js" "$CODE/vendor/package/examples/jsm/loaders/GLTFLoader.js" "$CODE/vendor/package/examples/jsm/exporters/GLTFExporter.js" "$PYX" /home/witcheer/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome /home/witcheer/voxelbench/lib
if [ -z "$STUB" ]; then
  for L in "${LEGS[@]}"; do
    IFS='|' read -r K F B S LB <<< "$L"
    SZ=$(stat -c %s "$F" 2>/dev/null || echo 0)
    [ "$SZ" = "$B" ] && nl_say "  ok: $K file $SZ bytes" || nl_fail "$K model file size $SZ != $B ($F)"
    REL=${F#/home/witcheer/models/}; REC=$(grep -F " $REL" "$CODE/model-sha256.txt" | cut -d' ' -f1)
    [ "$REC" = "$S" ] && nl_say "  ok: $K sha256 on record equals config.json (${S:0:12})" || nl_fail "$K sha256 on record '${REC:0:12}' != config.json '${S:0:12}'"
  done
fi
HELP=$($BIN --help 2>&1 || true)
for f in --jinja --parallel --flash-attn --ctx-size --alias --n-gpu-layers; do case "$HELP" in *"$f"*) ;; *) nl_fail "flag $f not in this build";; esac; done
nl_say "  ok: server flags exist in the build"
VER=$($BIN --version 2>&1 || true)
case "$VER" in *839a572*) nl_say "  ok: llama.cpp 839a572a4 (build 11777)";; *) nl_fail "llama.cpp build changed or missing (expected 839a572)";; esac
nl_check_disk 120
ss -ltnH "sport = :$PORT" 2>/dev/null | grep -q . && nl_fail "port $PORT is already in use"
cd "$CODE" || nl_fail "cannot cd $CODE"
sha256sum -c FROZEN.sha256 > /dev/null 2>&1 && nl_say "  ok: PREREG.md, prompts.json, system.txt, config.json match FROZEN.sha256" || nl_fail "frozen files changed (FROZEN.sha256)"
python3 - <<'PYEOF' || nl_fail "config check: counts, system prompt or sampler differ from the card values frozen in PREREG"
import json, re
c = json.load(open("config.json")); p = json.load(open("prompts.json"))["prompts"]
assert len(p) == 12 and len({x["id"] for x in p}) == 12 and len(c["seeds"]) == 2 and len(c["models"]) == 3
s = c["sampler"]; assert (s["temperature"], s["top_k"], s["top_p"], s["min_p"], s["repeat_penalty"]) == (0.8, 40, 0.95, 0.0, 1.0)
sy = open("system.txt").read(); assert sy.startswith("Create game assets for the Godot editor") and "Export GLB button" in sy
for x in p: assert not re.search(r"/home/|/Users/|witcheer|benchmark", x["text"], re.I), x["id"]
PYEOF
nl_say "  ok: 12 prompts, 2 seeds, 3 models, card sampler + card system prompt"
if [ "${GLB_NO_SELFTEST:-0}" != 1 ]; then
  ss -ltnH "sport = :8655" 2>/dev/null | grep -q . && nl_fail "port 8655 (selftest mock) in use"
  GLB_SELFTEST_RENDER=0 bash "$CODE/selftest.sh" /home/witcheer/scratch/glb91-selftest-dry 8655 > /home/witcheer/scratch/glb91-selftest-dry.log 2>&1 \
    && nl_say "  ok: CPU proof on a mock endpoint (driver + resume, checker branches, scorer branches): $(tail -n 1 /home/witcheer/scratch/glb91-selftest-dry.log)" || nl_fail "selftest red (see ~/scratch/glb91-selftest-dry.log)"
fi
nl_say "  params: legs=$(for L in "${LEGS[@]}"; do printf '%s ' "${L%%|*}"; done) sampler card, seeds config, ctx_slot=$CTX_SLOT parallel=$PAR port=$PORT"
if [ -f "$ART" ] && nl_skip_exact "$ART"; then nl_say "nothing to run, item complete ($ART exists)"; exit 0; fi
DL=$(nl_deadline_epoch); LEFT=$(( (DL-$(date +%s))/60 ))
nl_say "  planned RUN: $SLUG ($EXPECT_N cells over ${#LEGS[@]} legs; hard stop $(date -d "@$DL" '+%a %H:%M') local, $LEFT min away)"
if [ "$SMOKE" != 1 ] && [ -z "$STUB" ]; then
  [ "$LEFT" -ge 40 ] || { nl_say "  under 40 min left in the window: not starting (three loads + 72 long generations would be cut short; exit 3, resumable)"; exit 3; }
fi
if [ "${DRYRUN:-0}" = "1" ] || [ ! -f "$MANIFEST" ]; then
  PH=(); for L in "${LEGS[@]}"; do PH+=("leg-${L%%|*}"); done
  nl_manifest_init serve "${PH[@]}" check aggregate render
fi
nl_gate

SRV=""; CHKS=(); DONALD_STOPPED=0; DONALD_BACK=0
killsrv() { { [ -n "$SRV" ] && kill -9 "$SRV" && wait "$SRV"; } 2>/dev/null; SRV=""; }              # only the server THIS item started
donald_stop() { nl_say "draining Donald (nl_donald_stop)"; nl_donald_stop; DONALD_STOPPED=1; }
donald_restore() {
  [ "$DONALD_STOPPED" = 1 ] && [ "$DONALD_BACK" = 0 ] || return 0
  DONALD_BACK=1; nl_say "restoring Donald (nl_donald_restore)"
  nl_donald_restore
  for i in $(seq 1 60); do H=$(curl -s -m 5 localhost:8090/health); [[ "${H,,}" == *ok* ]] && { nl_say "Donald healthy again after ~$((i*5)) s"; return 0; }; sleep 5; done
  nl_say "WARNING: Donald /health did not answer within 5 min after the restore"
}
cleanup() { killsrv; for c in "${CHKS[@]:-}"; do [ -n "$c" ] && kill "$c" 2>/dev/null; done; sleep 2; donald_restore; }
trap cleanup EXIT
mem_clear() { local v=99999 i
  for i in $(seq 1 60); do v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${v:-99999}" -lt 1024 ] && { nl_say "card clear (${v} MiB)"; return 0; }; sleep 2; done
  nl_say "card memory still ${v:-?} MiB"; return 1; }
mem_base() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
mem_settle() { # smoke without a drain: wait until the card is back at the level it had before we loaded (our own server's memory released)
  local i v; for i in $(seq 1 60); do v=$(mem_base); [ "${v:-99999}" -le $(( ${MEM0:-0} + 300 )) ] && return 0; sleep 2; done; nl_say "card memory ${v:-?} MiB still above the pre-load ${MEM0:-?}"; return 1; }
mkdir -p "$RES" "$OUT" "$RUN/logs"
MEM0=$(mem_base)
t0=$(date +%s)
nl_mark serve running
if [ -n "$STUB" ]; then UP_PORT=${STUB##*:}; nl_mark serve done "stub upstream $STUB"
elif [ "$NODRAIN" = 1 ] && [ "$SMOKE" = 1 ]; then UP_PORT=$PORT; nl_mark serve done "smoke without drain (rule-3 pass, free VRAM)"
else
  donald_stop
  mem_clear || { nl_mark serve failed "card memory not clear after the drain: another process owns the card"; nl_say "FAIL: card memory not clear after the drain, not loading"; exit 1; }
  UP_PORT=$PORT; nl_mark serve done "Donald drained; one fresh server per leg on $PORT"
fi
DEADLINE=$(nl_deadline_epoch)
STOPAT=$(( DEADLINE - 420 ))                            # 7 min for the kill, the Donald restore and the aggregation
[ "$SMOKE" = 1 ] && STOPAT=$(( DEADLINE - ${GLB_SMOKE_MARGIN:-60} ))

serve_leg() { # key file -> 0 when /health answers; writes fit.jsonl + props
  local key=$1 file=$2 log=$RUN/logs/server-$1-$(date +%H%M).log up=0 used tot i
  $BIN --model "$file" --host 127.0.0.1 --port $PORT --alias "$key" --jinja --parallel $PAR --n-gpu-layers 99 --ctx-size $CTX --flash-attn on > "$log" 2>&1 &
  SRV=$!
  for i in $(seq 1 300); do H=$(curl -s -m 3 127.0.0.1:$PORT/health); [[ "${H,,}" == *ok* ]] && { up=1; break; }; kill -0 $SRV 2>/dev/null || break; sleep 2; done
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); tot=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
  echo "{\"model\":\"$key\",\"up\":$([ "$up" = 1 ] && echo true || echo false),\"used_mib\":${used:-0},\"free_mib\":$(( ${tot:-0} - ${used:-0} )),\"ctx\":$CTX,\"parallel\":$PAR}" >> "$RUN/fit.jsonl"
  if [ "$up" != 1 ] || [ $(( tot - used )) -lt 1000 ]; then nl_say "  FAIL: $key server up=$up used=$used free=$(( tot - used )): $(grep -m1 -iE 'out of memory|failed|error' "$log" | head -c 200)"; return 1; fi
  curl -s 127.0.0.1:$PORT/props > "$RUN/props-$key.json"
  nl_say "  $key up: ${used} MiB used, $(( tot - used )) free, template in props: $(python3 -c "import json,hashlib;j=json.load(open('$RUN/props-$key.json'));t=j.get('chat_template') or '';print(len(t),'chars sha',hashlib.sha256(t.encode()).hexdigest()[:10])" 2>/dev/null || echo '?')"
  return 0
}
count_rows() { [ -f "$RES/$1.jsonl" ] && grep -c . "$RES/$1.jsonl" || echo 0; }
rc_all=0; legs_left=0
GENARGS=(--prompts "$PROMPTS" --seeds "$SEEDS" --inflight "$PAR"); [ "$MAXTOK" != 0 ] && GENARGS+=(--max-tokens "$MAXTOK")
for L in "${LEGS[@]}"; do
  IFS='|' read -r K F B S LB <<< "$L"
  have=$(count_rows "$K")
  if [ "$have" -ge "$CELLS_PER_LEG" ]; then nl_say "leg $K: $have/$CELLS_PER_LEG cells, done"; nl_mark "leg-$K" done "$have cells (earlier block)"; continue; fi
  now=$(date +%s); left_min=$(( (STOPAT - now) / 60 )); need=10; case "$K" in qwen38*|ornith*) need=20;; esac
  if [ "$SMOKE" != 1 ] && [ -z "$STUB" ] && [ "$left_min" -lt "$need" ]; then nl_say "leg $K: only $left_min min left before the stop (needs $need): not started"; nl_mark "leg-$K" paused "$have cells, not started"; legs_left=1; continue; fi
  [ "$now" -ge "$STOPAT" ] && { nl_say "leg $K: past the stop time"; nl_mark "leg-$K" paused "$have cells"; legs_left=1; continue; }
  nl_mark "leg-$K" running "$have of $CELLS_PER_LEG cells"
  l0=$(date +%s)
  if [ -z "$STUB" ]; then
    if ! serve_leg "$K" "$F"; then
      killsrv; nl_mark "leg-$K" failed "server did not come up"; if [ "$NODRAIN" = 1 ]; then mem_settle; else mem_clear; fi; rc_all=1
      if [ "$K" = glb4b-q8 ]; then echo "$K server did not come up ($(tail -n 1 "$RUN/fit.jsonl" | cut -c1-100))" > "$RUN/BROKEN.note"; nl_say "FAIL: the 4B did not load: the item is BROKEN by rule, skipping the other legs"; break; fi
      continue
    fi
  fi
  python3 "$CODE/run_glb.py" --model "$K" --port "$UP_PORT" --out "$RES" "${GENARGS[@]}" --deadline-epoch "$STOPAT" >> "$RUN/logs/run.log" 2>&1
  rrc=$?
  have=$(count_rows "$K")
  nl_say "  leg $K generation rc=$rrc: $have/$CELLS_PER_LEG cells in $(( $(date +%s) - l0 )) s: $(tail -n 1 "$RUN/logs/run.log" | cut -c1-140)"
  if [ -z "$STUB" ]; then killsrv; sleep 2; if [ "$NODRAIN" = 1 ]; then mem_settle || { rc_all=1; break; }; else mem_clear || { nl_say "FAIL: card memory did not clear after $K, stopping the item"; rc_all=1; break; }; fi; fi
  # CPU checker for this leg in the background while the next model loads (nice, 3 workers; the card is not used)
  ( nice -n 10 $PYX "$CODE/check_cells.py" --res "$RES" --models "$K" --workers 3 >> "$RUN/logs/check.log" 2>&1 ) &
  CHKS+=($!)
  if [ "$have" -ge "$CELLS_PER_LEG" ]; then nl_mark "leg-$K" done "$have cells in $(( $(date +%s) - l0 )) s"; else nl_mark "leg-$K" paused "$have of $CELLS_PER_LEG cells (rc $rrc)"; legs_left=1; fi
done
killsrv; sleep 2
[ -z "$STUB" ] && donald_restore
nl_say "card released after $(( ($(date +%s)-t0)/60 )) min"

# ---- finish the checks (CPU) for every generated cell, aggregate, render ----
nl_mark check running
for c in "${CHKS[@]:-}"; do [ -n "$c" ] && wait "$c" 2>/dev/null; done; CHKS=()
nice -n 10 $PYX "$CODE/check_cells.py" --res "$RES" --workers 3 >> "$RUN/logs/check.log" 2>&1
nl_mark check done "$(grep -c . "$RES/checks.jsonl" 2>/dev/null || echo 0) rows"
nl_mark aggregate running
BN=""; [ -f "$RUN/BROKEN.note" ] && BN="--broken-note $RUN/BROKEN.note"
python3 "$CODE/score.py" --res "$RES" --out "$OUT" --cells "$CELLS_PER_LEG" $BN >> "$RUN/logs/aggregate.log" 2>&1
AGG=$(python3 -c "import json;d=json.load(open('$OUT/aggregate.json'));print(d['overall'],'|',d['why'],'|',{k:(v['full_pass'],v['page_valid'],v['n_scored']) for k,v in d['models'].items()})" 2>/dev/null || echo 'none yet')
nl_say "  aggregate: $AGG"
NRES=0; for L in "${LEGS[@]}"; do NRES=$(( NRES + $(count_rows "${L%%|*}") )); done
nl_say "cells generated: $NRES of $EXPECT_N"
if [ -f "$RUN/BROKEN.note" ]; then
  nl_mark aggregate done "BROKEN: $(cat "$RUN/BROKEN.note")"; cp "$OUT/aggregate.json" "$ART"; nl_say "  RULE: BROKEN, no post, vault note only"; nl_say "DONE $SLUG (BROKEN)"; exit 0
fi
if [ "$NRES" -ge "$EXPECT_N" ] && [ "$legs_left" = 0 ] && [ "$rc_all" = 0 ]; then
  nl_mark aggregate done "$NRES cells"
  if [ "$SMOKE" != 1 ]; then
    cp "$OUT/aggregate.json" "$ART"
    nl_judge_artifact aggregate 0 "$ART" "$t0" "$NRES cells, $(( ($(date +%s)-t0)/60 )) min this block" || { nl_say "FAIL: no fresh summary.json"; exit 1; }
    nl_say "  RULE: $(python3 -c "import json;d=json.load(open('$ART'));print(d['overall'], '|', d['why'], '|', d['line1'])")"
    nl_mark render running
    MOCKFLAG=""; [ -n "$STUB" ] && MOCKFLAG="--mock"
    ( nice -n 10 $PYX "$CODE/render_glb.py" --res "$RES" --out "$OUT/render" --agg "$OUT/aggregate.json" $MOCKFLAG >> "$RUN/logs/render.log" 2>&1 ) \
      && nl_mark render done "$(ls "$OUT/render"/turntable-*.mp4 2>/dev/null | wc -l) turntables" && nl_say "  render: $OUT/render ($(ls "$OUT/render"/turntable-*.mp4 2>/dev/null | wc -l) turntables + contacts + sheet + tally)" \
      || { nl_mark render failed "see $RUN/logs/render.log"; nl_say "WARNING: render failed (see $RUN/logs/render.log); the numbers are in $ART"; }
    nl_say "DONE $SLUG"
  else
    nl_say "smoke complete (cells are never data): $OUT/readout.md"
  fi
  exit 0
fi
nl_mark aggregate paused "$NRES of $EXPECT_N cells; resumable"
nl_say "legs remain ($NRES of $EXPECT_N): exit 3, the item stays queued"
[ "$rc_all" = 0 ] || { nl_say "FAIL: a leg failed (see $RUN/logs)"; exit 1; }
exit 3
