#!/bin/bash
# 88-units-orbit.sh: idle-queue item (STAGED in ~/voxel-stage; built 2026-11-05, esoteric lane).
# WHAT: arXiv 2609.25009 (numeric representation invariance, one author, 2026-07-27) tested units in ANSWERS. This tests units inside TOOL ARGUMENTS, which the paper never measured: the user writes a quantity
#   in one unit, the tool argument is defined in another (the schema names it: terminal `timeout` "Max seconds ...", process wait, browser CDP "Timeout in seconds", browser_exec timeout_s, computer_use wait seconds),
#   the model must convert. A power-of-ten slip there is a wrong action (asked for a 90-second timeout, it sets 25 hours). 40 tasks x 4 forms (canonical unit, sub-unit, larger unit, words/fraction) = 160 single-turn
#   tool-call requests per model, 24 tasks on REAL Hermes schemas (9bda188, extracted verbatim) + 16 on SYNTHETIC tools (cents, metres, milliseconds, grams; labelled synthetic everywhere). 5 models (800 calls):
#   Qwen3.8-27B Q6_K, Ornith-1.5-35B Q4_K_M, gpt-oss-20b MXFP4 (low effort, not off), Qwen3.5-9B Q4_K_M and Q8_0 (the paper's exact files). Exact-rational scorer, greedy, thinking off. Design: units/PREREG.md.
# MOJO: an orbit ring of 40 tool calls where each model's marker flies to the value it set and a slip is flung off by x10 / x100 / x1000, the twist as one big red number | esoteric (a 2-week-old paper's untested axis, on our card) + absurd if it slips | [line 1 from result: "asked for <a 90-second timeout>, <model> set <25 hours>" if SLIPS FOUND, else "local models convert units into tool arguments cleanly when the schema names the unit"]
# EV: private (projected post angle and reach bracket live in the ops vault status note)
#   (esoteric: a weeks-old paper's untested axis, checked not claimed as first: NUMCoT and Bui et al. 2025 are in the paper's related work). projected bracket 1-3k if SLIPS FOUND (the absurd line + the ring), 0.8-1.5k if CLEAN (a null with a good picture), UNVERIFIED.
#   the subject is how models convert units into a tool argument; the tool schemas state the unit.
# serves: experiment (a method question: does a local model slip by a power of ten when a tool argument's unit differs from the user's, and is the Q4/Q8 pair different there) + content (esoteric lane).
#
# PRE-REGISTERED RULE (units/PREREG.md, frozen before any model output; score.py applies it, aggregate_selftest.py proves the branches):
#   per model over its 160 calls: BROKEN (> 20% no_call or malformed: harness problem, declared invalid) | SLIPS FOUND (>= 3 magnitude slips: value x 10^k == expected, 0<|k|<=6) | CLEAN (0-2).
#   overall: INVALID (every model BROKEN) | SLIPS FOUND (any valid model) | CLEAN. PROVISIONAL while any leg is missing.
# LINE-1 VARIANTS (filled from the data by score.py, nothing worded by hand): SLIPS FOUND "asked for <ask of the worst slip>, <model> set <value in the argument's own unit ladder>." (+ the prompt wording, slip count)
#   CLEAN "local models convert units into tool arguments cleanly when the schema names the unit." (+ counts)   INVALID: no post, a harness note only.
# POSTABLE ARTEFACT (planned here, CPU only on capsule after the run): out/units-orbit.mp4 (16:9 1920x1080, ~66 s, render_orbit.py: motion from frame 1, the ring fills >= 30% of the frame, worst slip first, the twist as one big number, MOCK banner only on mock data)
#   + out/units-orbit-contact.png + out/readout.md + out/aggregate.json (+ out/bridge-*-compare.json when the bridge leg ran).
# HONEST LIMITS: one greedy pass per cell; single-turn, ONE tool offered; thinking off (gpt-oss low effort, a label not an equal); 40 small tasks and phrasings of one author; 16 of 40 tools are synthetic; the Qwen3.5 pair runs on llama.cpp 839a572a4 / RTX 5090,
#   the paper's on b10107 / H100 (per-item parity is checked by the bridge leg, not assumed); Q4 vs Q8 is one checkpoint pair, descriptive; 3 slips is an action threshold, not significance; nothing here says how a Hermes SESSION behaves.
# regime (label every number): llama.cpp 839a572a4 (build 11777), --jinja (each model's own embedded template), --parallel 1, ctx 8192, flash-attn on, f16 KV, temp 0 top_k 1 top_p 1 min_p 0 seed 20260725, max_tokens 512 (gpt-oss 1024), port 8198, Hermes schemas @ 9bda188.
# resumable PER (model, task, form): results/<model>.jsonl rows are skipped. needs >= 45 min left in the window to start (exit 3, stays queued, otherwise); a leg is only started with its own minimum left; exit 3 while legs remain.
# card time: smoke 2026-11-05 on Qwen3.5-9B Q4 (12 tool calls 9 s = 0.75 s/call, load ~4 s from page cache, 10 bridge calls 12 s). expected ~25-40 min for the 5 legs incl. 5 loads (gpt-oss at low effort and the two larger files are the slow ones; cold loads of 22 + 21 GB add ~1-2 min each), +~9-12 min for the bridge leg (default ON, UNITS_BRIDGE=0 turns it off; the bridge runs only with >= 25 min left, after all five legs). Likeliest block: Fri 2026-11-06 01:10 CET (sorts after the duck items 09z22/09z23, which own Thu 14:10).
# test knobs (empty in production): UNITS_SMOKE=1 (slug -smoke, no minimum rule, no summary/render, tasks UNITS_TASKS, models UNITS_MODELS=key,key, bridge n UNITS_BRIDGE_N), UNITS_STUB_UPSTREAM=http://127.0.0.1:PORT (CPU stub: no Donald drain, no server start;
#   DRYRUN-safe), UNITS_MODELS_JSON=<path> (alternative models.json, stub only), UNITS_NO_SELFTEST=1 (skip the mock proofs inside the DRYRUN), UNITS_BRIDGE=0 (skip the bridge leg), UNITS_SLUG_SUFFIX.
MANIFEST=/home/witcheer/units-orbit-status.json
set -uo pipefail
NL_TAG=units88
[ -n "${MANIFEST_OVERRIDE:-}" ] && MANIFEST=$MANIFEST_OVERRIDE
export MANIFEST
. /home/witcheer/night-lib.sh
CODE=/home/witcheer/esoteric/units
RUNS=/home/witcheer/esoteric/units-runs
BIN=/home/witcheer/llama.cpp-donald/build/bin/llama-server
PORT=8198                                              # own port: never :8090 (llama-swap / Donald)
CTX=8192
ANC=/home/witcheer/esoteric/units/anc                  # copy of the arXiv 2609.25009v1 ancillary archive (views + eval.jsonl + parser), used by the bridge leg only
SMOKE=${UNITS_SMOKE:-0}
STUB=${UNITS_STUB_UPSTREAM:-}
SLUG=units88
[ "$SMOKE" = 1 ] && SLUG=$SLUG-smoke
SLUG=$SLUG${UNITS_SLUG_SUFFIX:-}
RUN=$RUNS/$SLUG
RES=$RUN/results
OUT=$RUN/out
ART=$RUN/summary.json                                  # exists only when every planned row of every leg exists
MODELS_JSON=$CODE/models.json
[ -n "$STUB" ] && [ -n "${UNITS_MODELS_JSON:-}" ] && MODELS_JSON=$UNITS_MODELS_JSON
WANT_MODELS=${UNITS_MODELS:-all}
TASKS=${UNITS_TASKS:-all}
BRIDGE=${UNITS_BRIDGE:-1}
BRIDGE_N=${UNITS_BRIDGE_N:-100}
# legs: key|file|bytes|sha256|label
mapfile -t LEGS < <(python3 - "$MODELS_JSON" "$WANT_MODELS" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1])); want = None if sys.argv[2] == "all" else set(sys.argv[2].split(","))
for l in m["legs"]:
    if want is None or l["key"] in want:
        print("|".join([l["key"], l["file"], str(l["bytes"]), l["sha256"], l["label"]]))
PYEOF
)
if [ "$TASKS" = all ]; then ROWS_PER_LEG=160; else ROWS_PER_LEG=$(( $(echo "$TASKS" | tr ',' '\n' | wc -l) * 4 )); fi
EXPECT_N=$(( ${#LEGS[@]} * ROWS_PER_LEG ))

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0} smoke=$SMOKE stub=${STUB:-none} legs=${#LEGS[@]} tasks=$TASKS rows_per_leg=$ROWS_PER_LEG expect=$EXPECT_N bridge=$BRIDGE)"
nl_check_path "$BIN" "$CODE/run_units.py" "$CODE/score.py" "$CODE/ucommon.py" "$CODE/freeze.py" "$CODE/tasks.json" "$CODE/schemas.json" "$CODE/models.json" "$CODE/FROZEN.sha256" "$CODE/CODE.sha256" "$CODE/PREREG.md" \
  "$CODE/mock_units.py" "$CODE/aggregate_selftest.py" "$CODE/render_orbit.py" "$CODE/selftest_render.sh" "$CODE/make_mock_results.py" "$CODE/bridge.py" "$ANC/data/generated/views_benchmark.jsonl" "$ANC/src/num_equiv/parse.py" \
  "$ANC/artifacts/full/qwen35-9b-q4/eval.jsonl" "$ANC/artifacts/full/qwen35-9b-q8/eval.jsonl" /home/witcheer/esoteric/units/model-sha256.txt
if [ -z "$STUB" ]; then
  for L in "${LEGS[@]}"; do
    IFS='|' read -r K F B S LB <<< "$L"
    SZ=$(stat -c %s "$F" 2>/dev/null || echo 0)
    [ "$SZ" = "$B" ] && nl_say "  ok: $K file $SZ bytes" || nl_fail "$K model file size $SZ != $B ($F)"
    # the sha256 recorded when the files were fetched (esoteric/units/model-sha256.txt) must equal the frozen models.json value
    REL=${F#/home/witcheer/models/}
    REC=$(grep -F " $REL" /home/witcheer/esoteric/units/model-sha256.txt | cut -d' ' -f1)
    [ "$REC" = "$S" ] && nl_say "  ok: $K sha256 on record equals models.json (${S:0:12})" || nl_fail "$K sha256 on record '${REC:0:12}' != models.json '${S:0:12}'"
  done
fi
HELP=$($BIN --help 2>&1 || true)
for f in --jinja --parallel --flash-attn --ctx-size --alias --n-gpu-layers; do case "$HELP" in *"$f"*) ;; *) nl_fail "flag $f not in this build";; esac; done
nl_say "  ok: server flags exist in the build"
VER=$($BIN --version 2>&1 || true)
case "$VER" in *839a572*) nl_say "  ok: llama.cpp 839a572a4 (build 11777)";; *) nl_fail "llama.cpp build changed or missing (expected 839a572)";; esac
nl_check_disk 20
ss -ltnH "sport = :$PORT" 2>/dev/null | grep -q . && nl_fail "port $PORT is already in use"
# frozen task set + code hashes + CPU proofs at every DRYRUN (the item refuses to run when one is red)
cd "$CODE" || nl_fail "cannot cd $CODE"
(sha256sum -c FROZEN.sha256 > /dev/null 2>&1 && python3 freeze.py --check > /dev/null 2>&1) && nl_say "  ok: frozen task set unchanged (sha256sum -c FROZEN.sha256 + tasks.json matches tasks_def.py and schemas.json)" || nl_fail "frozen task set changed (FROZEN.sha256 / tasks.json)"
sha256sum -c CODE.sha256 > /dev/null 2>&1 && nl_say "  ok: driver, classifier, scorer, bridge and renderer match CODE.sha256" || nl_fail "code changed since CODE.sha256 was written (re-run freeze.py after a reviewed fix)"
PV=$(python3 -c "import ucommon;print(ucommon.PARSER_VERSION)" 2>/dev/null || echo none)
[ "$PV" = "units-parse-v1" ] && nl_say "  ok: parser $PV" || nl_fail "parser version '$PV' != units-parse-v1 (PREREG says v1; bump both deliberately)"
python3 - <<'PYEOF' || nl_fail "tasks.json check: counts, real-schema provenance or the prompt leak scan failed"
import json, re, sys
t = json.load(open("tasks.json"))
assert t["n_tasks"] == 40 and t["n_forms"] == 160, (t["n_tasks"], t["n_forms"])
assert sum(1 for x in t["tasks"] if x["real"]) == 24 and sum(1 for x in t["tasks"] if not x["real"]) == 16
for name, tl in t["tools"].items():
    if tl["real"]:
        assert tl["hermes_pin"] == "9bda188" and tl["file"].startswith("tools/") and tl["schema_line"] and tl["arg_line"], name
for x in t["tasks"]:
    for f in x["forms"]:
        if re.search(r"/home/|/Users/|witcheer|orbit|invarian|experiment|benchmark|magnitude|synthetic|unit conversion", f["prompt_user"], re.I):
            sys.exit("prompt of %s/%s names something it must not" % (x["id"], f["role"]))
PYEOF
nl_say "  ok: 40 tasks x 4 forms, 24 real + 16 synthetic, real schemas carry pin 9bda188 + file + line, no prompt names the experiment or a path"
CH=$(cd /home/witcheer/.hermes/hermes-agent && git rev-parse --short HEAD 2>/dev/null)
case "$CH" in 9bda188*) nl_say "  ok: Hermes checkout $CH (the schemas were extracted at this pin)";; *) nl_say "  note: Hermes checkout is now $CH, the frozen schemas come from 9bda188 (schemas.json is the record)";; esac
if [ "${UNITS_NO_SELFTEST:-0}" != 1 ]; then
  ss -ltnH "sport = :8651" 2>/dev/null | grep -q . && nl_fail "port 8651 (selftest mock) in use"
  python3 aggregate_selftest.py > /home/witcheer/scratch/units-selftest-dry.log 2>&1 && nl_say "  ok: mock proof: driver (resume, deadline, dead server), exact-rational classifier, verdict branches, line 1, McNemar: $(tail -n 1 /home/witcheer/scratch/units-selftest-dry.log)" || nl_fail "aggregate_selftest red (see ~/scratch/units-selftest-dry.log)"
  rm -rf /home/witcheer/scratch/units-mock-dry
  python3 make_mock_results.py /home/witcheer/scratch/units-mock-dry 8652 > /home/witcheer/scratch/units-mockres-dry.log 2>&1 && bash selftest_render.sh /home/witcheer/scratch/units-mock-dry/results /home/witcheer/scratch/units-mock-dry/mock-models.json > /home/witcheer/scratch/units-render-dry.log 2>&1 \
    && nl_say "  ok: render proof on mock data: $(tail -n 1 /home/witcheer/scratch/units-render-dry.log)" || nl_fail "render proof red (see ~/scratch/units-render-dry.log)"
fi
nl_say "  params: legs=$(printf '%s ' "${LEGS[@]%%|*}") sampling temp0/top_k1/seed20260725 ctx=$CTX port=$PORT max_tokens 512 (gpt-oss 1024)"
if [ -f "$ART" ] && nl_skip_exact "$ART"; then nl_say "nothing to run, item complete ($ART exists)"; exit 0; fi
DL=$(nl_deadline_epoch); LEFT=$(( (DL-$(date +%s))/60 ))
nl_say "  planned RUN: $SLUG ($EXPECT_N rows over ${#LEGS[@]} legs; hard stop $(date -d "@$DL" '+%a %H:%M') local, $LEFT min away)"
if [ "$SMOKE" != 1 ] && [ -z "$STUB" ]; then
  [ "$LEFT" -ge 45 ] || { nl_say "  under 45 min left in the window: not starting (five loads + 800 calls would be cut short; exit 3, resumable)"; exit 3; }
fi
if [ "${DRYRUN:-0}" = "1" ] || [ ! -f "$MANIFEST" ]; then
  PH=(); for L in "${LEGS[@]}"; do PH+=("leg-${L%%|*}"); done
  nl_manifest_init serve "${PH[@]}" bridge aggregate render
fi
nl_gate

SRV=""; DONALD_STOPPED=0; DONALD_BACK=0
killsrv() { { [ -n "$SRV" ] && kill -9 "$SRV" && wait "$SRV"; } 2>/dev/null; SRV=""; }              # only the server THIS item started
donald_stop() { nl_say "draining Donald (nl_donald_stop)"; nl_donald_stop; DONALD_STOPPED=1; }
donald_restore() {
  [ "$DONALD_STOPPED" = 1 ] && [ "$DONALD_BACK" = 0 ] || return 0
  DONALD_BACK=1; nl_say "restoring Donald (nl_donald_restore)"
  nl_donald_restore
  for i in $(seq 1 60); do H=$(curl -s -m 5 localhost:8090/health); [[ "${H,,}" == *ok* ]] && { nl_say "Donald healthy again after ~$((i*5)) s"; return 0; }; sleep 5; done
  nl_say "WARNING: Donald /health did not answer within 5 min after the restore"
}
cleanup() { killsrv; sleep 2; donald_restore; }
trap cleanup EXIT
mem_clear() { # poll card memory under 1 GiB; 0 = clear
  local v=99999 i
  for i in $(seq 1 60); do v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${v:-99999}" -lt 1024 ] && { nl_say "card clear (${v} MiB)"; return 0; }; sleep 2; done
  nl_say "card memory still ${v:-?} MiB"; return 1
}
mkdir -p "$RES" "$OUT"
t0=$(date +%s)
nl_mark serve running
if [ -n "$STUB" ]; then
  UP_PORT=${STUB##*:}; nl_mark serve done "stub upstream $STUB"
else
  donald_stop
  mem_clear || { nl_mark serve failed "card memory not clear after the drain: another process owns the card"; nl_say "FAIL: card memory not clear after the drain, not loading"; exit 1; }
  UP_PORT=$PORT; nl_mark serve done "Donald drained; one fresh server per leg on $PORT"
fi
DEADLINE=$(nl_deadline_epoch)
STOPAT=$(( DEADLINE - 420 ))                            # 7 min for the kill, the Donald restore, the aggregation and the render
[ "$SMOKE" = 1 ] && STOPAT=$(( DEADLINE - ${UNITS_SMOKE_MARGIN:-60} ))

serve_leg() { # key file label -> 0 when /health answers; writes fit.jsonl + props
  local key=$1 file=$2 log=$RUN/server-$1-$(date +%H%M).log up=0 used tot i
  # shellcheck disable=SC2086
  $BIN --model "$file" --host 127.0.0.1 --port $PORT --alias "$key" --jinja --parallel 1 --n-gpu-layers 99 --ctx-size $CTX --flash-attn on > "$log" 2>&1 &
  SRV=$!
  for i in $(seq 1 240); do H=$(curl -s -m 3 127.0.0.1:$PORT/health); [[ "${H,,}" == *ok* ]] && { up=1; break; }; kill -0 $SRV 2>/dev/null || break; sleep 2; done
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); tot=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
  echo "{\"model\":\"$key\",\"up\":$([ "$up" = 1 ] && echo true || echo false),\"used_mib\":${used:-0},\"free_mib\":$(( ${tot:-0} - ${used:-0} )),\"ctx\":$CTX}" >> "$RUN/fit.jsonl"
  if [ "$up" != 1 ] || [ $(( tot - used )) -lt 1500 ]; then nl_say "  FAIL: $key server up=$up used=$used free=$(( tot - used )): $(grep -m1 -iE 'out of memory|failed|error' "$log" | head -c 200)"; return 1; fi
  curl -s 127.0.0.1:$PORT/props > "$RUN/props-$key.json"
  nl_say "  $key up: ${used} MiB used, $(( tot - used )) free, template in props: $(python3 -c "import json,hashlib;j=json.load(open('$RUN/props-$key.json'));t=j.get('chat_template') or '';print(len(t),'chars sha',hashlib.sha256(t.encode()).hexdigest()[:10])" 2>/dev/null || echo '?')"
  return 0
}

count_rows() { [ -f "$RES/$1.jsonl" ] && grep -c . "$RES/$1.jsonl" || echo 0; }
rc_all=0; legs_left=0
for L in "${LEGS[@]}"; do
  IFS='|' read -r K F B S LB <<< "$L"
  have=$(count_rows "$K")
  if [ "$have" -ge "$ROWS_PER_LEG" ]; then nl_say "leg $K: $have/$ROWS_PER_LEG rows, done"; nl_mark "leg-$K" done "$have rows (earlier block)"; continue; fi
  now=$(date +%s); left_min=$(( (STOPAT - now) / 60 ))
  need=10; case "$K" in qwen38*|ornith*) need=12;; esac
  if [ "$SMOKE" != 1 ] && [ -z "$STUB" ] && [ "$left_min" -lt "$need" ]; then nl_say "leg $K: only $left_min min left before the stop (needs $need): not started"; nl_mark "leg-$K" paused "$have rows, not started"; legs_left=1; continue; fi
  [ "$now" -ge "$STOPAT" ] && { nl_say "leg $K: past the stop time"; nl_mark "leg-$K" paused "$have rows"; legs_left=1; continue; }
  nl_mark "leg-$K" running "$have of $ROWS_PER_LEG rows"
  l0=$(date +%s)
  if [ -z "$STUB" ]; then
    serve_leg "$K" "$F" || { killsrv; nl_mark "leg-$K" failed "server did not come up"; mem_clear; rc_all=1; continue; }
  fi
  python3 "$CODE/run_units.py" --model "$K" --port "$UP_PORT" --out "$RES" --models "$MODELS_JSON" --tasks "$TASKS" --deadline-epoch "$STOPAT" >> "$RUN/run.log" 2>&1
  rrc=$?
  have=$(count_rows "$K")
  nl_say "  leg $K runner rc=$rrc: $have/$ROWS_PER_LEG rows in $(( $(date +%s) - l0 )) s: $(tail -n 1 "$RUN/run.log" | cut -c1-160)"
  [ -z "$STUB" ] && { killsrv; sleep 2; mem_clear || { nl_say "FAIL: card memory did not clear after $K, stopping the item"; rc_all=1; break; }; }
  if [ "$have" -ge "$ROWS_PER_LEG" ]; then nl_mark "leg-$K" done "$have rows in $(( $(date +%s) - l0 )) s"; else nl_mark "leg-$K" paused "$have of $ROWS_PER_LEG rows (rc $rrc)"; legs_left=1; [ "$rrc" = 4 ] && rc_all=1; fi
done

# ---- optional bridge leg: runtime parity with the paper's per-item answers on the paper's exact Qwen3.5-9B files ----
BR_DONE=0
if [ "$BRIDGE" = 1 ] && [ "$legs_left" = 0 ] && [ "$rc_all" = 0 ]; then
  left_min=$(( (STOPAT - $(date +%s)) / 60 ))
  if [ "$SMOKE" != 1 ] && [ -z "$STUB" ] && [ "$left_min" -lt 25 ]; then nl_say "bridge: only $left_min min left (needs 25): skipped"; nl_mark bridge skipped "under 25 min left"
  else
    nl_mark bridge running "n=$BRIDGE_N per view"
    for pair in "qwen35-9b-q4km:qwen35-9b-q4" "qwen35-9b-q8_0:qwen35-9b-q8"; do
      K=${pair%%:*}; SYS=${pair##*:}
      [ "$SMOKE" = 1 ] && [ "$WANT_MODELS" != all ] && case ",$WANT_MODELS," in *",$K,"*) ;; *) continue;; esac
      F=$(python3 -c "import json;print(next(l['file'] for l in json.load(open('$MODELS_JSON'))['legs'] if l['key']=='$K'))")
      if [ -z "$STUB" ]; then serve_leg "$K" "$F" || { killsrv; nl_say "  bridge: $K server failed"; continue; }; fi
      python3 "$CODE/bridge.py" --anc "$ANC" --system "$SYS" --port "$UP_PORT" --out "$OUT" --n "$BRIDGE_N" --deadline-epoch "$STOPAT" >> "$RUN/bridge.log" 2>&1
      nl_say "  bridge $SYS rc=$?: $(tail -n 1 "$RUN/bridge.log" | cut -c1-160)"
      [ -z "$STUB" ] && { killsrv; sleep 2; mem_clear || break; }
    done
    BR_DONE=1; nl_mark bridge done "n=$BRIDGE_N per view"
  fi
else
  nl_mark bridge skipped "bridge=$BRIDGE legs_left=$legs_left"
fi
killsrv; sleep 2
[ -z "$STUB" ] && donald_restore
nl_say "card released after $(( ($(date +%s)-t0)/60 )) min"

# ---- aggregate + compare + render (CPU) ----
nl_mark aggregate running
python3 "$CODE/score.py" --res "$RES" --out "$OUT" --models "$MODELS_JSON" >> "$RUN/aggregate.log" 2>&1
if [ "$BR_DONE" = 1 ]; then for SYS in qwen35-9b-q4 qwen35-9b-q8; do [ -f "$OUT/bridge-$SYS.jsonl" ] && python3 "$CODE/bridge.py" --compare --anc "$ANC" --system "$SYS" --out "$OUT" >> "$RUN/aggregate.log" 2>&1; done; fi
AGG=$(python3 -c "import json;d=json.load(open('$OUT/aggregate.json'));print(d['overall'], 'prov' if d['provisional'] else 'final', {k:(v['verdict'],v['counts']['magnitude_slip'],v['n']) for k,v in d['models'].items() if v['verdict']!='NOT RUN'})" 2>/dev/null || echo 'none yet')
nl_say "  aggregate: $AGG"
NRES=0; for L in "${LEGS[@]}"; do NRES=$(( NRES + $(count_rows "${L%%|*}") )); done
nl_say "rows so far: $NRES of $EXPECT_N"
if [ "$NRES" -ge "$EXPECT_N" ] && [ "$legs_left" = 0 ] && [ "$rc_all" = 0 ]; then
  nl_mark aggregate done "$NRES rows"
  if [ "$SMOKE" != 1 ]; then
    cp "$OUT/aggregate.json" "$ART"
    nl_judge_artifact aggregate 0 "$ART" "$t0" "$NRES rows, $(( ($(date +%s)-t0)/60 )) min this block" || { nl_say "FAIL: no fresh summary.json"; exit 1; }
    nl_say "  RULE: $(python3 -c "import json;d=json.load(open('$ART'));print(d['overall'], '|', d['line1'])")"
    # the picture: CPU only, after the card is released
    nl_mark render running
    MOCKFLAG=""; [ -n "$STUB" ] && MOCKFLAG="--mock"
    ( cd "$CODE" && nice python3 render_orbit.py --res "$RES" --models "$MODELS_JSON" $MOCKFLAG --out "$OUT/units-orbit.mp4" >> "$RUN/render.log" 2>&1 ) && FFB=$(cd "$CODE" && python3 -c "import render_orbit as r;print(r.ffmpeg_bin())") \
      && "$FFB" -nostdin -y -loglevel error -i "$OUT/units-orbit.mp4" -vf "fps=1/6,scale=480:-1,tile=4x3" -frames:v 1 "$OUT/units-orbit-contact.png" </dev/null \
      && nl_mark render done "$(stat -c %s "$OUT/units-orbit.mp4") bytes" && nl_say "  render: $OUT/units-orbit.mp4 + units-orbit-contact.png" || { nl_mark render failed "see $RUN/render.log"; nl_say "WARNING: render failed (see $RUN/render.log); the numbers are in $ART"; }
    nl_say "DONE $SLUG"
  else
    nl_say "smoke complete (rows are never data): $OUT/readout.md"
  fi
  exit 0
fi
nl_mark aggregate paused "$NRES of $EXPECT_N rows; resumable"
nl_say "legs remain ($NRES of $EXPECT_N): exit 3, the item stays queued"
[ "$rc_all" = 0 ] || { nl_say "FAIL: a leg failed (see $RUN/run.log and the server logs)"; exit 1; }
exit 3
