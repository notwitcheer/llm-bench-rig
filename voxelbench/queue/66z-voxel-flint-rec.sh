#!/bin/bash
# 66z-voxel-flint-rec.sh: idle-queue item (2026-10-22 ~11:00 CEST, mercury, GPU thread; operator: "yes queue the flint second pass
# on voxel with its own recommended settings"). second pass of Flint 40B-A2.5B (unsloth UD-Q4_K_XL) on the series prompt msm-v2.
# first pass (sibling release-day run, ~/day0-voxel.sh, 2026-10-22 08:51-09:25 CEST, slug flint-ud-q4kxl-msm-v2): gate PASS 5/5
# but a weak scene (29,527 html bytes vs 660,756 for the Qwen3.8-27B Q6_K msm-v2 reference). day0-voxel.sh passed no sampling
# flags, BUT the GGUF carries general.sampling.temp 1.0 / top_k 20 / top_p 0.95 and this build applies them as defaults
# (common.cpp, unless a flag overrides): CPU /props probe 2026-10-22 11:05 CEST (GPU hidden, real Flint file, no flags) =
# temp 1.0, top_k 20, top_p 0.95, min_p 0.05 (llama.cpp default; the GGUF has no min_p key). so pass 1 already had the card's
# temp/top-k/top-p; it differed in min_p 0.05 and in our Qwen-tuned 16,384 reasoning budget, hit at least 5 times.
# ONE condition changes vs the first pass, "maker settings", from the model card (huggingface.co/RecursiveSI/flint and
# huggingface.co/unsloth/flint-GGUF, fetched 2026-10-22): "Temperature 1.0 | Top-p / top-k 0.95 / 20 | Output budget for hard
# problems: At least 32768 tokens, including reasoning" and "preserve returned reasoning_content in conversation history".
#   server flags: --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 (explicit; temp/top-k/top-p equal pass 1's effective values,
#   the card names no min_p, 0 = the vLLM/SGLang default vs llama.cpp's 0.05) and --reasoning-budget 32768 (budget message kept).
#   so the effective deltas are the output budget 16,384 -> 32,768 and min_p 0.05 -> 0. requests carry no max_tokens
#   (n_predict -1); the server log already shows the template preserves reasoning.
# everything else = the first pass: same file, same build (~/llama.cpp-flint b11810 e3797b184, pin-checked), ctx 131072, q8_0 KV,
# fa, --parallel 1, same msm-v2 prompt + exact-path line, same harness (run_agent.sh, fresh bench profile, 60 min wall, 20 goal
# turns), gate + metrics. sampling reach verified 2026-10-22 10:52 CEST (CPU smoke ~/flint-samp-smoke.sh: Hermes' main-loop body
# carries NO temperature/top_p/top_k/min_p/max_tokens, only reasoning_effort "medium", which this template ignores; /slots after the
# request = temp 1.0 top_k 20 top_p 0.95 min_p 0). this item re-checks /props before the run and traces /slots every 60 s.
# on a gate pass: orbit showcase with the page's own drag handler measured by orbit_pick.py (degrees vs radians, px per turn;
# >1,600 px per turn -> orbit_clip_multidrag.py), keys n@11,n@22.
# artifact: runs/<slug>/gate/gate.json (exact). bench profiles are NEVER deleted (host gateway holds their state.db).
set -uo pipefail
NL_TAG=voxflintrec
MANIFEST=/home/witcheer/voxel-flint-rec-status.json
# throwaway manifest for a staging DRYRUN only (the drain greps the literal ^MANIFEST= line above)
[ -n "${MANIFEST_OVERRIDE:-}" ] && MANIFEST=$MANIFEST_OVERRIDE
PVER=msm-v2
export MANIFEST
. /home/witcheer/night-lib.sh
V=/home/witcheer/voxelbench
BIN=/home/witcheer/llama.cpp-flint/build/bin/llama-server
PIN=e3797b184
GPY=/home/witcheer/benchmark-rig/.venv/bin/python
RUNLOG=/home/witcheer/voxel-flint-rec-run.log
SLUG=flint-ud-q4kxl-$PVER-rec
MODEL_FILE=/home/witcheer/models/flint/flint-UD-Q4_K_XL.gguf
ALIAS=flint-UD-Q4_K_XL.gguf
CTX=131072; WALL=60; TURNS=20; KV=q8_0
# the one condition: maker settings (sampling + output budget)
SAMP="--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0"
RB=32768
RB_MSG="Thinking budget reached. Stop planning and write the files now with your tools; refine in later turns."
ART=$V/runs/$SLUG/gate/gate.json

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0}) slug=$SLUG samp='$SAMP' rb=$RB ctx=$CTX kv=$KV"
nl_check_path "$BIN" "$GPY" "$MODEL_FILE" "$V/scripts/run_agent.sh" "$V/scripts/gate.py" "$V/scripts/metrics.py" \
  "$V/prompts/$PVER.txt" "$V/scripts/orbit_clip.py" "$V/scripts/orbit_clip_multidrag.py" "$V/scripts/orbit_pick.py" \
  "$V/lib/libasound.so.2" "$V/ffmpeg" /home/witcheer/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome
HELP=$("$BIN" --help 2>&1 || true)
FMISS=""
for want in --reasoning-budget-message --temp --top-p --top-k --min-p --cache-type-k --flash-attn --metrics; do
  case "$HELP" in *"$want"*) ;; *) FMISS="$FMISS $want"; nl_fail "flag $want missing from build";; esac
done
[ -z "$FMISS" ] && nl_ok "flags in build: --reasoning-budget(-message) --temp --top-p --top-k --min-p --cache-type-k --flash-attn --metrics"
VER=$("$BIN" --version 2>&1 | grep -m1 -i 'version' | tr -d '"')
case "$VER" in *"$PIN"*) nl_say "  ok: llama.cpp $VER (pin $PIN)";; *) nl_say "  llama.cpp build changed or missing: '$VER' (want $PIN)"; exit 2;; esac
# first-pass comparison inputs must exist (the readout diffs against them)
nl_check_path "$V/runs/flint-ud-q4kxl-msm-v2/gate/gate.json" "$V/runs/qwen38-27b-q6k-msm2/gate/gate.json"
nl_check_disk 20
if nl_skip_exact "$ART"; then nl_say "nothing to run, item complete"; exit 0; fi
DL=$(nl_deadline_epoch); LEFT=$(( (DL-$(date +%s))/60 ))
nl_say "  planned RUN: $SLUG (wall $WALL min + ~3 min gate + ~2 min clip; hard stop $(date -d "@$DL" '+%a %H:%M') local, $LEFT min away)"
[ "$LEFT" -ge 75 ] || { nl_say "  under 75 min left in the window: not starting (would cut the run short)"; exit 3; }
if [ "${DRYRUN:-0}" = "1" ] || [ ! -f "$MANIFEST" ]; then nl_manifest_init "$SLUG"; fi
nl_gate

TRACER=""
restore() { [ -n "$TRACER" ] && kill "$TRACER" 2>/dev/null; tmux kill-session -t "=voxel-$SLUG" 2>/dev/null; pkill -9 -x llama-server 2>/dev/null; sleep 5; nl_say "EXIT trap: restoring Donald"; nl_donald_restore; sleep 5; curl -s -m5 127.0.0.1:8090/health | grep -qi ok && nl_say "donald health ok" || nl_say "WARN: donald health not ok after restore"; }
trap restore EXIT
DEADLINE=$(nl_deadline_epoch)
nl_say "draining Donald"; nl_donald_stop
# Flint needs ~26.5 GB of 32: wait for the card to be really empty (a sibling engine's VRAM would OOM the load)
CLEAR=0
for i in $(seq 1 90); do v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${v:-99999}" -lt 1024 ] && { nl_say "drain clear (${v} MiB)"; CLEAR=1; break; }; sleep 2; done
if [ "$CLEAR" != 1 ]; then
  nl_say "  card not clear after drain (${v} MiB): $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\n' ';') -> not loading, retry next block"
  exit 3
fi
[ "$(ls -A $V/runs/$SLUG/work 2>/dev/null)" ] && { nl_say "  stale work dir from an earlier attempt: moving aside"; mv $V/runs/$SLUG $V/runs/$SLUG.stale-$(date +%s); }
mkdir -p $V/runs/$SLUG

t0=$(date +%s); nl_mark "$SLUG" running
# shellcheck disable=SC2086
$BIN --model "$MODEL_FILE" --host 127.0.0.1 --port 8090 --n-gpu-layers 99 --ctx-size $CTX --parallel 1 --flash-attn on \
  --cache-type-k $KV --cache-type-v $KV --alias "$ALIAS" --metrics $SAMP --reasoning-budget $RB --reasoning-budget-message "$RB_MSG" \
  > $V/runs-$SLUG-server.log 2>&1 &
for i in $(seq 1 150); do curl -s 127.0.0.1:8090/health | grep -qi ok && break; sleep 2; done
curl -s 127.0.0.1:8090/health | grep -qi ok || { nl_mark "$SLUG" failed "server did not come up (see runs-$SLUG-server.log)"; exit 1; }
nl_say "server up ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader)), ctx $CTX"
# sampling check BEFORE any agent work: the server defaults must be the maker settings
curl -s 127.0.0.1:8090/props > $V/runs/$SLUG/props.json
SCHK=$(python3 - "$V/runs/$SLUG/props.json" <<'PY'
import json, sys
g = json.load(open(sys.argv[1])).get("default_generation_settings", {}); p = g.get("params", g)
want = {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0}
got = {k: p.get(k) for k in want}
ok = all(got[k] is not None and abs(float(got[k]) - v) < 1e-3 for k, v in want.items())
print(("OK " if ok else "BAD ") + json.dumps(got))
PY
)
nl_say "  server sampling defaults: $SCHK"
case "$SCHK" in OK*) ;; *) nl_mark "$SLUG" failed "server sampling defaults wrong: $SCHK"; exit 1;; esac
PMD5=$(md5sum $V/prompts/$PVER.txt | cut -c1-12)
echo "{\"model_file\": \"$MODEL_FILE\", \"ctx\": $CTX, \"kv\": \"$KV\", \"extra\": \"\", \"sampling\": \"$SAMP\", \"reasoning_budget\": $RB, \"build\": \"$VER\", \"prompt\": \"$PVER.txt\", \"prompt_md5\": \"$PMD5\", \"condition\": \"maker settings (model card 2026-10-22)\"}" > $V/runs/$SLUG/run-config.json
# per-request sampling trace: /slots every 60 s while the agent runs (params of the live or last task)
( while :; do curl -s -m5 127.0.0.1:8090/slots 2>/dev/null | python3 -c "
import json,sys,time
try:
    d=json.load(sys.stdin); s=d[0] if isinstance(d,list) else d; p=s.get('params',{})
    print(json.dumps({'t':int(time.time()),'processing':s.get('is_processing'),'id_task':s.get('id_task'),**{k:p.get(k) for k in ('temperature','top_k','top_p','min_p','n_predict')}}))
except Exception as e: print(json.dumps({'t':int(time.time()),'err':str(e)[:80]}))
" >> $V/runs/$SLUG/sampling-trace.jsonl; sleep 60; done ) &
TRACER=$!
export DEADLINE_EPOCH=$(( DEADLINE - 420 ))
PROMPT_RUN=$V/runs/$SLUG/prompt.txt
{ cat $V/prompts/$PVER.txt; printf '\nThe working directory is %s. The final file must be exactly %s/index.html (copy this path, never retype it).\n' "$V/runs/$SLUG/work" "$V/runs/$SLUG/work"; } > "$PROMPT_RUN"
bash $V/scripts/run_agent.sh "$SLUG" "$ALIAS" $CTX $WALL $TURNS "$PROMPT_RUN" >> "$RUNLOG" 2>&1
nl_say "driver: $(cat $V/runs/$SLUG/driver.json 2>/dev/null)"
kill "$TRACER" 2>/dev/null; TRACER=""
nl_say "  sampling trace: $(python3 -c "
import json
rows=[json.loads(l) for l in open('$V/runs/$SLUG/sampling-trace.jsonl') if l.strip()]
live=[r for r in rows if r.get('temperature') is not None]
sets=sorted({(r['temperature'],r['top_k'],round(r['top_p'],3),r['min_p']) for r in live})
print(len(rows),'samples,',len(live),'with params, distinct (temp,top_k,top_p,min_p):',sets)
" 2>&1 | head -c 300)"
curl -s 127.0.0.1:8090/metrics | grep -E "^llamacpp:(tokens_predicted_total|prompt_tokens_total|tokens_predicted_seconds_total|prompt_seconds_total)" > $V/runs/$SLUG/server-metrics.txt 2>/dev/null
pkill -9 -x llama-server 2>/dev/null; sleep 3
HTML=$V/runs/$SLUG/work/index.html
[ -f "$HTML" ] || HTML=$(ls -t $V/runs/$SLUG/work/*.html 2>/dev/null | head -1)
if [ -z "$HTML" ] || [ ! -f "$HTML" ]; then
  STRAY=$(find $V/runs -name '*.html' -newermt "@$t0" -size +1k -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
  if [ -n "$STRAY" ]; then HTML=$STRAY; echo "{\"path_miss\": true, \"found\": \"$STRAY\"}" > $V/runs/$SLUG/path-miss.json; nl_say "  PATH MISS: no html in work/, grading the fresh file at $STRAY (flagged)"; fi
fi
mkdir -p $V/runs/$SLUG/gate
if [ -n "$HTML" ] && [ -f "$HTML" ]; then
  setsid timeout 300 $GPY $V/scripts/gate.py "$HTML" $V/runs/$SLUG/gate --video-seconds 20 < /dev/null >> "$RUNLOG" 2>&1; grc=$?
else
  echo '{"pass": false, "no_html": true}' > "$ART"; grc=1
fi
python3 $V/scripts/metrics.py /home/witcheer/.hermes/profiles/$(python3 -c "import json;print(json.load(open('$V/runs/$SLUG/driver.json'))['profile'])" 2>/dev/null || echo vx${SLUG//[^a-z0-9]/}) $V/runs/$SLUG/metrics.json >> "$RUNLOG" 2>&1
if nl_judge_artifact "$SLUG" 0 "$ART" "$t0" "$(( ($(date +%s)-t0)/60 )) min, gate rc $grc"; then
  nl_say "  gate: $(python3 -c "import json;d=json.load(open('$ART'));print('pass' if d.get('pass') else 'FAIL', d.get('checks'), 'fps', d.get('fps'), 'bytes', d.get('bytes'))")"
  if python3 -c "import json,sys;sys.exit(0 if json.load(open('$ART')).get('pass') else 1)"; then
    # read the page's own drag handler (units + px per turn), then render with the matching driver
    mkdir -p $V/runs/$SLUG/showcase
    setsid timeout 120 $GPY $V/scripts/orbit_pick.py "$HTML" --clip-seconds 30 < /dev/null 2>>"$RUNLOG" | tail -1 > $V/runs/$SLUG/showcase/orbit-choice.json
    OPX=$(python3 -c "import json;print(json.load(open('$V/runs/$SLUG/showcase/orbit-choice.json'))['orbit_px'])" 2>/dev/null || echo 1257)
    OSC=$(python3 -c "import json;print(json.load(open('$V/runs/$SLUG/showcase/orbit-choice.json'))['script'])" 2>/dev/null || echo orbit_clip.py)
    case "$OSC" in orbit_clip.py|orbit_clip_multidrag.py) ;; *) OSC=orbit_clip.py;; esac
    nl_say "  orbit choice: $OSC --orbit-px $OPX ($(python3 -c "import json;d=json.load(open('$V/runs/$SLUG/showcase/orbit-choice.json'));print(d.get('method'),d.get('var'),d.get('units'))" 2>/dev/null))"
    setsid timeout 240 $GPY $V/scripts/$OSC "$HTML" $V/runs/$SLUG/showcase --seconds 30 --orbit-px "$OPX" --keys "n@11,n@22" < /dev/null >> "$RUNLOG" 2>&1 \
      && nl_say "  showcase: $V/runs/$SLUG/showcase/orbit.mp4" || nl_say "  showcase clip failed (gate result stands)"
  fi
fi
nl_say "DONE $SLUG"
exit 0
