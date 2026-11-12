#!/bin/bash
# 99z-fp8-kv-race.sh: idle-queue item (2026-11-12). FP8 (f8_e4m3) KV cache vs f16 KV, same Qwen3.8-27B Q6_K file, same llama.cpp build (09338b85), same Hermes task.
# WHAT: model ~/models/qwen38-27b/Qwen3.8-27B-Q6_K.gguf (22,884,408,288 B, on disk since 2026-08-14). arms: f16 (default KV) vs fp8 (-ctk f8_e4m3 -ctv f8_e4m3).
#   measures: VRAM at the same context (nvidia-smi after load), max context that fits (short sweep, both arms), decode tok/s (server timings), Hermes task passes.
#   headline leg = 3 takes per arm on the same task (4 planted bugs, 5 tests) ~40 min; extension: a long-context leg at the max the f16 arm cannot fit.
# MOJO: two Hermes panes on one 5090, result card first: VRAM and max-context numbers, then the live race | intense (a memory claim checked on our card) |
#   "fp8 kv cache on qwen3.8-27b, one 5090: [X] GB vs [Y] GB at 65k, passes [N] vs [M]"
# EV: private (projected post angle and reach bracket live in the ops vault status note)
#
# PRE-REGISTERED (99z harness/PREREG.md, sha256 in FROZEN.sha256, BEFORE any counted output; this item refuses to run when a frozen hash differs; wf_summary.py --selftest proves 8 branches).
#   take CLEAN = ended done or wall, >= 3 tool calls, request errors <= 20%. take PASS = CLEAN and pristine tests 5 passed 0 failed 0 errors, tests/ untouched.
#   verdict: BROKEN | INCOMPLETE (< 2 CLEAN takes in an arm: exit 3) | TIE (same pass count) | CHANGED | NEW MORE/FEWER PASSES (direction only at >= 2 passes and >= 8 takes per arm).
#   a TIE is a valid result: "fp8 KV did not change it on our task". VRAM, max context and decode tok/s are reported per arm, never gating.
# POSTABLE ARTEFACT: out/whichfile-final.mp4 (16:9 1920x1080, result card first, then the race, cut at the result) + out/contact.png + summary.json.
# CARD TIME (ESTIMATE, basis: 99y stub-proof shape + 99a take times, ~2 min per take + two loads + sweep): headline 6 takes ~25-35 min + loads ~6 min + max-context sweep ~6 min => ~40-50 min typical; worst ~80 (every take at the 10 min cap, sweep to the cap).
# DISK: 0 GB new download (the Q6_K file is already on disk, size-checked). the fp8 build (~/llama.cpp-fp8) is CPU compile output only.
# WINDOW RULE: needs >= 60 min left in the block or exits 3 without draining Donald; a take is skipped when < 11 min remain.
# GPU lock: flock -w 1800 /home/witcheer/.gpu-smoke.lock for the card section. hard stop = NL_HARD_STOP_EPOCH only (no own cap epoch).
#
# test knobs (empty in production): WF_SMOKE=1 (slug -smoke, 1 take per arm, 3 min wall, never counted), WF_HOLDS_LOCK=1, WF_TAKES, WF_EXT_TAKES, WF_WALL_MIN, WF_THINK=on|off,
#   WF_STUB_UPSTREAM / WF_STUB_UPSTREAM_OLD / WF_STUB_UPSTREAM_NEW (CPU mock per arm, no Donald drain, DRYRUN-safe), WF_MODEL (red-path tests), MANIFEST_OVERRIDE, WF_NO_SELFTEST=1.
set -uo pipefail
NL_TAG=kvfp8-99z
MANIFEST=/home/witcheer/kvfp8-99z-status.json
[ -n "${MANIFEST_OVERRIDE:-}" ] && MANIFEST=$MANIFEST_OVERRIDE
export MANIFEST
. /home/witcheer/night-lib.sh
H=/home/witcheer/voxel-stage/kvfp8-99z
RUNROOT=/home/witcheer/kvfp8-99z-runs
GPY=/home/witcheer/benchmark-rig/.venv/bin/python
RPY=/home/witcheer/scratch/race/venv/bin/python
FF=/home/witcheer/voxelbench/ffmpeg
CHECKOUT=/home/witcheer/.hermes/hermes-agent
HBIN=$CHECKOUT/.hermes/bin/hermes
HERMES_PIN=9bda188
LL=/home/witcheer/llama.cpp-fp8
SRV=$LL/build/bin/llama-server
BUILD_SHA=09338b85
MODEL=${WF_MODEL:-/home/witcheer/models/qwen38-27b/Qwen3.8-27B-Q6_K.gguf}
MODEL_BYTES=22884408288
LPORT=8277; PPORT=8293
CTXCFG=65536; THREADS=6
RB=8192; RB_MSG="Thinking budget reached. Act now with your commands."
PROMPT=$H/prompts/fix-shopcart.txt
SEEDDIR=$H/seed
H_FROZEN=d87ba8b7d26d5495b4ffc27a8480d4b1708a1bd6e3c1b220592a78aaf0011e274dddf0bf191ba581ca7ffef90990618dea8d4692576b0269
SMOKE=${WF_SMOKE:-0}
NT=${WF_TAKES:-3}; EXT=${WF_EXT_TAKES:-8}; WALLMIN=${WF_WALL_MIN:-10}; THINK=${WF_THINK:-on}; QUIET=${WF_QUIET:-20}
[ "$SMOKE" = 1 ] && { NT=${WF_TAKES:-1}; WALLMIN=${WF_WALL_MIN:-3}; EXT=$NT; }
SLUG=99z-fp8-kv-race; [ "$SMOKE" = 1 ] && SLUG=$SLUG-smoke
STUBO=${WF_STUB_UPSTREAM_OLD:-${WF_STUB_UPSTREAM:-}}; STUBN=${WF_STUB_UPSTREAM_NEW:-${WF_STUB_UPSTREAM:-}}
RUN=$RUNROOT/$SLUG
ART=$RUN/summary.json
REGIME="Qwen3.8-27B Q6_K (22,884,408,288 B), llama.cpp 09338b85 (FP8 PRs 28392+28413), -ngl 99, flash-attn on, --jinja, ctx $CTXCFG, one slot, $THREADS threads, thinking $THINK, --reasoning-budget $RB; arms: f16 KV vs f8_e4m3 KV (-ctk/-ctv f8_e4m3); Hermes terminal+file, approvals off, memory off; same task (4 planted bugs, 5 tests), $NT takes per arm; verdict = pass counts (TIE = same count)"

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0} smoke=$SMOKE takes=$NT+$NT ext=$EXT wall=${WALLMIN}min stub=${STUBO:-none})"
nl_check_path "$GPY" "$RPY" "$HBIN" "$FF" "$PROMPT" "$SEEDDIR/shopcart/cart.py" "$SEEDDIR/tests/test_cart.py" "$SRV" \
  "$H/mc_take.sh" "$H/rp_proxy.py" "$H/rp_count.py" "$H/rp_recount.py" "$H/rp_record.py" "$H/mc_mkhome.py" "$H/mc_render.py" "$H/wf_summary.py" "$H/wf_facts.py" "$H/rp_gate.py" \
  "$H/PREREG.md" "$H/FROZEN.sha256" "$H/SEED.sha256" /home/witcheer/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome
if [ -z "${WF_MODEL:-}" ]; then [ "$(stat -c %s "$MODEL" 2>/dev/null)" = "$MODEL_BYTES" ] && nl_ok "Q6_K file $MODEL_BYTES B" || nl_fail "Q6_K file size != $MODEL_BYTES"; fi
HELPL=$(CUDA_VISIBLE_DEVICES= $SRV --help 2>&1 || true)
for f in --jinja --parallel --ctx-size --alias --n-gpu-layers --flash-attn --cache-type-k --cache-type-v --metrics --threads --reasoning-budget --reasoning-budget-message; do
  [[ $HELPL == *"$f"* ]] || nl_fail "llama-server flag $f not in this build"
done
VER=$(CUDA_VISIBLE_DEVICES= $SRV --version 2>&1 || true)
PINSHA=$(cat "$LL/PINNED_SHA" 2>/dev/null)
[ "${PINSHA:0:8}" = "$BUILD_SHA" ] && nl_ok "llama.cpp source pin ${PINSHA:0:8} (PINNED_SHA; the tarball has no .git, so --version prints commit unknown)" || nl_fail "llama.cpp PINNED_SHA ${PINSHA:0:8} != $BUILD_SHA"
HF8=$(CUDA_VISIBLE_DEVICES= $SRV --help 2>&1 | grep -c "f8_e4m3" || true); [ "$HF8" -ge 1 ] && nl_ok "f8_e4m3 KV type in this build" || nl_fail "f8_e4m3 KV type NOT in this build (check --help for cache-type spelling)"
HV=$(cd "$CHECKOUT" && git rev-parse --short HEAD 2>/dev/null)
case "$HV" in "$HERMES_PIN"*) nl_ok "Hermes checkout $HV (pin $HERMES_PIN)";; *) nl_fail "Hermes checkout $HV != pin $HERMES_PIN";; esac
for dd in installs tools; do [ -d "/home/witcheer/.hermes/$dd" ] || nl_fail "main Hermes root has no $dd/"; done
nl_say "  regime: $REGIME"
nl_check_disk 125
for p in $LPORT $PPORT; do LS=$(ss -ltnH "sport = :$p" 2>/dev/null || true); [ -z "$LS" ] || nl_fail "port $p is already in use"; done
for t in kv-f16 kv-fp8; do tmux has-session -t "=$t" 2>/dev/null && nl_fail "tmux session $t already exists"; done
PT=$(cat "$PROMPT"); [[ $PT == */home/* || $PT == */Users/* || $PT == *witcheer* || $PT == *whichfile99y* ]] && nl_fail "prompt file names a home path, the account or the harness dir"
if nl_skip_exact "$ART"; then nl_say "nothing to run, item complete ($ART exists)"; exit 0; fi
DL=$(nl_deadline_epoch); LEFT=$(( (DL-$(date +%s))/60 ))
nl_say "  planned: $((NT*2)) headline takes + extension to $EXT per arm (wall cap ${WALLMIN} min each; hard stop $(date -d "@$DL" '+%a %H:%M') local, $LEFT min away)"
if [ "$SMOKE" != 1 ] && [ -z "$STUBO" ]; then
  [ "$LEFT" -ge 60 ] || { nl_say "  under 60 min left in the window: not starting (exit 3, resumable)"; exit 3; }
fi
if [ "${DRYRUN:-0}" = "1" ] || [ ! -f "$MANIFEST" ]; then nl_manifest_init serve_f16 takes_f16 serve_fp8 takes_fp8 gate render; fi
nl_gate

# ---- phase 2 ----
if [ -z "$STUBO" ] && [ "${WF_HOLDS_LOCK:-0}" != 1 ]; then
  exec 9>/home/witcheer/.gpu-smoke.lock
  flock -w 1800 9 || { nl_say "gpu lock still held after 30 min: not starting, exit 3"; exit 3; }
  nl_say "gpu lock taken"
fi
SP=""; DONALD_STOPPED=0; DONALD_BACK=0
mem_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
stop_srv() { [ -n "$SP" ] || return 0; kill -TERM "$SP" 2>/dev/null; for i in $(seq 1 30); do kill -0 "$SP" 2>/dev/null || break; sleep 2; done; kill -KILL "$SP" 2>/dev/null; sleep 3; SP=""; }
kill_own_srv() { local pid; for pid in $(pgrep -x llama-server 2>/dev/null); do [ "$(readlink -f /proc/$pid/exe 2>/dev/null)" = "$(readlink -f "$SRV")" ] && kill -9 "$pid" 2>/dev/null; done; return 0; }
wait_mem_free() { local i v; for i in $(seq 1 90); do v=$(mem_used); [ "${v:-99999}" -lt 1024 ] && { nl_say "  card clear (${v} MiB)"; return 0; }; sleep 2; done; nl_say "  card NOT clear after 180 s (${v} MiB)"; return 1; }
donald_stop() { nl_say "draining Donald (night-lib nl_donald_stop)"; nl_donald_stop; DONALD_STOPPED=1; }
donald_restore() {
  [ "$DONALD_STOPPED" = 1 ] && [ "$DONALD_BACK" = 0 ] || return 0
  DONALD_BACK=1; nl_say "restoring Donald (night-lib nl_donald_restore)"; nl_donald_restore
  local i h; for i in $(seq 1 60); do h=$(curl -s -m 5 localhost:8090/health || true); [[ ${h,,} == *ok* ]] && { nl_say "Donald healthy again after ~$((i*5)) s"; return 0; }; sleep 5; done
  nl_say "WARNING: Donald /health did not answer within 5 min after the restore"
}
cleanup() { for t in kv-f16 kv-fp8; do tmux kill-session -t "=$t" 2>/dev/null; done; stop_srv
  if [ -z "$STUBO" ] && [ "$DONALD_STOPPED" = 1 ]; then kill_own_srv; sleep 2; wait_mem_free || true; fi
  donald_restore; }
trap cleanup EXIT
mkdir -p "$RUN/out" "$RUN/takes"
t0=$(date +%s)
fail() { nl_mark "$1" failed "$2"; nl_say "FAIL: $2"; exit 1; }
DEADLINE=$(nl_deadline_epoch)
SHORT=0
BASEARGS=(-ngl 99 -t $THREADS -fa on --jinja -c $CTXCFG --parallel 1)
take_done() { [ -f "$RUN/takes/$1/rec/result.json" ] && grep -qE '"end": *"(done|wall)"' "$RUN/takes/$1/rec/result.json" 2>/dev/null; }
leg_pending() { local tag=$1 lim=$2 i; for i in $(seq 1 "$lim"); do take_done "$tag-$i" || return 0; done; return 1; }
wait_up() { local i h t1=$(date +%s); for i in $(seq 1 "$2"); do h=$(curl -s -m 3 "http://127.0.0.1:$1/health" || true); [[ ${h,,} == *ok* ]] && { LSTART=$(( $(date +%s)-t1 )); return 0; }; kill -0 "$SP" 2>/dev/null || return 1; sleep 2; done; return 1; }
serve_arm() { # <arm> <model>: sets SP
  local arm=$1 model=$2 KVA=()
  [ "$arm" = fp8 ] && KVA=(-ctk f8_e4m3 -ctv f8_e4m3)
  "$SRV" -m "$model" "${BASEARGS[@]}" "${KVA[@]}" --host 127.0.0.1 --port $LPORT --alias qwen38 --metrics --reasoning-budget $RB --reasoning-budget-message "$RB_MSG" >> "$RUN/server-$arm.log" 2>&1 &
  SP=$!
  wait_up $LPORT 300 && { nl_say "  llama-server ($arm) up after ${LSTART} s (VRAM $(mem_used) MiB)"; return 0; }
  return 1
}
chat_smoke() {
  python3 - "$1" <<'PYEOF'
import json, sys, time, urllib.request
port = sys.argv[1]
body = {"model": "qwen38", "messages": [{"role": "user", "content": "What time is it in Paris? Use the tool."}], "max_tokens": 3000, "temperature": 0.6, "stream": False,
        "tools": [{"type": "function", "function": {"name": "get_time", "description": "current time in a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]}
t = time.time()
try:
    r = json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions" % port, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=300).read())
except Exception as e:
    print("SMOKE FAIL request: %s" % str(e)[:160]); sys.exit(1)
m = r["choices"][0]["message"]
tc = m.get("tool_calls") or []
ok = bool(tc) and tc[0]["function"]["name"] == "get_time" and "city" in json.loads(tc[0]["function"]["arguments"] or "{}")
tm = r.get("timings") or {}
print(("SMOKE OK" if ok else "SMOKE FAIL") + " tool_calls=%d completion_tokens=%s decode=%.1f %.1fs" % (len(tc), (r.get("usage") or {}).get("completion_tokens"), tm.get("predicted_per_second") or 0, time.time() - t))
sys.exit(0 if ok else 1)
PYEOF
}
probe() {
  python3 - "$2" "$RUN/probe-$1.json" <<'PYEOF'
import json, sys, urllib.request
port, out = sys.argv[1], sys.argv[2]
P = ("The lighthouse keeper had kept a logbook for thirty-one years, and in all that time he had never once written down a lie. So when the ship appeared on the horizon on the first calm morning of October, "
     "with no flag, no lights and no one on deck, he sat at his small wooden table for a long while before he picked up the pen. ")
res = []
for i in range(2):
    try:
        r = json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:%s/completion" % port, json.dumps({"prompt": P, "n_predict": 256, "temperature": 0.0, "ignore_eos": True, "cache_prompt": False}).encode(), {"Content-Type": "application/json"}), timeout=600).read())
        tm = r["timings"]
        res.append({"decode_tok_s": round(tm["predicted_per_second"], 2), "predicted_n": tm["predicted_n"]})
    except Exception as e:
        res.append({"error": str(e)[:120]})
ok = [x for x in res if "decode_tok_s" in x]
json.dump({"runs": res, "decode_tok_s": ok[-1]["decode_tok_s"] if ok else None}, open(out, "w"))
print("probe", json.dumps(res))
PYEOF
}
run_takes() { # <tag> <upstream> <from> <to>
  local tag=$1 up=$2 from=$3 to=$4 n td left w rc
  for n in $(seq "$from" "$to"); do
    td=$RUN/takes/$tag-$n
    if take_done "$tag-$n"; then nl_say "  take $tag-$n: already done (resume), kept"; continue; fi
    left=$(( DEADLINE - $(date +%s) ))
    [ -n "$STUBO" ] && left=99999
    if [ "$left" -lt 660 ]; then nl_say "  take $tag-$n: only $((left/60)) min left (< 11), skipped"; SHORT=1; continue; fi
    w=$(( WALLMIN * 60 )); [ "$((left - 360))" -lt "$w" ] && w=$(( left - 360 ))
    [ -d "$td" ] && mv "$td" "$td.partial-$(date +%s)"
    nl_say "=== take $tag-$n (wall cap $w s, $((left/60)) min left) ==="
    UPSTREAM=$up ALIAS=qwen38 TAG=$tag-$n PROMPT=$PROMPT SEEDDIR=$SEEDDIR WALL_S=$w CTXCFG=$CTXCFG FR_PORT=$PPORT FR_PY=$RPY FR_GPY=$GPY FR_CHECKOUT=$CHECKOUT FR_QUIET=$QUIET THINK=$THINK ARENA=/tmp/arena99z bash "$H/mc_take.sh" "$td" >> "$RUN/take.log" 2>&1; rc=$?
    nl_say "  take $tag-$n: rc=$rc $(cat "$td/rec/result.json" 2>/dev/null)"
    [ "$rc" = 0 ] || { nl_say "  take $tag-$n: harness rc $rc (see $RUN/take.log)"; SHORT=1; }
    mkdir -p "$td/gate"
    nl_say "  gate $tag-$n: $("$GPY" "$H/rp_gate.py" "$td/work" "$SEEDDIR/tests" "$td/gate" --pytest-python "$GPY" --seed-pkg "$SEEDDIR/shopcart" 2>&1 | head -1 | cut -c1-220)"
  done
}
leg() { # <arm> <model> <from> <to>
  local arm=$1 model=$2 from=$3 to=$4 UP
  if ! leg_pending "$arm" "$to"; then nl_say "leg $arm: takes $from-$to already done"; nl_mark "serve_$arm" done "resume: skipped"; return 0; fi
  nl_mark "serve_$arm" running
  if [ -n "$STUBO" ]; then
    UP=$STUBO; [ "$arm" = fp8 ] && UP=$STUBN
    echo "{\"loaded\":true,\"stub\":true,\"arm\":\"$arm\",\"vram_mib\":0,\"bytes\":$(stat -c %s "$model"),\"file\":\"$(basename "$model")\"}" > "$RUN/fit-$arm.json"; echo '{"runs":[],"decode_tok_s":null,"stub":true}' > "$RUN/probe-$arm.json"
  else
    LSTART=0
    serve_arm "$arm" "$model" || { stop_srv; echo "{\"loaded\":false,\"why\":\"did not come up\"}" > "$RUN/fit-$arm.json"; nl_mark "serve_$arm" failed "no load"; nl_say "  llama-server did not come up for $arm: arm BROKEN, going on"; return 0; }
    UP=http://127.0.0.1:$LPORT
  fi
  SM=$(chat_smoke "${UP##*:}"); nl_say "  tool-call smoke ($arm): $SM"
  case "$SM" in
    "SMOKE OK"*) [ -z "$STUBO" ] && echo "{\"loaded\":true,\"start_s\":${LSTART:-0},\"vram_mib\":$(mem_used),\"bytes\":$(stat -c %s "$model"),\"file\":\"$(basename "$model")\",\"ctx\":$CTXCFG,\"kv\":\"$( [ "$arm" = fp8 ] && echo f8_e4m3 || echo f16 )\"}" > "$RUN/fit-$arm.json";;
    *) echo "{\"loaded\":false,\"why\":\"tool-call smoke failed\"}" > "$RUN/fit-$arm.json"; stop_srv; nl_mark "serve_$arm" failed "smoke"; nl_say "  arm $arm BROKEN: tool-call smoke failed, going on"; return 0;;
  esac
  [ -z "$STUBO" ] && nl_say "  probe ($arm): $(probe "$arm" "${UP##*:}" 2>&1 | tail -1 | cut -c1-200)"
  nl_mark "serve_$arm" done "up"
  nl_mark "takes_$arm" running
  run_takes "$arm" "$UP" "$from" "$to"
  [ -z "$STUBO" ] && { stop_srv; wait_mem_free || fail "takes_$arm" "card not clear after the $arm stop"; }
  nl_mark "takes_$arm" done "takes $from-$to"
}

if [ -z "$STUBO" ]; then
  FOREIGN=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ' | tr '\n' ' ')
  nl_say "  card owners before the drain: ${FOREIGN:-none}; tmux: $(tmux ls -F '#{session_name}' 2>/dev/null | tr '\n' ' ')"
  donald_stop
  for i in $(seq 1 60); do v=$(mem_used); [ "${v:-99999}" -lt 1024 ] && { nl_say "drain clear (${v} MiB)"; break; }; sleep 2; done
  [ "${v:-99999}" -lt 1024 ] || fail serve_f16 "card memory still ${v:-?} MiB after the drain: another process owns the card, not loading"
else nl_say "STUB mode: no Donald drain, no server start (upstreams $STUBO / $STUBN)"; fi


# ---- max-context sweep (brief: the max context that fits, both arms) ----
# each arm: try ctx in 32768 65536 98304 131072; keep the largest that loads and answers /health (no Hermes takes here). fit json gets max_ctx.
sweep_arm() { # <arm> <model>
  local arm=$1 model=$2 c ok=0 best=0 KVA=()
  [ "$arm" = fp8 ] && KVA=(-ctk f8_e4m3 -ctv f8_e4m3)
  for c in 32768 65536 98304 131072; do
    if [ -n "$STUBO" ]; then best=$c; continue; fi
    "$SRV" -m "$model" -ngl 99 -t $THREADS -fa on --jinja -c $c --parallel 1 "${KVA[@]}" --host 127.0.0.1 --port $LPORT >> "$RUN/sweep-$arm.log" 2>&1 &
    SP=$!
    if wait_up $LPORT 300; then best=$c; else stop_srv; break; fi
    stop_srv; wait_mem_free || break
  done
  echo "{\"max_ctx\":$best,\"arm\":\"$arm\"}" > "$RUN/sweep-$arm.json"
  nl_say "  max context ($arm): $best"
}

# headline leg: 3 takes per arm, old then new, interleaved would double the loads: the arms run as blocks (one load each)
leg f16 "$MODEL" 1 "$NT"
sweep_arm f16 "$MODEL"
  sweep_arm fp8 "$MODEL"
leg fp8 "$MODEL" 1 "$NT"
# extension legs (resumable): more takes per arm, same blocks
if [ "$EXT" -gt "$NT" ]; then
  leg f16 "$MODEL" 1 "$EXT"
  leg fp8 "$MODEL" 1 "$EXT"
fi
if [ -z "$STUBO" ]; then donald_restore; fi
nl_say "card released after $(( ($(date +%s)-t0)/60 )) min, the rest is CPU"

for td in "$RUN"/takes/*; do
  case "$td" in *.partial-*) continue;; esac
  [ -s "$td/hud/hud.jsonl" ] || continue
  python3 "$H/rp_recount.py" "$td" --json "$RUN/out/recount-$(basename "$td").json" > /dev/null 2>&1
done
"$RPY" - "$RUN" "$NT" <<'PYEOF' > "$RUN/pair.json"
import json, os, sys
run, takes = sys.argv[1], int(sys.argv[2])
def ok(d):
    try:
        r = json.load(open(d + "/rec/result.json"))
        return r.get("end") in ("done", "wall") and os.path.getsize(d + "/rec/events-a.jsonl") > 1000 and os.path.getsize(d + "/hud/hud.jsonl") > 100
    except Exception:
        return False
pick, why = {}, {}
for e in ("f16", "fp8"):
    for n in range(1, takes + 1):
        d = "%s/takes/%s-%d" % (run, e, n)
        if ok(d):
            pick[e] = d; why[e] = "take %d" % n; break
    else:
        pick[e], why[e] = None, "no usable take"
print(json.dumps({"left": pick["f16"], "right": pick["fp8"], "why": why}))
PYEOF
LT=$($RPY -c "import json;print(json.load(open('$RUN/pair.json'))['left'] or '')"); RT=$($RPY -c "import json;print(json.load(open('$RUN/pair.json'))['right'] or '')")
nl_say "  clip pair: $(cat "$RUN/pair.json")"
nl_mark gate done "gates ran per take"
"$GPY" "$H/wf_summary.py" "$RUN" "$SLUG" "$REGIME" > "$RUN/summary.pre.json" 2>>"$RUN/render.log"
VERD=$("$GPY" -c "import json;d=json.load(open('$RUN/summary.pre.json'));print(d['verdict'])" 2>/dev/null)
nl_say "  verdict (pre-registered rule): ${VERD:-unreadable}"
nl_mark render running
rr=1
if [ -n "$LT" ] && [ -n "$RT" ]; then
  "$RPY" "$H/mc_render.py" race "$RUN/out/race.mp4" --left "$LT" --right "$RT" --left-label "f16 KV" --right-label "fp8 KV" --title "Qwen3.8-27B Q6_K, f16 vs fp8 KV, one 5090" --ffmpeg "$FF" --stills "$RUN/out/stills" --cut-after-first 8 --real-max-s 40 --target-main-s 30 --open-s 10 --close-s 12 --pre-s 3 >> "$RUN/render.log" 2>&1; rr=$?
  if [ "$rr" = 0 ]; then
    "$GPY" "$H/wf_summary.py" "$RUN" "$SLUG" "$REGIME" > "$RUN/summary.card.json" 2>>"$RUN/render.log"
    "$RPY" "$H/wf_facts.py" "$RUN/summary.card.json" "$RUN/facts.json" >> "$RUN/render.log" 2>&1 && \
    "$RPY" "$H/mc_render.py" card "$RUN/out/card.mp4" --facts "$RUN/facts.json" --ffmpeg "$FF" --seconds 8 >> "$RUN/render.log" 2>&1 && \
    "$FF" -nostdin -hide_banner -loglevel error -y -i "$RUN/out/card.mp4" -i "$RUN/out/race.mp4" \
      -filter_complex "[0:v]setsar=1,fps=25[a];[1:v]setsar=1,fps=25[b];[a][b]concat=n=2:v=1:a=0[v]" -map "[v]" -c:v libx264 -crf 20 -pix_fmt yuv420p -movflags +faststart "$RUN/out/whichfile-final.mp4" </dev/null >> "$RUN/render.log" 2>&1
    rr=$?
  fi
else nl_say "  no usable take on one of the arms: no race clip (the table and verdict still stand)"; fi
"$RPY" - "$RUN/out" "$RUN/out/contact.png" <<'PYEOF' 2>>"$RUN/render.log"
import glob, os, sys
from PIL import Image
out = sys.argv[1]
fs = sorted(glob.glob(out + "/stills/*.png"))[:8]
if fs:
    w, h = 640, 360
    sheet = Image.new("RGB", (w * 3, h * 3), "#000")
    for i, f in enumerate(fs[:9]):
        sheet.paste(Image.open(f).convert("RGB").resize((w, h)), ((i % 3) * w, (i // 3) * h))
    sheet.save(sys.argv[2])
PYEOF
nl_say "  render rc=$rr; final clip: $([ -f "$RUN/out/whichfile-final.mp4" ] && du -h "$RUN/out/whichfile-final.mp4" | cut -f1 || echo missing)"
if [ "$SHORT" = 1 ] || [ "$VERD" = INCOMPLETE ]; then
  nl_say "  a leg could not finish inside the window (verdict ${VERD:-?}): exit 3, resumable, summary.json NOT written (PROVISIONAL: $RUN/summary.pre.json)"
  nl_mark render failed "incomplete, resumable"
  exit 3
fi
"$GPY" "$H/wf_summary.py" "$RUN" "$SLUG" "$REGIME" > "$ART" 2>>"$RUN/render.log"
nl_judge_artifact render "$rr" "$ART" "$t0" "$(( ($(date +%s)-t0)/60 )) min, render rc $rr" || { nl_say "FAIL: no fresh summary.json or render rc $rr"; exit 1; }
nl_say "  summary: verdict $VERD; line 1: $("$GPY" -c "import json;d=json.load(open('$ART'));print(d.get('line1'))")"
nl_say "DONE $SLUG"
exit 0
