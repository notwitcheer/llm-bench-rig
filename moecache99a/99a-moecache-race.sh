#!/bin/bash
# 99a-moecache-race.sh: idle-queue item (2026-11-08). Release-week item: llama.cpp's new MoE expert cache
#   (PR #30966 merged 2026-11-07 16:20 UTC, #31233 ClassLRU 11-06, #31853 CRF 11-07, #31497 11-08; master a2b301831 = tag b12201) against the best static offload we measured, on the SAME model file.
# WHAT: Ornith-1.5-122B UD-IQ3_XXS (49,515,667,712 B, already on this box, 5.06 GB non-expert + ~41 GiB experts) does not fit 32 GB. Two arms, one build, one file, real Hermes Agent sessions behind the counting proxy:
#   plain = `--n-cpu-moe 24` (the optimum of our 2026-10-25 sweep: 47.63 tok/s, build b10371; if it does not load at ctx 65,536 with q8_0 KV, 26 then 28, the first that loads is used and reported)
#   cache = `-cmoe --moe-cache-mib N --moe-cache-cpu-share F`, N and F = the recommendation of the PR's own tuner scripts/moe-cache-tune.py (sha256 frozen), run on our card inside the item; fixed fallback 16384 MiB / share 0, labelled.
#   task = item 89's fix-the-shopcart-repo prompt (4 planted bugs, 5 tests, pristine tests re-run by the pre-registered gate); 3 takes per arm. The verdict is DECODE SPEED (server timings), the gate pass counts are reported only.
# MOJO: two Hermes Agent panes on the same 122B file on one 32 GB gaming card, tok/s counters and a clock, no cache against llama.cpp's new MoE cache merged two days ago, cut at the first finisher | ahead / intense (a merge from this weekend replayed on our own 47.6 tok/s number) | on one RTX 5090 (32 GB) a 122B model, run by Hermes Agent: llama.cpp's new MoE cache took decode from [P] to [C] tok/s ([R]x) with the same model file
#
# PRE-REGISTERED (moecache99a/PREREG.md, sha256 in moecache99a/FROZEN.sha256, BEFORE any counted output; this item refuses to run when any frozen hash differs; mc_summary.py applies the rule, `mc_summary.py --selftest` proves 12 branches):
#   take CLEAN = ended done or wall, >= 3 tool calls, request errors <= 20%.  take PASS = CLEAN and pristine tests 5 passed 0 failed 0 errors and nothing under tests/ touched and the agent never named the harness dir (reported, never gating).
#   arm decode = token-weighted llama-server predicted_per_second over the arm's CLEAN takes.  r = cache / plain.  FASTER r >= 1.10 | SLOWER r <= 0.90 | TIE between | CACHE FAILED (plain fine, cache arm broken) | BROKEN (plain broken) | INCOMPLETE (< 2 CLEAN takes in an arm: exit 3, resumes).
#   a TIE or a SLOWER is a valid result, framed as one rig's measurement. no speed multiplier is claimed beyond the measured ratio on this file and card.
# POSTABLE ARTEFACT (planned here): runs/<slug>/out/moecache-final.mp4 (16:9 1920x1080, ~65 s: an 8 s result card FIRST with counting numbers, a grey how-it-was-measured line under each row and a footer with the claim tested and the limits,
#   then the two live Hermes panes (no cache | MoE cache), big tok/s counters and clocks, cut 8 s after the first pane is done) + out/contact.png + summary.json. HUD LEAK READ: status bar blanked, account / home path / tailnet / hostname / e-mail shapes drawn as blocks.
# CARD TIME (ESTIMATE, basis: 10-25 sweep numbers + item 89 take times; smoke numbers go into the status note): drain + plain load ~3 min, 3 plain takes ~12-20 min, tuner ~10-18 min (6 server starts), cache load + 3 takes ~15-23 min => ~45-70 min typical,
#   worst case ~150 (every take at the 10 min cap + slow loads). CPU after the card is released: recounts + gates + render ~5-8 min. Disk: no download, ~3 GB build already done, runs < 1 GB.
# WINDOW RULE: needs >= 110 min left in the block or exits 3 without draining Donald (stays queued, resumable); a take is skipped when under 11 min remain; the tuner is skipped (fallback params, labelled) when under 70 min remain.
# GPU lock: the item takes `flock -w 1800 /home/witcheer/.gpu-smoke.lock` for the whole card section (a sibling staging smoke may hold it); every Hermes-agent harness starts llama-server with --reasoning-budget + the series budget message.
#
# test knobs (empty in production): MC_SMOKE=1 (slug -smoke, 1 take per arm, wall 3 min, tuner skipped = fallback params, no window rule), MC_HOLDS_LOCK=1 (the caller already holds the gpu lock), MC_TAKES, MC_WALL_MIN, MC_THINK=on|off,
#   MC_STUB_UPSTREAM / MC_STUB_UPSTREAM_CACHE=http://127.0.0.1:PORT (CPU mock per arm: no Donald drain, no server start, DRYRUN-safe), MC_MODEL (red-path test), MANIFEST_OVERRIDE (staging DRYRUN only), MC_NO_SELFTEST=1.
set -uo pipefail
NL_TAG=moecache99a
MANIFEST=/home/witcheer/moecache99a-status.json
[ -n "${MANIFEST_OVERRIDE:-}" ] && MANIFEST=$MANIFEST_OVERRIDE
export MANIFEST
. /home/witcheer/night-lib.sh
H=/home/witcheer/voxel-stage/moecache99a
RUNROOT=/home/witcheer/moecache99a-runs
GPY=/home/witcheer/benchmark-rig/.venv/bin/python          # pytest (the gate, the agent's test runner wrapper)
RPY=/home/witcheer/scratch/race/venv/bin/python             # pyte + PIL + fontTools (recorder + renderer)
FF=/home/witcheer/voxelbench/ffmpeg
CHECKOUT=/home/witcheer/.hermes/hermes-agent
HBIN=$CHECKOUT/.hermes/bin/hermes
HERMES_PIN=9bda188
LL=/home/witcheer/llama.cpp-moecache
SRV=$LL/build/bin/llama-server
BUILD_SHA=a2b301831
TUNER=$LL/scripts/moe-cache-tune.py
MODEL=${MC_MODEL:-/home/witcheer/models/ornith-15-122b/Ornith-1.5-122B-UD-IQ3_XXS.gguf}
MODEL_BYTES=49515667712
LPORT=8277; PPORT=8293                                      # own ports: never :8090 (Donald), 8261-8263 / 8291 (earlier staging), 8196-8198 (items 78/80/86)
CTXCFG=65536; KV=q8_0; THREADS=6
NCMOE_LADDER="24 26 28"
FALLBACK_MIB=16384; FALLBACK_SHARE=0
RB=8192; RB_MSG="Thinking budget reached. Act now with your commands."
PROMPT=$H/prompts/fix-shopcart.txt
SEEDDIR=$H/seed
H_FROZEN=58de78d651f3fa783f65977a6c258c363e822d01bb75aed9721c81131f547a13                                       # sha256 of FROZEN.sha256 (the list of frozen files), set at freeze time
SMOKE=${MC_SMOKE:-0}
NT=${MC_TAKES:-3}; WALLMIN=${MC_WALL_MIN:-10}; THINK=${MC_THINK:-on}; QUIET=${MC_QUIET:-20}
[ "$SMOKE" = 1 ] && { NT=${MC_TAKES:-1}; WALLMIN=${MC_WALL_MIN:-3}; }
SLUG=moecache99a-moecache-race; [ "$SMOKE" = 1 ] && SLUG=$SLUG-smoke
STUBP=${MC_STUB_UPSTREAM:-}; STUBC=${MC_STUB_UPSTREAM_CACHE:-$STUBP}
RUN=$RUNROOT/$SLUG
ART=$RUN/summary.json
REGIME="Ornith-1.5-122B UD-IQ3_XXS (one file), llama.cpp master a2b301831 (b12201), --jinja, -ngl 99, flash-attn on, KV $KV, ctx $CTXCFG, one slot, $THREADS threads, thinking $THINK with --reasoning-budget $RB; plain = --n-cpu-moe 24 (ladder 24/26/28 on a load failure), cache = -cmoe + --moe-cache-mib/--moe-cache-cpu-share from the PR's tuner (fallback $FALLBACK_MIB / $FALLBACK_SHARE); Hermes terminal+file toolsets, approvals off, memory off; sampling temp 0.6 / top-p 0.95 / top-k 20 (Ornith card) + seed 721 (ours); same task as item 89 (4 planted bugs, 5 tests), $NT takes per arm; verdict = token-weighted decode ratio, FASTER >= 1.10, SLOWER <= 0.90"

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0} smoke=$SMOKE takes=$NT+$NT wall=${WALLMIN}min stub=${STUBP:-none})"
nl_check_path "$GPY" "$RPY" "$HBIN" "$FF" "$PROMPT" "$SEEDDIR/shopcart/cart.py" "$SEEDDIR/tests/test_cart.py" "$SRV" "$TUNER" \
  "$H/mc_take.sh" "$H/rp_proxy.py" "$H/rp_count.py" "$H/rp_recount.py" "$H/rp_record.py" "$H/mc_mkhome.py" "$H/mc_render.py" "$H/mc_summary.py" "$H/mc_facts.py" "$H/rp_gate.py" "$H/PREREG.md" "$H/FROZEN.sha256" "$H/SEED.sha256" \
  /home/witcheer/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome
if [ -z "${MC_MODEL:-}" ]; then
  [ "$(stat -c %s "$MODEL" 2>/dev/null)" = "$MODEL_BYTES" ] && nl_ok "model size $MODEL_BYTES" || nl_fail "model size != $MODEL_BYTES"
else nl_say "  model override (test): $MODEL"; fi
# the pre-registration is frozen: FROZEN.sha256 lists every frozen file, its own hash is pinned in this script, PREREG.md must list the same hashes
FZ=$(sha256sum "$H/FROZEN.sha256" | cut -c1-64)
[ "$FZ" = "$H_FROZEN" ] && nl_ok "FROZEN.sha256 $(echo "$FZ" | cut -c1-12)" || nl_fail "FROZEN.sha256 differs from the pin ($(echo "$FZ" | cut -c1-12) != $(echo "$H_FROZEN" | cut -c1-12))"
CK=$(cd "$H" && sha256sum -c FROZEN.sha256 2>&1) && nl_ok "frozen files match ($(echo "$CK" | grep -c ': OK') OK)" || nl_fail "frozen file differs: $(echo "$CK" | grep -v ': OK' | head -3 | tr '\n' ' ')"
SEEDH=$(cd "$SEEDDIR" && find . -type f ! -name '*.pyc' ! -path '*/__pycache__/*' | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-64)
H_SEED=$(cut -d' ' -f1 "$H/SEED.sha256" 2>/dev/null)
[ -n "$H_SEED" ] && [ "$SEEDH" = "$H_SEED" ] && nl_ok "frozen seed repo $(echo "$SEEDH" | cut -c1-12)" || nl_fail "seed repo differs from the pre-registration ($(echo "$SEEDH" | cut -c1-12))"
for h in $(grep -v 'PREREG.md' "$H/FROZEN.sha256" | cut -d' ' -f1) "$H_SEED"; do
  PRG=$(cat "$H/PREREG.md"); [[ $PRG == *"$h"* ]] || nl_fail "PREREG.md does not list frozen hash $(echo "$h" | cut -c1-12)"
done
"$GPY" -c "import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$H/rp_gate.py" "$H/rp_proxy.py" "$H/rp_count.py" "$H/rp_recount.py" "$H/rp_record.py" "$H/mc_mkhome.py" "$H/mc_summary.py" "$H/mc_facts.py" || nl_fail "harness syntax"
(cd / && "$RPY" -c "import ast,sys,pyte,PIL,fontTools; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$H/mc_render.py") || nl_fail "renderer deps/syntax (pyte, PIL, fontTools in $RPY)"
bash -n "$H/mc_take.sh" || nl_fail "mc_take.sh syntax"
if [ "${MC_NO_SELFTEST:-0}" != 1 ]; then
  ST=$("$GPY" "$H/mc_summary.py" --selftest 2>&1); case "$ST" in *BAD*) nl_fail "verdict self-test: $ST";; *) nl_ok "verdict rule self-test: $(echo "$ST" | grep -c 'OK') branches OK";; esac
  GT=$(mktemp -d /tmp/mc99agate.XXXXXX); cp -a "$SEEDDIR/." "$GT/work"; mkdir -p "$GT/o"
  G0=$("$GPY" "$H/rp_gate.py" "$GT/work" "$SEEDDIR/tests" "$GT/o" --pytest-python "$GPY" --seed-pkg "$SEEDDIR/shopcart" 2>&1 | head -1)
  case "$G0" in "GATE FAIL: 1 passed, 4 failed"*) nl_ok "gate: the untouched seed fails ($G0)";; *) nl_fail "gate on the seed: $G0";; esac
  rm -rf "$GT"
fi
HELPL=$(CUDA_VISIBLE_DEVICES= $SRV --help 2>&1 || true)
for f in --jinja --parallel --ctx-size --alias --n-gpu-layers --flash-attn --cache-type-k --cpu-moe --n-cpu-moe --moe-cache-mib --moe-cache-cpu-share --reasoning-budget --reasoning-budget-message --metrics --threads; do
  [[ $HELPL == *"$f"* ]] || nl_fail "llama-server flag $f not in this build"
done
VER=$(CUDA_VISIBLE_DEVICES= $SRV --version 2>&1 || true)
[[ $VER == *"$BUILD_SHA"* ]] && nl_ok "llama.cpp build $BUILD_SHA" || nl_fail "llama.cpp moecache build changed (expected $BUILD_SHA)"
LLH=$(cd "$LL" && git rev-parse --short=9 HEAD 2>/dev/null); [ "$LLH" = "$BUILD_SHA" ] && nl_ok "source tree at $LLH" || nl_fail "source tree HEAD $LLH != $BUILD_SHA"
HV=$(cd "$CHECKOUT" && git rev-parse --short HEAD 2>/dev/null)
case "$HV" in "$HERMES_PIN"*) nl_ok "Hermes checkout $HV (pin $HERMES_PIN)";; *) nl_fail "Hermes checkout $HV != pin $HERMES_PIN";; esac
for dd in installs tools; do [ -d "/home/witcheer/.hermes/$dd" ] || nl_fail "main Hermes root has no $dd/ to share with the take homes"; done
nl_say "  regime: $REGIME"
nl_check_disk 125
for p in $LPORT $PPORT; do LS=$(ss -ltnH "sport = :$p" 2>/dev/null || true); [ -z "$LS" ] || nl_fail "port $p is already in use"; done
for t in mc-plain mc-cache; do tmux has-session -t "=$t" 2>/dev/null && nl_fail "tmux session $t already exists"; done
PT=$(cat "$PROMPT"); [[ $PT == */home/* || $PT == */Users/* || $PT == *witcheer* || $PT == *moecache99a* ]] && nl_fail "prompt file names a home path, the account or the harness dir"
if nl_skip_exact "$ART"; then nl_say "nothing to run, item complete ($ART exists)"; exit 0; fi
DL=$(nl_deadline_epoch); LEFT=$(( (DL-$(date +%s))/60 ))
for arm in plain cache; do for n in $(seq 1 "$NT"); do nl_say "  RUN take $arm-$n -> $RUN/takes/$arm-$n (literal artefact $RUN/takes/$arm-$n/rec/result.json)"; done; done
nl_say "  RUN tuner -> $RUN/tune.json ; RUN clip -> $RUN/out/moecache-final.mp4 ; RUN artefact -> $ART"
nl_say "  planned: $((NT*2)) takes (wall cap ${WALLMIN} min each; card typical ~45-70 min, worst ~150; hard stop $(date -d "@$DL" '+%a %H:%M') local, $LEFT min away)"
if [ "$SMOKE" != 1 ] && [ -z "$STUBP" ]; then
  [ "$LEFT" -ge 110 ] || { nl_say "  under 110 min left in the window: not starting (drain + 2 arms + tuner would be cut short; exit 3, resumable)"; exit 3; }
fi
if [ "${DRYRUN:-0}" = "1" ] || [ ! -f "$MANIFEST" ]; then nl_manifest_init serve_plain takes_plain tune serve_cache takes_cache gate render; fi
nl_gate

# ---- phase 2 ----
if [ -z "$STUBP" ] && [ "${MC_HOLDS_LOCK:-0}" != 1 ]; then
  exec 9>/home/witcheer/.gpu-smoke.lock
  flock -w 1800 9 || { nl_say "gpu lock still held after 30 min (a sibling staging smoke?): not starting, exit 3"; exit 3; }
  nl_say "gpu lock taken"
fi
SP=""; DONALD_STOPPED=0; DONALD_BACK=0
mem_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
wait_mem_free() { local i v; for i in $(seq 1 90); do v=$(mem_used); [ "${v:-99999}" -lt 1024 ] && { nl_say "  card clear (${v} MiB)"; return 0; }; sleep 2; done; nl_say "  card NOT clear after 180 s (${v} MiB)"; return 1; }
stop_srv() { # own server only: the pid this item started
  [ -n "$SP" ] || return 0
  kill -TERM "$SP" 2>/dev/null
  for i in $(seq 1 30); do kill -0 "$SP" 2>/dev/null || break; sleep 2; done
  kill -KILL "$SP" 2>/dev/null; sleep 3; SP=""; }
kill_own_srv() { # kill only llama-server processes that run OUR binary (never Donald's, never a sibling's): pgrep -x + /proc exe match
  local pid
  for pid in $(pgrep -x llama-server 2>/dev/null); do [ "$(readlink -f /proc/$pid/exe 2>/dev/null)" = "$(readlink -f "$SRV")" ] && kill -9 "$pid" 2>/dev/null; done; return 0; }
donald_stop() { nl_say "draining Donald (night-lib nl_donald_stop)"; nl_donald_stop; DONALD_STOPPED=1; }
donald_restore() {
  [ "$DONALD_STOPPED" = 1 ] && [ "$DONALD_BACK" = 0 ] || return 0
  DONALD_BACK=1; nl_say "restoring Donald (night-lib nl_donald_restore)"
  nl_donald_restore
  local i h
  for i in $(seq 1 60); do h=$(curl -s -m 5 localhost:8090/health || true); [[ ${h,,} == *ok* ]] && { nl_say "Donald healthy again after ~$((i*5)) s"; return 0; }; sleep 5; done
  nl_say "WARNING: Donald /health did not answer within 5 min after the restore"
}
cleanup() { for t in mc-plain mc-cache; do tmux kill-session -t "=$t" 2>/dev/null; done; stop_srv
  if [ -z "$STUBP" ] && [ "$DONALD_STOPPED" = 1 ]; then kill_own_srv; sleep 2; wait_mem_free || true; fi   # only after our own drain: the card is ours, a tuner child may still hold it
  donald_restore; }
trap cleanup EXIT
mkdir -p "$RUN/out" "$RUN/takes"      # resumable: an existing RUN keeps its finished takes
t0=$(date +%s)
fail() { nl_mark "$1" failed "$2"; nl_say "FAIL: $2"; exit 1; }
DEADLINE=$(nl_deadline_epoch)
SHORT=0
BASEARGS=(-m "$MODEL" -ngl 99 -t $THREADS -fa on --jinja -c $CTXCFG --parallel 1 -ctk $KV -ctv $KV)

take_done() { [ -f "$RUN/takes/$1/rec/result.json" ] && grep -qE '"end": *"(done|wall)"' "$RUN/takes/$1/rec/result.json" 2>/dev/null; }
leg_pending() { local tag=$1 i; for i in $(seq 1 "$NT"); do take_done "$tag-$i" || return 0; done; return 1; }
wait_up() { # <port> <secs>
  local i h t1=$(date +%s)
  for i in $(seq 1 "$2"); do
    h=$(curl -s -m 3 "http://127.0.0.1:$1/health" || true); [[ ${h,,} == *ok* ]] && { LSTART=$(( $(date +%s)-t1 )); return 0; }
    kill -0 "$SP" 2>/dev/null || return 1; sleep 2; done
  return 1; }
serve_arm() { # <arm> <extra args...>: sets SP
  local arm=$1; shift
  "$SRV" "${BASEARGS[@]}" --host 127.0.0.1 --port $LPORT --alias ornith --metrics --reasoning-budget $RB --reasoning-budget-message "$RB_MSG" "$@" >> "$RUN/server-$arm.log" 2>&1 &
  SP=$!
  wait_up $LPORT 240 && { nl_say "  llama-server ($arm: $*) up after ${LSTART} s (VRAM $(mem_used) MiB)"; return 0; }
  return 1; }
chat_smoke() { # <port>: a real tool-call request with thinking on: the chat template + tool parser must return tool_calls, else Hermes would not work on it
  python3 - "$1" <<'PYEOF'
import json, sys, time, urllib.request
port = sys.argv[1]
body = {"model": "ornith", "messages": [{"role": "user", "content": "What time is it in Paris? Use the tool."}], "max_tokens": 3000, "temperature": 0.6, "stream": False,
        "tools": [{"type": "function", "function": {"name": "get_time", "description": "current time in a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]}
t = time.time()
try:
    r = json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions" % port, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=300).read())
except Exception as e:
    print("SMOKE FAIL request: %s" % str(e)[:160]); sys.exit(1)
m = r["choices"][0]["message"]
tc = m.get("tool_calls") or []
rs = m.get("reasoning") or m.get("reasoning_content") or ""
ok = bool(tc) and tc[0]["function"]["name"] == "get_time" and "city" in json.loads(tc[0]["function"]["arguments"] or "{}")
tm = r.get("timings") or {}
print(("SMOKE OK" if ok else "SMOKE FAIL") + " tool_calls=%d reasoning_chars=%d completion_tokens=%s decode=%.1f %.1fs %s" % (len(tc), len(rs), (r.get("usage") or {}).get("completion_tokens"), tm.get("predicted_per_second") or 0, time.time() - t, json.dumps(tc[0]["function"]) if tc else json.dumps(m)[:200]))
sys.exit(0 if ok else 1)
PYEOF
}
probe() { # <arm> <port>: fixed prompt, temperature 0, 256 tokens, twice; the 2nd run is reported (no agent, no thinking: the same work on both arms)
  python3 - "$2" "$RUN/probe-$1.json" <<'PYEOF'
import json, sys, time, urllib.request
port, out = sys.argv[1], sys.argv[2]
P = ("The lighthouse keeper had kept a logbook for thirty-one years, and in all that time he had never once written down a lie. So when the ship appeared on the horizon on the first calm morning of October, "
     "with no flag, no lights and no one on deck, he sat at his small wooden table for a long while before he picked up the pen. ")
res = []
for i in range(2):
    try:
        r = json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:%s/completion" % port, json.dumps({"prompt": P, "n_predict": 256, "temperature": 0.0, "ignore_eos": True, "cache_prompt": False}).encode(), {"Content-Type": "application/json"}), timeout=600).read())
        tm = r["timings"]
        res.append({"decode_tok_s": round(tm["predicted_per_second"], 2), "predicted_n": tm["predicted_n"], "prompt_tok_s": round(tm.get("prompt_per_second") or 0, 1)})
    except Exception as e:
        res.append({"error": str(e)[:120]})
ok = [x for x in res if "decode_tok_s" in x]
json.dump({"runs": res, "decode_tok_s": ok[-1]["decode_tok_s"] if ok else None}, open(out, "w"))
print("probe", json.dumps(res))
PYEOF
}
run_takes() { # <tag> <upstream>
  local tag=$1 up=$2 n td left w rc
  for n in $(seq 1 "$NT"); do
    td=$RUN/takes/$tag-$n
    if take_done "$tag-$n"; then nl_say "  take $tag-$n: already done (resume), kept"; continue; fi
    left=$(( DEADLINE - $(date +%s) ))
    [ -n "$STUBP" ] && left=99999
    if [ "$left" -lt 660 ]; then nl_say "  take $tag-$n: only $((left/60)) min left (< 11), skipped"; SHORT=1; continue; fi
    w=$(( WALLMIN * 60 )); [ "$((left - 360))" -lt "$w" ] && w=$(( left - 360 ))
    [ -d "$td" ] && mv "$td" "$td.partial-$(date +%s)"
    nl_say "=== take $tag-$n (wall cap $w s, $((left/60)) min left in the window) ==="
    UPSTREAM=$up ALIAS=ornith TAG=$tag-$n PROMPT=$PROMPT SEEDDIR=$SEEDDIR WALL_S=$w CTXCFG=$CTXCFG FR_PORT=$PPORT FR_PY=$RPY FR_GPY=$GPY FR_CHECKOUT=$CHECKOUT FR_QUIET=$QUIET THINK=$THINK ARENA=/tmp/arena99a bash "$H/mc_take.sh" "$td" >> "$RUN/take.log" 2>&1; rc=$?
    nl_say "  take $tag-$n: rc=$rc $(tail -1 "$RUN/take.log" | cut -c1-200)"
    nl_say "  take $tag-$n: $(cat "$td/rec/rc" 2>/dev/null) $(cat "$td/rec/result.json" 2>/dev/null)"
    [ "$rc" = 0 ] || { nl_say "  take $tag-$n: harness rc $rc (see $RUN/take.log)"; SHORT=1; }
    mkdir -p "$td/gate"
    nl_say "  gate $tag-$n: $("$GPY" "$H/rp_gate.py" "$td/work" "$SEEDDIR/tests" "$td/gate" --pytest-python "$GPY" --seed-pkg "$SEEDDIR/shopcart" 2>&1 | head -1 | cut -c1-220)"
  done
}
leg() { # <arm> : plain|cache
  local arm=$1 UP ok=0 N
  if ! leg_pending "$arm"; then nl_say "leg $arm: all $NT takes already done"; nl_mark "serve_$arm" done "resume: skipped"; nl_mark "takes_$arm" done "resume: skipped"; return 0; fi
  nl_mark "serve_$arm" running
  if [ -n "$STUBP" ]; then
    UP=$STUBP; [ "$arm" = cache ] && UP=$STUBC
    echo "{\"loaded\":true,\"stub\":true,\"arm\":\"$arm\"}" > "$RUN/fit-$arm.json"; echo '{"runs":[],"decode_tok_s":null,"stub":true}' > "$RUN/probe-$arm.json"
  else
    LSTART=0
    if [ "$arm" = plain ]; then
      for N in $NCMOE_LADDER; do
        serve_arm plain --n-cpu-moe "$N" && { ok=1; break; }
        nl_say "  plain: --n-cpu-moe $N did not come up (see $RUN/server-plain.log), trying the next"; stop_srv; wait_mem_free || true
      done
      FITX="\"n_cpu_moe\":${N:-0}"
    else
      read -r CMIB CSHARE < <("$GPY" -c "import json;d=json.load(open('$RUN/tune.json'));print(d['mib'],d['share'])")
      serve_arm cache -cmoe --moe-cache-mib "$CMIB" --moe-cache-cpu-share "$CSHARE" && ok=1
      FITX="\"moe_cache_mib\":$CMIB,\"moe_cache_cpu_share\":$CSHARE"
    fi
    if [ "$ok" != 1 ]; then stop_srv; echo "{\"loaded\":false,\"why\":\"did not come up\"}" > "$RUN/fit-$arm.json"; nl_mark "serve_$arm" failed "no load"; nl_say "  llama-server did not come up for $arm: arm BROKEN, going on"; return 0; fi
    UP=http://127.0.0.1:$LPORT
  fi
  SM=$(chat_smoke "${UP##*:}"); nl_say "  tool-call smoke ($arm): $SM"
  case "$SM" in
    "SMOKE OK"*) [ -z "$STUBP" ] && echo "{\"loaded\":true,\"start_s\":${LSTART:-0},\"vram_mib\":$(mem_used),$FITX,\"ctx\":$CTXCFG,\"kv\":\"$KV\",\"smoke\":\"$(echo "$SM" | cut -c1-90 | tr -d '"\\')\"}" > "$RUN/fit-$arm.json";;
    *) echo "{\"loaded\":false,\"why\":\"tool-call smoke failed\"}" > "$RUN/fit-$arm.json"; stop_srv; [ -z "$STUBP" ] && { wait_mem_free || true; }; nl_mark "serve_$arm" failed "smoke"; nl_say "  arm $arm BROKEN: tool-call smoke failed, going on"; return 0;;
  esac
  [ -z "$STUBP" ] && nl_say "  probe ($arm): $(probe "$arm" "${UP##*:}" 2>&1 | tail -1 | cut -c1-240)"
  nl_mark "serve_$arm" done "up"
  nl_mark "takes_$arm" running
  run_takes "$arm" "$UP"
  [ -z "$STUBP" ] && { curl -s -m 5 "$UP/slots" > "$RUN/slots-end-$arm.json" 2>/dev/null; stop_srv; wait_mem_free || fail "takes_$arm" "card not clear after the $arm stop"; }
  nl_mark "takes_$arm" done "$NT takes"
}
tune_leg() {
  if [ -s "$RUN/tune.json" ]; then nl_say "tuner: already done (resume): $(cat "$RUN/tune.json")"; nl_mark tune done "resume: skipped"; return 0; fi
  nl_mark tune running
  local left=$(( (DEADLINE-$(date +%s))/60 )) rc=99 M="" S="" src=fallback
  if [ -n "$STUBP" ] || [ "$SMOKE" = 1 ]; then nl_say "tuner skipped (stub/smoke): fallback parameters"
  elif [ "$left" -lt 70 ]; then nl_say "tuner skipped: only $left min left (< 70): fallback parameters"
  else
    nl_say "tuner: $TUNER on the cache-arm command line (6 server starts, ~10-18 min)"
    timeout 2400 python3 "$TUNER" --out "$RUN/tune-work" -- "$SRV" "${BASEARGS[@]}" -cmoe > "$RUN/tune.out" 2>&1; rc=$?
    TO=$(cat "$RUN/tune.out"); [[ $TO =~ --moe-cache-mib\ ([0-9]+)\ --moe-cache-cpu-share\ ([0-9.]+) ]] && { M=${BASH_REMATCH[1]}; S=${BASH_REMATCH[2]}; src=tuner; }
    nl_say "tuner rc=$rc: ${M:+recommended $M MiB share $S}${M:-no recommendation parsed, fallback ($(tail -1 "$RUN/tune.out" | cut -c1-160))}"
    kill_own_srv; sleep 2; wait_mem_free || fail tune "card not clear after the tuner"
  fi
  [ -n "$M" ] || { M=$FALLBACK_MIB; S=$FALLBACK_SHARE; }
  echo "{\"mib\":$M,\"share\":$S,\"source\":\"$src\",\"tuner_rc\":$rc}" > "$RUN/tune.json"
  nl_mark tune done "$M MiB share $S ($src)"
}

if [ -z "$STUBP" ]; then
  # rule 3, live: nothing but Donald may hold the card when we drain him
  FOREIGN=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ' | tr '\n' ' ')
  nl_say "  card owners before the drain: ${FOREIGN:-none}; tmux: $(tmux ls -F '#{session_name}' 2>/dev/null | tr '\n' ' ')"
  donald_stop
  for i in $(seq 1 60); do v=$(mem_used); [ "${v:-99999}" -lt 1024 ] && { nl_say "drain clear (${v} MiB)"; break; }; sleep 2; done
  [ "${v:-99999}" -lt 1024 ] || fail serve_plain "card memory still ${v:-?} MiB after the drain: another process owns the card, not loading"
else nl_say "STUB mode: no Donald drain, no server start (upstreams $STUBP / $STUBC)"; fi

leg plain
tune_leg
leg cache
# the card is free from here: Donald back, then the CPU-only legs
if [ -z "$STUBP" ]; then donald_restore; fi
nl_say "card released after $(( ($(date +%s)-t0)/60 )) min (drain to restore), the rest is CPU"

# ---- offline re-count of every take's HUD series against the engine's own numbers (CPU, seconds) ----
for td in "$RUN"/takes/*; do
  case "$td" in *.partial-*) continue;; esac
  [ -s "$td/hud/hud.jsonl" ] || continue
  python3 "$H/rp_recount.py" "$td" --json "$RUN/out/recount-$(basename "$td").json" > /dev/null 2>&1
  nl_say "  recount $(basename "$td"): $(python3 -c "import json;d=json.load(open('$RUN/out/recount-$(basename "$td").json'));print('recount',d['new_series_final'],'engine usage',d['exact_sum_predicted_n'],'hud as recorded',d['recorded_hud_final'],'(%s%%)'%d['hud_vs_recount_pct'],'HUD_OK' if d['hud_ok'] else 'HUD_OFF','recount_vs_engine','OK' if d['pass'] else 'OFF')" 2>&1 | cut -c1-200)"
done
# ---- which take goes in the clip (rule fixed in the header: take 1 of each arm unless a recorder fault) ----
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
for e in ("plain", "cache"):
    for n in range(1, takes + 1):
        d = "%s/takes/%s-%d" % (run, e, n)
        if ok(d):
            pick[e] = d
            why[e] = "take %d%s" % (n, "" if n == 1 else " (earlier take had a recorder fault)")
            break
    else:
        pick[e], why[e] = None, "no usable take"
print(json.dumps({"left": pick["plain"], "right": pick["cache"], "why": why}))
PYEOF
LT=$($RPY -c "import json;print(json.load(open('$RUN/pair.json'))['left'] or '')"); RT=$($RPY -c "import json;print(json.load(open('$RUN/pair.json'))['right'] or '')")
nl_say "  clip pair: $(cat "$RUN/pair.json")"
nl_mark gate done "gates ran per take"

# ---- summary (pre-registered rule) then clip (CPU): card FIRST, then the live race cut 8 s after the first pane is done ----
"$GPY" "$H/mc_summary.py" "$RUN" "$SLUG" "$REGIME" > "$RUN/summary.pre.json" 2>>"$RUN/render.log"
VERD=$("$GPY" -c "import json;d=json.load(open('$RUN/summary.pre.json'));print(d['verdict'])" 2>/dev/null)
nl_say "  verdict (pre-registered rule): ${VERD:-unreadable}"
nl_mark render running
rr=1
if [ -n "$LT" ] && [ -n "$RT" ]; then
  "$RPY" "$H/mc_render.py" race "$RUN/out/race.mp4" --left "$LT" --right "$RT" --ffmpeg "$FF" --stills "$RUN/out/stills" --cut-after-first 8 --real-max-s 40 --target-main-s 30 --open-s 10 --close-s 12 --pre-s 3 >> "$RUN/render.log" 2>&1; rr=$?
  if [ "$rr" = 0 ]; then
    "$GPY" "$H/mc_summary.py" "$RUN" "$SLUG" "$REGIME" > "$RUN/summary.json.tmp" 2>>"$RUN/render.log" && mv "$RUN/summary.json.tmp" "$RUN/summary.card.json"
    "$RPY" "$H/mc_facts.py" "$RUN/summary.card.json" "$RUN/facts.json" >> "$RUN/render.log" 2>&1 && \
    "$RPY" "$H/mc_render.py" card "$RUN/out/card.mp4" --facts "$RUN/facts.json" --ffmpeg "$FF" --seconds 8 >> "$RUN/render.log" 2>&1 && \
    "$FF" -nostdin -hide_banner -loglevel error -y -i "$RUN/out/card.mp4" -i "$RUN/out/race.mp4" \
      -filter_complex "[0:v]setsar=1,fps=25[a];[1:v]setsar=1,fps=25[b];[a][b]concat=n=2:v=1:a=0[v]" -map "[v]" -c:v libx264 -crf 20 -pix_fmt yuv420p -movflags +faststart "$RUN/out/moecache-final.mp4" </dev/null >> "$RUN/render.log" 2>&1
    rr=$?
  fi
else nl_say "  no usable take on one of the arms: no race clip (the table and verdict still stand)"; fi
"$RPY" - "$RUN/out" "$RUN/out/contact.png" <<'PYEOF' 2>>"$RUN/render.log"
import glob, os, sys
from PIL import Image
out = sys.argv[1]
fs = sorted(glob.glob(out + "/stills/*.png"))[:8]
fs += [p for p in (out + "/card.mp4.png",) if os.path.exists(p)]
if fs:
    w, h = 640, 360
    sheet = Image.new("RGB", (w * 3, h * 3), "#000")
    for i, f in enumerate(fs[:9]):
        sheet.paste(Image.open(f).convert("RGB").resize((w, h)), ((i % 3) * w, (i // 3) * h))
    sheet.save(sys.argv[2])
PYEOF
nl_say "  render rc=$rr; final clip: $([ -f "$RUN/out/moecache-final.mp4" ] && du -h "$RUN/out/moecache-final.mp4" | cut -f1 || echo missing)"
if [ "$SHORT" = 1 ] || [ "$VERD" = INCOMPLETE ]; then
  nl_say "  a leg could not finish inside the window (verdict ${VERD:-?}): exit 3, resumable, summary.json NOT written (PROVISIONAL: $RUN/summary.pre.json)"
  nl_mark render failed "incomplete, resumable"
  exit 3
fi
"$GPY" "$H/mc_summary.py" "$RUN" "$SLUG" "$REGIME" > "$ART" 2>>"$RUN/render.log"
nl_judge_artifact render "$rr" "$ART" "$t0" "$(( ($(date +%s)-t0)/60 )) min, render rc $rr" || { nl_say "FAIL: no fresh summary.json or render rc $rr"; exit 1; }
nl_say "  summary: verdict $VERD; line 1: $("$GPY" -c "import json;d=json.load(open('$ART'));print(d.get('line1'))")"
nl_say "DONE $SLUG"
exit 0
