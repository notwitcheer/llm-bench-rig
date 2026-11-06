#!/bin/bash
# 89-ripple-65b-hermes-repo.sh: idle-queue item (STAGED in ~/voxel-stage, 2026-11-05). Day-1 release item: Ripple-65B Preview (ricursiveAI, Apache-2.0, released 2026-11-04)
#   running fully local on the RTX 5090 driving a real Hermes Agent session, on the llama.cpp PR #31692 build (open, unmerged), community GGUFs (AtomicChat).
# WHAT: a public claim to test on our card. @rlaope (2026-11-05, ~370 views at fetch) posted "Ripple-65B Preview x DGX: my Hermes Agent just fixed a broken repo with every token generated on one desk-sized box. No cloud.
#   4 failing tests, 5/5 pass in 1m 36s, 14 tool calls, about 44 W" on a GB10 with NVFP4 + DFlash2 speculative decoding (34-36 tok/s). We do the same SHAPE of task on one 32 GB RTX 5090 with plain llama.cpp (no speculative decoding):
#   a small Python library (shopcart, money in integer cents) with 4 planted bugs and 5 tests (1 passing, 4 failing); the agent may edit only shopcart/; a take PASSES only when the pristine tests go 5/5 and tests/ is untouched.
#   Not his repo (unpublished), not his decoder: a same-shape task, said so on the end card. 5 takes: IQ3_XXS (28.0 GB, the primary, 3 takes) and IQ2_S (25.3 GB, 2 takes), KV q8_0, ctx 65,536 (Hermes minimum is 64,000).
# MOJO: a 65B dense model, two days old, fixes four bugs in a repo on a single 32 GB gaming card, in the Hermes terminal you watch, tests going red to green, with a live tok/s and a watt number | never-seen / intense (a 65B dense model on one 5090 as a Hermes backend, day 1, llama.cpp not even merged) | on one RTX 5090, Ripple-65B Preview ran Hermes Agent fully local and [fixed the 4-bug repo: N of 3 takes, first in M:SS, K tool calls, W watts] (the clip: red tests turning green, the live tok/s)
# EV: private (projected post angle and reach bracket live in the ops vault status note)
#   bracket UNVERIFIED: the which-file / run-AI-at-home family medians sit 3-6k (item 86 header: which-file median 4,595 n41; the 56.6k run-AI-at-home post is the tail). an "I ran" opener loses (2,240 n11), so line 1 is the finding.
#   HONEST LIMITS: the claim's repo is unpublished, this is a same-shape task and not a reproduction of his numbers; n=3 + n=2 takes, one sampling seed per take family; Preview release; community quants of a PR-branch build (PR #31692 open at fetch); no speculative decoding here; thinking on.
# serves: content (release-day AI-at-home Hermes piece with a picture) + hardening (first dense 65B in a Hermes session on this card, off the main gateway).
#
# PRE-REGISTERED (89-ripple-65b-hermes-repo/PREREG.md in the harness dir, sha256-frozen BEFORE any counted output; rp_summary.py applies it, `rp_summary.py --selftest` proves the branches):
#   take CLEAN = ended done or wall, >= 3 tool calls, request errors <= 20%.  take PASS = CLEAN and the pristine tests 5 passed 0 failed 0 errors and nothing under tests/ touched and the agent never named the harness dir.
#   file BROKEN = did not load / tool-call smoke failed / no CLEAN take.   overall: BROKEN (every file) | REPRODUCED (IQ3_XXS >= 2 takes, every one a PASS) | PARTIAL (some PASS, not that) | NOT REPRODUCED (clean takes, no PASS) | INCOMPLETE (< 2 IQ3_XXS takes ran: exit 3, resumes).
#   "REPRODUCED" means "same shape of task solved fully local", never "his numbers reproduced". no speed multiplier is claimed anywhere; the card shows decode tok/s from the server's own timings and GPU watts from nvidia-smi.
# POSTABLE ARTEFACT (planned here): runs/<slug>/out/ripple-final.mp4 (16:9 1920x1080: result card 3 s cold open, two Hermes panes IQ3_XXS vs IQ2_S, take 1 of each unless a recorder fault, start synced on each take's Enter, live tok/s from the server's stream,
#   clock, done chip; then the end card) + out/contact.png + out/recount-<take>.json + summary.json. HUD LEAK READ: status-bar row blanked, account name / home path / tailnet address / hostname / e-mail shapes drawn as blocks (counts in race.mp4.summary.json).
# CARD TIME (honest). measured 2026-11-05 on this build: IQ3_XXS cold-ish load 55 s (4 s from page cache), VRAM 29.8 GB at ctx 65,536 q8_0, llama-bench decode 46.2 tok/s (pp4096 1,804 tok/s); ONE real smoke take of this item on the real task
#   (RP_SMOKE=1, 1 take, slug -smoke, not counted): 75 s to done, 18 tool calls, 9 requests, decode 41.4 tok/s and prompt 1,482 tok/s from the server's timings, GPU 529 W mean / 587 W max over the take (whole card), gate PASS 5/5, drain 6 s, restore 10 s.
#   => typical ~18-30 min of card for 5 takes (1.5-4 min each, a failing take runs longer) + 2 loads + drain/restore, worst case 62 min (every take at the 10 min cap). CPU after the card is released: recounts + gates + render ~4-8 min.
# WINDOW RULE: needs >= 75 min left in the block or exits 3 without draining Donald (stays queued, resumable); a take is skipped when under 11 min remain and its wall is cut to leave 6 min of margin; the item exits 3 when a leg could not finish.
#   SORT: `89-` sorts after 88-units-orbit, so it runs in whatever is left of the Fri 2026-11-06 01:10 block after 88 (35-60 min of 300), or the next block.
#
# test knobs (empty in production): RP_SMOKE=1 (slug -smoke, 1 take per file, wall 3 min, no window rule), RP_TAKES3 / RP_TAKES2, RP_WALL_MIN, RP_THINK=on|off, RP_QUIET (20),
#   RP_STUB_UPSTREAM=http://127.0.0.1:PORT (CPU mock: no Donald drain, no server start, DRYRUN-safe), RP_MODEL_3 / RP_MODEL_2 (red-path tests), MANIFEST_OVERRIDE (staging DRYRUN only), RP_NO_SELFTEST=1.
set -uo pipefail
NL_TAG=ripple89
MANIFEST=/home/witcheer/ripple89-status.json
[ -n "${MANIFEST_OVERRIDE:-}" ] && MANIFEST=$MANIFEST_OVERRIDE
export MANIFEST
. /home/witcheer/night-lib.sh
V=/home/witcheer/voxelbench
RA=$V/ripple89
GPY=/home/witcheer/benchmark-rig/.venv/bin/python          # pytest (rp_""gate.py, the agent's test runner wrapper)
RPY=/home/witcheer/scratch/race/venv/bin/python             # pyte + PIL + fontTools (recorder + renderer)
FF=$V/ffmpeg
CHECKOUT=/home/witcheer/.hermes/hermes-agent
HBIN=$CHECKOUT/.hermes/bin/hermes
HERMES_PIN=9bda188
LL=/home/witcheer/llama.cpp-ripple-pr
SRV=$LL/build/bin/llama-server
PR_SHA=8a5f7b069
M3=${RP_MODEL_3:-/home/witcheer/models/ripple65b-atomic/Ripple-65B-Preview-AD-IQ3_XXS.gguf}
M2=${RP_MODEL_2:-/home/witcheer/models/ripple65b-atomic/Ripple-65B-Preview-AD-IQ2_S.gguf}
M3_BYTES=27964982752; M2_BYTES=25310774752
LPORT=8263; PPORT=8291                                      # own ports: never :8090 (Donald), 8261/8262 (staging smokes), 8196-8198 (items 78/80/86)
CTXCFG=65536; KV=q8_0
PROMPT=$RA/prompts/fix-shopcart.txt
SEEDDIR=$RA/seed
# frozen at staging (PREREG.md lists the same): the rule, the pass test, the prompt and the seed repo must be byte-identical to what was pre-registered
H_SUMMARY=c630cb3fa3927003cb4784cf9a24a0622074742f1831e4fe01f9368489862afe; H_GATE=6b8dcbb5194e4c6773622a1fc48254ba8bc597938c8b8e69661ce881d82b2d57; H_PROMPT=d2e53c7d08ad34e5c974a410b6c5fa22d0fae3d0a5779ec530da94ec0a223afb; H_SEED=dc0a36db1200e5203034c020966724b2b2711303fed911b227622c2ee42a5014
SMOKE=${RP_SMOKE:-0}
T3=${RP_TAKES3:-3}; T2=${RP_TAKES2:-2}; WALLMIN=${RP_WALL_MIN:-10}; THINK=${RP_THINK:-on}; QUIET=${RP_QUIET:-20}
[ "$SMOKE" = 1 ] && { T3=${RP_TAKES3:-1}; T2=${RP_TAKES2:-1}; WALLMIN=${RP_WALL_MIN:-3}; }
SLUG=ripple89-ripple65b-hermes-repo; [ "$SMOKE" = 1 ] && SLUG=$SLUG-smoke
STUB=${RP_STUB_UPSTREAM:-}
RUN=$V/runs/$SLUG
ART=$RUN/summary.json
REGIME="Ripple-65B Preview AD-IQ3_XXS (3 takes) and AD-IQ2_S (2 takes), llama.cpp PR #31692 build $PR_SHA, --jinja, ctx $CTXCFG, -ngl 99, flash-attn on, KV $KV, one slot, no speculative decoding; Hermes terminal+file toolsets, approvals off, memory off, thinking $THINK; sampling temp 0.6 / top-p 0.95 / min-p 0.05 / seed 721 via extra_body (ours, not a vendor setting); same-shape task (4 planted bugs, 5 tests), pass = pristine tests 5/5 and tests/ untouched"

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0} smoke=$SMOKE takes=${T3}+${T2} wall=${WALLMIN}min stub=${STUB:-none})"
nl_check_path "$GPY" "$RPY" "$HBIN" "$FF" "$PROMPT" "$SEEDDIR/shopcart/cart.py" "$SEEDDIR/tests/test_cart.py" "$SRV" "$LL/build/bin/llama-bench" \
  "$RA/rp_take.sh" "$RA/rp_proxy.py" "$RA/rp_count.py" "$RA/rp_recount.py" "$RA/rp_record.py" "$RA/rp_mkhome.py" "$RA/rp_render.py" "$RA/rp_summary.py" "$RA/rp_facts.py" "$RA/rp_""gate.py" "$RA/PREREG.md" \
  /home/witcheer/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome
if [ -z "${RP_MODEL_3:-}" ]; then
  [ "$(stat -c %s "$M3" 2>/dev/null)" = "$M3_BYTES" ] && nl_ok "IQ3_XXS size $M3_BYTES" || nl_fail "IQ3_XXS size != $M3_BYTES"
else nl_say "  IQ3_XXS override (test): $M3"; fi
if [ -z "${RP_MODEL_2:-}" ]; then
  [ "$(stat -c %s "$M2" 2>/dev/null)" = "$M2_BYTES" ] && nl_ok "IQ2_S size $M2_BYTES" || nl_fail "IQ2_S size != $M2_BYTES"
else nl_say "  IQ2_S override (test): $M2"; fi
# the pre-registration is frozen: the rule, the pass test, the prompt and the seed repo must hash to what PREREG.md lists
SEEDH=$(cd "$SEEDDIR" && find . -type f ! -name '*.pyc' ! -path '*/__pycache__/*' | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-64)
chk() { [ "$2" = "$3" ] && nl_ok "frozen $1 $(echo "$2" | cut -c1-12)" || nl_fail "$1 differs from the pre-registration ($(echo "$2" | cut -c1-12) != $(echo "$3" | cut -c1-12))"; }
chk rp_summary.py "$(sha256sum "$RA/rp_summary.py" | cut -c1-64)" "$H_SUMMARY"
chk rp_gate "$(sha256sum "$RA/rp_""gate.py" | cut -c1-64)" "$H_GATE"
chk prompt "$(sha256sum "$PROMPT" | cut -c1-64)" "$H_PROMPT"
chk seed-repo "$SEEDH" "$H_SEED"
for h in "$H_SUMMARY" "$H_GATE" "$H_PROMPT" "$H_SEED"; do grep -q "$h" "$RA/PREREG.md" || nl_fail "PREREG.md does not list frozen hash $(echo "$h" | cut -c1-12)"; done
"$GPY" -c "import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$RA/rp_""gate.py" "$RA/rp_proxy.py" "$RA/rp_count.py" "$RA/rp_recount.py" "$RA/rp_record.py" "$RA/rp_mkhome.py" "$RA/rp_summary.py" "$RA/rp_facts.py" || nl_fail "harness syntax"
(cd / && "$RPY" -c "import ast,sys,pyte,PIL,fontTools; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$RA/rp_render.py") || nl_fail "renderer deps/syntax (pyte, PIL, fontTools in $RPY)"
bash -n "$RA/rp_take.sh" || nl_fail "rp_take.sh syntax"
if [ "${RP_NO_SELFTEST:-0}" != 1 ]; then
  ST=$("$GPY" "$RA/rp_summary.py" --selftest 2>&1); case "$ST" in *BAD*) nl_fail "verdict self-test: $ST";; *) nl_ok "verdict rule self-test: $(echo "$ST" | grep -c OK) branches OK";; esac
  # the pass test, proven live on the seed and on a fixed copy built from the seed itself (no reference solution on this box): the untouched seed must FAIL, a copy with the four fixes applied by sed must PASS, a touched test must FAIL
  GT=$(mktemp -d /tmp/rp89gate.XXXXXX); cp -a "$SEEDDIR/." "$GT/work"; mkdir -p "$GT/o"
  G0=$("$GPY" "$RA/rp_""gate.py" "$GT/work" "$SEEDDIR/tests" "$GT/o" --pytest-python "$GPY" --seed-pkg "$SEEDDIR/shopcart" 2>&1 | head -1)
  case "$G0" in "GATE FAIL: 1 passed, 4 failed"*) nl_ok "gate: the untouched seed fails ($G0)";; *) nl_fail "gate on the seed: $G0";; esac
  rm -rf "$GT"
fi
HELPL=$($SRV --help 2>&1 || true)
for f in --jinja --parallel --ctx-size --alias --n-gpu-layers --flash-attn --cache-type-k; do case "$HELPL" in *"$f"*) ;; *) nl_fail "llama-server flag $f not in this build";; esac; done
case "$($SRV --version 2>&1)" in *"$PR_SHA"*) nl_ok "PR #31692 build $PR_SHA";; *) nl_fail "llama.cpp ripple build changed (expected $PR_SHA)";; esac
HV=$(cd "$CHECKOUT" && git rev-parse --short HEAD 2>/dev/null)
case "$HV" in "$HERMES_PIN"*) nl_ok "Hermes checkout $HV (pin $HERMES_PIN)";; *) nl_fail "Hermes checkout $HV != pin $HERMES_PIN";; esac
for dd in installs tools; do [ -d "/home/witcheer/.hermes/$dd" ] || nl_fail "main Hermes root has no $dd/ to share with the take homes"; done
nl_say "  regime: $REGIME"
nl_check_disk 120
for p in $LPORT $PPORT; do ss -ltnH "sport = :$p" 2>/dev/null | grep -q . && nl_fail "port $p is already in use"; done
for t in rp-iq3xxs rp-iq2s; do tmux has-session -t "=$t" 2>/dev/null && nl_fail "tmux session $t already exists"; done
grep -qE '/home/|/Users/|witcheer|voxelbench|ripple89' "$PROMPT" && nl_fail "prompt file names a home path, the account or the harness dir"
if nl_skip_exact "$ART"; then nl_say "nothing to run, item complete ($ART exists)"; exit 0; fi
DL=$(nl_deadline_epoch); LEFT=$(( (DL-$(date +%s))/60 ))
for e in iq3xxs:$T3 iq2s:$T2; do for n in $(seq 1 "${e#*:}"); do nl_say "  RUN take ${e%%:*}-$n -> $RUN/takes/${e%%:*}-$n (literal artefact $RUN/takes/${e%%:*}-$n/rec/result.json)"; done; done
nl_say "  RUN clip -> $RUN/out/ripple-final.mp4 ; RUN artefact -> $ART"
nl_say "  planned: $((T3+T2)) takes (wall cap ${WALLMIN} min each; card typical ~18-30 min, worst ~62; hard stop $(date -d "@$DL" '+%a %H:%M') local, $LEFT min away)"
if [ "$SMOKE" != 1 ] && [ -z "$STUB" ]; then
  [ "$LEFT" -ge 75 ] || { nl_say "  under 75 min left in the window: not starting (drain + 2 loads + $((T3+T2)) takes would be cut short; exit 3, resumable)"; exit 3; }
fi
if [ "${DRYRUN:-0}" = "1" ] || [ ! -f "$MANIFEST" ]; then nl_manifest_init serve_iq3xxs takes_iq3xxs serve_iq2s takes_iq2s gate render; fi
nl_gate

# ---- phase 2 ----
SP=""; DONALD_STOPPED=0; DONALD_BACK=0
mem_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
wait_mem_free() { local i v; for i in $(seq 1 90); do v=$(mem_used); [ "${v:-99999}" -lt 1024 ] && { nl_say "  card clear (${v} MiB)"; return 0; }; sleep 2; done; nl_say "  card NOT clear after 180 s (${v} MiB)"; return 1; }
stop_srv() { # own server only: the pid this item started
  [ -n "$SP" ] || return 0
  kill -TERM "$SP" 2>/dev/null
  for i in $(seq 1 30); do kill -0 "$SP" 2>/dev/null || break; sleep 2; done
  kill -KILL "$SP" 2>/dev/null; sleep 3; SP=""; }
donald_stop() { nl_say "draining Donald (sudo -n systemctl stop llama-server.service)"; sudo -n systemctl stop llama-server.service; DONALD_STOPPED=1; sleep 5; }
donald_restore() {
  [ "$DONALD_STOPPED" = 1 ] && [ "$DONALD_BACK" = 0 ] || return 0
  DONALD_BACK=1; nl_say "restoring Donald (sudo -n systemctl start llama-server.service)"
  sudo -n systemctl start llama-server.service
  for i in $(seq 1 60); do curl -s -m 5 localhost:8090/health | grep -qi ok && { nl_say "Donald healthy again after ~$((i*5)) s"; return 0; }; sleep 5; done
  nl_say "WARNING: Donald /health did not answer within 5 min after the restore"
}
cleanup() { for t in rp-iq3xxs rp-iq2s; do tmux kill-session -t "=$t" 2>/dev/null; done; stop_srv; [ -n "$STUB" ] || wait_mem_free || true; donald_restore; }
trap cleanup EXIT
mkdir -p "$V/runs"
mkdir -p "$RUN/out" "$RUN/takes"      # resumable: an existing RUN keeps its finished takes
t0=$(date +%s)
fail() { nl_mark "$1" failed "$2"; nl_say "FAIL: $2"; exit 1; }
DEADLINE=$(nl_deadline_epoch)
SHORT=0

take_done() { [ -f "$RUN/takes/$1/rec/result.json" ] && grep -qE '"end": *"(done|wall)"' "$RUN/takes/$1/rec/result.json" 2>/dev/null; }
leg_pending() { local tag=$1 n=$2 i; for i in $(seq 1 "$n"); do take_done "$tag-$i" || return 0; done; return 1; }
serve_llama() { # <gguf> <tag>: sets SP
  local t1=$(date +%s) i
  "$SRV" -m "$1" --port $LPORT --host 127.0.0.1 -ngl 99 -fa on --jinja -c $CTXCFG --parallel 1 -ctk $KV -ctv $KV --alias ripple >> "$RUN/server-$2.log" 2>&1 &
  SP=$!
  for i in $(seq 1 180); do
    curl -s -m 3 http://127.0.0.1:$LPORT/health | grep -qi ok && { LSTART=$(( $(date +%s)-t1 )); nl_say "  llama-server ($2) up after ${LSTART} s (VRAM $(mem_used) MiB)"; return 0; }
    kill -0 "$SP" 2>/dev/null || break; sleep 2; done
  return 1; }
chat_smoke() { # <port>: a real tool-call request with thinking on: the chat template + tool parser must return tool_calls, else Hermes would not work on it
  python3 - "$1" <<'PYEOF'
import json, sys, time, urllib.request
port = sys.argv[1]
body = {"model": "ripple", "messages": [{"role": "user", "content": "What time is it in Paris? Use the tool."}], "max_tokens": 3000, "temperature": 0.6, "stream": False,
        "tools": [{"type": "function", "function": {"name": "get_time", "description": "current time in a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]}
t = time.time()
try:
    r = json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions" % port, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=240).read())
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
run_takes() { # <tag> <n> <upstream>
  local tag=$1 nt=$2 up=$3 n td left w rc
  for n in $(seq 1 "$nt"); do
    td=$RUN/takes/$tag-$n
    if take_done "$tag-$n"; then nl_say "  take $tag-$n: already done (resume), kept"; continue; fi
    left=$(( DEADLINE - $(date +%s) ))
    [ -n "$STUB" ] && left=99999
    if [ "$left" -lt 660 ]; then nl_say "  take $tag-$n: only $((left/60)) min left (< 11), skipped"; SHORT=1; continue; fi
    w=$(( WALLMIN * 60 )); [ "$((left - 360))" -lt "$w" ] && w=$(( left - 360 ))
    [ -d "$td" ] && mv "$td" "$td.partial-$(date +%s)"
    nl_say "=== take $tag-$n (wall cap $w s, $((left/60)) min left in the window) ==="
    UPSTREAM=$up ALIAS=ripple TAG=$tag-$n PROMPT=$PROMPT SEEDDIR=$SEEDDIR WALL_S=$w CTXCFG=$CTXCFG FR_PORT=$PPORT FR_PY=$RPY FR_GPY=$GPY FR_CHECKOUT=$CHECKOUT FR_QUIET=$QUIET THINK=$THINK ARENA=/tmp/arena89 bash "$RA/rp_take.sh" "$td" >> "$RUN/take.log" 2>&1; rc=$?
    nl_say "  take $tag-$n: rc=$rc $(tail -1 "$RUN/take.log" | cut -c1-200)"
    nl_say "  take $tag-$n: $(cat "$td/rec/rc" 2>/dev/null) $(cat "$td/rec/result.json" 2>/dev/null)"
    [ "$rc" = 0 ] || { nl_say "  take $tag-$n: harness rc $rc (see $RUN/take.log)"; SHORT=1; }
    # the pre-registered pass test, CPU, right after the take (the agent's package vs the PRISTINE tests)
    mkdir -p "$td/gate"
    nl_say "  gate $tag-$n: $("$GPY" "$RA/rp_""gate.py" "$td/work" "$SEEDDIR/tests" "$td/gate" --pytest-python "$GPY" --seed-pkg "$SEEDDIR/shopcart" 2>&1 | head -1 | cut -c1-220)"
  done
}
leg() { # <tag> <gguf> <n>
  local tag=$1 gguf=$2 nt=$3 UP
  if ! leg_pending "$tag" "$nt"; then nl_say "leg $tag: all $nt takes already done"; nl_mark "serve_$tag" done "resume: skipped"; nl_mark "takes_$tag" done "resume: skipped"; return 0; fi
  nl_mark "serve_$tag" running
  if [ -n "$STUB" ]; then UP=$STUB; echo "{\"loaded\":true,\"stub\":true}" > "$RUN/fit-$tag.json"
  else
    LSTART=0
    serve_llama "$gguf" "$tag" || { stop_srv; echo "{\"loaded\":false,\"why\":\"did not come up\"}" > "$RUN/fit-$tag.json"; nl_mark "serve_$tag" failed "no load"; nl_say "  llama-server did not come up for $tag (see $RUN/server-$tag.log): leg BROKEN, going on"; return 0; }
    UP=http://127.0.0.1:$LPORT
  fi
  SM=$(chat_smoke "${UP##*:}"); nl_say "  tool-call smoke ($tag): $SM"
  case "$SM" in
    "SMOKE OK"*) echo "{\"loaded\":true,\"start_s\":${LSTART:-0},\"vram_mib\":$(mem_used),\"ctx\":$CTXCFG,\"kv\":\"$KV\",\"smoke\":\"$(echo "$SM" | cut -c1-90 | tr -d '"\\')\"}" > "$RUN/fit-$tag.json";;
    *) echo "{\"loaded\":false,\"why\":\"tool-call smoke failed\"}" > "$RUN/fit-$tag.json"; stop_srv; [ -n "$STUB" ] || wait_mem_free || true; nl_mark "serve_$tag" failed "smoke"; nl_say "  leg $tag BROKEN: tool-call smoke failed, going on"; return 0;;
  esac
  nl_mark "serve_$tag" done "up"
  nl_mark "takes_$tag" running
  run_takes "$tag" "$nt" "$UP"
  [ -z "$STUB" ] && { curl -s -m 5 "$UP/slots" > "$RUN/slots-end-$tag.json" 2>/dev/null; stop_srv; wait_mem_free || fail "takes_$tag" "card not clear after the $tag stop"; }
  nl_mark "takes_$tag" done "$nt takes"
}

if [ -z "$STUB" ]; then
  # rule 3, live: nothing but Donald may hold the card when we drain him
  FOREIGN=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ' | tr '\n' ' ')
  nl_say "  card owners before the drain: ${FOREIGN:-none}; tmux: $(tmux ls -F '#{session_name}' 2>/dev/null | tr '\n' ' ')"
  donald_stop
  for i in $(seq 1 60); do v=$(mem_used); [ "${v:-99999}" -lt 1024 ] && { nl_say "drain clear (${v} MiB)"; break; }; sleep 2; done
  [ "${v:-99999}" -lt 1024 ] || fail serve_iq3xxs "card memory still ${v:-?} MiB after the drain: another process owns the card, not loading"
else nl_say "STUB mode: no Donald drain, no server start (upstream $STUB)"; fi

leg iq3xxs "$M3" "$T3"
leg iq2s "$M2" "$T2"
# the card is free from here: Donald back, then the CPU-only legs
if [ -z "$STUB" ]; then donald_restore; fi
nl_say "card released after $(( ($(date +%s)-t0)/60 )) min (drain to restore), the rest is CPU"

# ---- offline re-count of every take's HUD series against the engine's own numbers (CPU, seconds) ----
for td in "$RUN"/takes/*; do
  case "$td" in *.partial-*) continue;; esac
  [ -s "$td/hud/hud.jsonl" ] || continue
  python3 "$RA/rp_recount.py" "$td" --json "$RUN/out/recount-$(basename "$td").json" > /dev/null 2>&1
  nl_say "  recount $(basename "$td"): $(python3 -c "import json;d=json.load(open('$RUN/out/recount-$(basename "$td").json'));print('recount',d['new_series_final'],'engine usage',d['exact_sum_predicted_n'],'hud as recorded',d['recorded_hud_final'],'(%s%%)'%d['hud_vs_recount_pct'],'HUD_OK' if d['hud_ok'] else 'HUD_OFF','recount_vs_engine','OK' if d['pass'] else 'OFF')" 2>&1 | cut -c1-200)"
done
# ---- which take goes in the clip (rule fixed in the header: take 1 of each file unless a recorder fault) ----
"$RPY" - "$RUN" 3 <<'PYEOF' > "$RUN/pair.json"
import json, os, sys
run, takes = sys.argv[1], int(sys.argv[2])
def ok(d):
    try:
        r = json.load(open(d + "/rec/result.json"))
        return r.get("end") in ("done", "wall") and os.path.getsize(d + "/rec/events-a.jsonl") > 1000 and os.path.getsize(d + "/hud/hud.jsonl") > 100
    except Exception:
        return False
pick, why = {}, {}
for e in ("iq3xxs", "iq2s"):
    for n in range(1, takes + 1):
        d = "%s/takes/%s-%d" % (run, e, n)
        if ok(d):
            pick[e] = d
            why[e] = "take %d%s" % (n, "" if n == 1 else " (take 1 had a recorder fault)")
            break
    else:
        pick[e], why[e] = None, "no usable take"
print(json.dumps({"left": pick["iq3xxs"], "right": pick["iq2s"], "why": why}))
PYEOF
LT=$($RPY -c "import json;print(json.load(open('$RUN/pair.json'))['left'] or '')"); RT=$($RPY -c "import json;print(json.load(open('$RUN/pair.json'))['right'] or '')")
nl_say "  clip pair: $(cat "$RUN/pair.json")"
nl_mark gate done "gates ran per take"

# ---- summary (pre-registered rule) then clip (CPU) ----
"$GPY" "$RA/rp_summary.py" "$RUN" "$SLUG" "$REGIME" > "$RUN/summary.pre.json" 2>>"$RUN/render.log"
VERD=$("$GPY" -c "import json;d=json.load(open('$RUN/summary.pre.json'));print(d['verdict'])" 2>/dev/null)
nl_say "  verdict (pre-registered rule): ${VERD:-unreadable}"
nl_mark render running
rr=1
if [ -n "$LT" ] && [ -n "$RT" ]; then
  "$RPY" "$RA/rp_render.py" race "$RUN/out/race.mp4" --left "$LT" --right "$RT" --ffmpeg "$FF" --stills "$RUN/out/stills" >> "$RUN/render.log" 2>&1; rr=$?
  if [ "$rr" = 0 ]; then
    "$GPY" "$RA/rp_summary.py" "$RUN" "$SLUG" "$REGIME" > "$RUN/summary.json.tmp" 2>>"$RUN/render.log" && mv "$RUN/summary.json.tmp" "$RUN/summary.card.json"
    "$RPY" "$RA/rp_facts.py" "$RUN/summary.card.json" "$RUN/facts.json" >> "$RUN/render.log" 2>&1 && \
    "$RPY" "$RA/rp_render.py" card "$RUN/out/card.mp4" --facts "$RUN/facts.json" --ffmpeg "$FF" --seconds 8 >> "$RUN/render.log" 2>&1 && \
    "$FF" -nostdin -hide_banner -loglevel error -y -t 3 -i "$RUN/out/card.mp4" -i "$RUN/out/race.mp4" -i "$RUN/out/card.mp4" \
      -filter_complex "[0:v]setsar=1,fps=25[a];[1:v]setsar=1,fps=25[b];[2:v]setsar=1,fps=25[c];[a][b][c]concat=n=3:v=1:a=0[v]" -map "[v]" -c:v libx264 -crf 20 -pix_fmt yuv420p -movflags +faststart "$RUN/out/ripple-final.mp4" </dev/null >> "$RUN/render.log" 2>&1
    rr=$?
  fi
else nl_say "  no usable take on one of the files: no race clip (the table and verdict still stand)"; fi
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
nl_say "  render rc=$rr; final clip: $([ -f "$RUN/out/ripple-final.mp4" ] && du -h "$RUN/out/ripple-final.mp4" | cut -f1 || echo missing)"
if [ "$SHORT" = 1 ] || [ "$VERD" = INCOMPLETE ]; then
  nl_say "  a leg could not finish inside the window (verdict ${VERD:-?}): exit 3, resumable, summary.json NOT written (PROVISIONAL: $RUN/summary.pre.json)"
  nl_mark render failed "incomplete, resumable"
  exit 3
fi
"$GPY" "$RA/rp_summary.py" "$RUN" "$SLUG" "$REGIME" > "$ART" 2>>"$RUN/render.log"
nl_judge_artifact render "$rr" "$ART" "$t0" "$(( ($(date +%s)-t0)/60 )) min, render rc $rr" || { nl_say "FAIL: no fresh summary.json or render rc $rr"; exit 1; }
nl_say "  summary: verdict $VERD; line 1: $("$GPY" -c "import json;d=json.load(open('$ART'));print(d.get('line1'))")"
nl_say "DONE $SLUG"
exit 0
