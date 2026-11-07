#!/bin/bash
# 90a-qwen38-nerf-audit.sh: idle-queue item (STAGED in ~/voxel-stage, 2026-11-06, mercury subagent for the GPU thread, mojo-scan pick E). Sorts first among the bench items of the Sat drain on purpose: the thread's news wave closes ~48 h after 11-05.
# WHAT: "did they nerf Qwen3.8-27B?" answered with the file and our own August board. (1) sha256 of the Q6_K we serve and of the plain Q4_K_M, against the LFS oids Hugging Face listed for the exact files our day-0 board ran
#   (unsloth/Qwen3.8-27B-GGUF @ f1bfb127, 15 Aug) and the weights repo's shard oids against its 13 Aug upload; (2) the same board items re-run with the same harness (benchmark-rig lib/evals @ 6c7ecd8, copied unchanged), think-off, greedy:
#   GPQA-diamond 198 + HumanEval 164 + ARC-Challenge 1,172 + 19 of 57 MMLU subjects (2,614 items, the day-0 50% sample), against the day-0 per-item records. Scores carry Wilson 95% bands, per-item agreement is paired (exact McNemar);
#   the runtime build differs from day 0 and is labelled: passes = NEW (Donald build 839a572a4, b11777) and OLD (5d16e81dd, b10371, the build the board config pointed at from 11 Aug) x Q6_K (headline) and Q4_K_M (secondary).
# MOJO: the file's sha256 typed against August's character by character (64 of 64 green), then 362 answers replayed side by side, August vs today, with a running "same answer" counter, then today's scores landing inside August's bands | funny (the hash cannot lie; the thread was satire) + honest null | Qwen3.8-27B was not nerfed: the Q6_K is byte-identical to August and the same board gives [GPQA N/198 vs 97/198, HumanEval N/164 vs 155/164] (same-answer counts in reply 1)
# EV: private (projected post angle and reach bracket live in the ops vault status note)
#   bracket UNVERIFIED: ~1.5k-5k, riding a thread that closes inside ~48 h (mojo-scan-2026-11-06 pick 5; the day-0 post it follows had 8,995 views). a null is the EXPECTED result and is the joke; a shift is a finding about the runtime, never the weights (the hash decides).
#   RAILS: never name or quote the thread's poster; hosted-API nerf claims are out of scope and the card says so; never "they were wrong", say what the file and the board show; Hermes Agent is not involved in this item.
# serves: content (a which-file / trust verdict on a brand-pull model, riding a live thread) + hardening (a same-file re-run under two llama.cpp builds measures our own board's noise and build sensitivity).
#
# PRE-REGISTERED (nerf90a/code/PREREG.md, sha256-frozen BEFORE any counted output in FROZEN.sha256, checked in the PLAN phase; nerf_summary.py applies the rule, nothing is worded by hand):
#   task MOVED = outside the day-0 Wilson band AND p<0.01 (exact McNemar on per-item pairs for GPQA/HumanEval, two-proportion z for ARC/MMLU). headline pass = NEW build + Q6_K.
#   verdicts: INCOMPLETE (a task short of the day-0 total: exit 3, resumes) | CHANGED FILE (sha256 != recorded oid) | SHIFTED DOWN / UP / MIXED (>=1 task MOVED; attribution "runtime build, not the weights" only when the OLD-build pass is itself inside the band) | SAME FILE, SAME SCORES.
#   secondary passes (OLD Q6_K, NEW/OLD Q4_K_M) are reported with their own words and never gate. a null is valid and is the expected result.
# POSTABLE ARTEFACT (planned here, CPU only on capsule after the card is released): run/nerf90a/out/nerf-audit.mp4 (16:9 1920x1080 ~68 s: result card 8 s FIRST, then the hash fingerprint typed against August's oid, the 362-answer replay with a running same-answer counter,
#   the band picture, ends on the card; motion from frame 1; no paths/hostnames/question text on screen, a leak regex guards every drawn string) + out/contact-*.png + out/readout.md + summary.json. mock-render proof: PLAN phase renders a MOCK-stamped clip at 0.1x time.
# HONEST LIMITS: greedy think-off, one run per pass, one rig; n=198 and 164 on the paired tasks (a GPQA item is ~0.5 pt); MMLU is a 19-subject slice; GSM8K, HellaSwag and 38 MMLU subjects are not re-run (they do not fit 75 min of card);
#   OLD build identity is inferred (UNVERIFIED); the day-0 harness tree is the 17 Aug commit, not recorded in the 15 Aug meta.json (UNVERIFIED); community GGUFs of the open weights; hosted-API claims out of scope.
# CARD TIME (honest). GPU SMOKE 2026-11-06 11:40-11:45 CET (card free, no drain, flock'd, never counted, slug nerf90a-smoke): all four passes x the first 20 items of each of the four tasks (320 requests): load 14-20 s from page cache, template smoke OK on all four,
#   mean s/item NEW Q6_K gpqa 0.264 he 1.19 arc 0.192 mmlu 0.156 | OLD Q6_K 0.275 1.307 0.242 0.21 | NEW Q4_K_M 0.304 0.93 0.214 0.188 | OLD Q4_K_M 0.35 0.954 0.231 0.179. extrapolated to the full item counts (198/164/1,172/2,614):
#   NEW Q6_K ~14.7 min, OLD Q6_K ~18.4, NEW Q4_K_M ~15.9, OLD Q4_K_M ~16.1 => ~65 min + 4 loads (~1-2 min cold) + drain/restore ~3 min = ~70 min; plan 75. HumanEval is the noisy term (20 items is 12% of 164). The day-0 per-task elapsed (MMLU 1,028 s for 7,010 items) agrees within 10%.
#   hashing the two GGUFs (~40 GB read) takes ~50 s from page cache at staging (cold NVMe ~2-3 min): it runs BEFORE the drain, so it costs no card time.
#   CPU after the card is released: summary + render ~30 s (stub-run proof: PLAN-to-DONE 75 s incl. render at full length).
# DISK: 0 GB new at run time (the 17.1 GB Q4_K_M at ~/models/qwen38-27b-day0/ was fetched at staging, by commit id, sha256-verified); runs/results < 5 MB.
# WINDOW RULE: needs >= 25 min left in the block at start or exits 3 without draining Donald; a pass is skipped when < 18 min remain (resumable per item, exit 3 at the end); run_legs starts no new item in the last 4 min.
#   SORT: `90a-` sorts first among the 9x bench items; the headline pass (NEW Q6_K) runs first, then OLD Q6_K, then the Q4_K_M pair.
#
# test knobs (empty in production): NERF_SMOKE=1 (slug -smoke, first NERF_LIMIT=8 items per task, NERF_PASSES default "new:q6k", no window rule, no summary/render), NERF_NO_DRAIN=1 (smoke only, card must show < 8 GB used),
#   NERF_STUB=http://127.0.0.1:PORT (CPU mock: no drain, no server start, facts flagged mock, render stamped MOCK), NERF_NO_SELFTEST=1, NERF_NO_SHA=1 (DRYRUN only: skip hashing), MANIFEST_OVERRIDE (staging DRYRUN only).
set -uo pipefail
NL_TAG=nerf90a
MANIFEST=/home/witcheer/nerf90a-status.json
[ -n "${MANIFEST_OVERRIDE:-}" ] && MANIFEST=$MANIFEST_OVERRIDE
export MANIFEST
. /home/witcheer/night-lib.sh
NA=/home/witcheer/voxel-stage/nerf90a
CODE=$NA/code
BPY=/home/witcheer/benchmark-rig/venv/bin/python           # httpx + datasets (the harness python)
RPY=/home/witcheer/scratch/race/venv/bin/python            # PIL (renderer)
FF=/home/witcheer/voxelbench/ffmpeg
SRV_OLD=/home/witcheer/llama.cpp-nl/build/bin/llama-server; OLD_SHA=5d16e81dd
SRV_NEW=/home/witcheer/llama.cpp-donald/build/bin/llama-server; NEW_SHA=839a572a4
G_Q6=/home/witcheer/models/qwen38-27b/Qwen3.8-27B-Q6_K.gguf; B_Q6=22884408288
G_Q4=/home/witcheer/models/qwen38-27b-day0/Qwen3.8-27B-Q4_K_M.gguf; B_Q4=17106775008
PORT=8276; STUBPORT=8277
SMOKE=${NERF_SMOKE:-0}; NODRAIN=${NERF_NO_DRAIN:-0}; STUB=${NERF_STUB:-}
LIMIT=0; [ "$SMOKE" = 1 ] && LIMIT=${NERF_LIMIT:-8}
PASSES=${NERF_PASSES:-"new:q6k old:q6k new:q4km old:q4km"}; [ "$SMOKE" = 1 ] && PASSES=${NERF_PASSES:-"new:q6k"}
SLUG=nerf90a; [ "$SMOKE" = 1 ] && SLUG=$SLUG-smoke; [ -n "$STUB" ] && SLUG=$SLUG-stub
RUN=$NA/run/$SLUG; RES=$RUN/results; OUT=$RUN/out; ART=$RUN/summary.json
PASS_MIN=18
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
gg() { [ "$1" = q6k ] && echo "$G_Q6" || echo "$G_Q4"; }
sb() { [ "$1" = old ] && echo "$SRV_OLD" || echo "$SRV_NEW"; }

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0} smoke=$SMOKE limit=$LIMIT passes=[$PASSES] nodrain=$NODRAIN stub=${STUB:-none})"
nl_check_path "$SRV_OLD" "$SRV_NEW" "$G_Q6" "$G_Q4" "$BPY" "$RPY" "$FF" "$CODE/nerf_common.py" "$CODE/run_legs.py" "$CODE/nerf_summary.py" "$CODE/render_nerf.py" "$CODE/facts.py" "$CODE/make_mock_results.py" "$CODE/stub_server.py" \
  "$CODE/PREREG.md" "$CODE/FROZEN.sha256" "$CODE/nerfevals/gpqa.py" "$NA/day0/q6k/gpqa_progress.json" "$NA/day0/q4km/gpqa_progress.json" /home/witcheer/night-lib.sh
[ "$(stat -c %s "$G_Q6" 2>/dev/null)" = "$B_Q6" ] && nl_ok "Q6_K size $B_Q6" || nl_fail "Q6_K size != $B_Q6"
[ "$(stat -c %s "$G_Q4" 2>/dev/null)" = "$B_Q4" ] && nl_ok "Q4_K_M size $B_Q4" || nl_fail "Q4_K_M size != $B_Q4"
nl_check_disk 120
# frozen files (prereg, harness copy, item sets, runner, verdict rule, day-0 comparators) must be byte-identical to what was registered
FZ=$(cd "$CODE" && sha256sum -c FROZEN.sha256 2>&1 | grep -v ': OK$' || true)
[ -z "$FZ" ] && nl_ok "FROZEN.sha256: every frozen file matches" || nl_fail "frozen file changed: $FZ"
V_OLD=$($SRV_OLD --version 2>&1); V_NEW=$($SRV_NEW --version 2>&1)
case "$V_OLD" in *"$OLD_SHA"*) nl_ok "OLD build $OLD_SHA";; *) nl_fail "OLD llama.cpp build changed (expected $OLD_SHA)";; esac
case "$V_NEW" in *"$NEW_SHA"*) nl_ok "NEW build $NEW_SHA";; *) nl_fail "NEW llama.cpp build changed (expected $NEW_SHA)";; esac
for B in "$SRV_OLD" "$SRV_NEW"; do HELPL=$($B --help 2>&1 || true); for f in --jinja --n-gpu-layers --host --port; do case "$HELPL" in *"$f"*) ;; *) nl_fail "$(basename "$(dirname "$(dirname "$(dirname "$B")")")") llama-server lacks $f";; esac; done; done
ss -ltnH 2>/dev/null | grep -q ":$PORT " && nl_fail "port $PORT already in use"
if [ -z "$STUB" ]; then for t in nerf90a; do tmux has-session -t "=$t" 2>/dev/null && nl_fail "tmux session $t exists"; done; fi
if [ "${DRYRUN:-0}" = "1" ]; then
  # CPU proofs, no card: item sets identical to day 0, verdict branches, plumbing against a stub server, mock render, file facts, leak scan
  ID=$($BPY "$CODE/nerf_common.py" identity 2>&1 | grep -E '^IDENTITY' | head -1); case "$ID" in "IDENTITY OK") nl_ok "item sets identical to the day-0 records (GPQA expected letters, MMLU per-subject sizes, ARC/HumanEval sizes)";; *) nl_fail "item-set identity: ${ID:-no output}";; esac
  if [ -z "${NERF_NO_SELFTEST:-}" ]; then
    ST=$($BPY "$CODE/nerf_summary.py" --selftest 2>&1 | grep -E 'BAD|^selftest'); case "$ST" in *BAD*) nl_fail "verdict self-test: $ST";; *"selftest 10/10"*) nl_ok "verdict rule self-test: 9 branches + attribution OK (10/10)";; *) nl_fail "verdict self-test: $ST";; esac
    ST=$(mktemp -d /home/witcheer/voxel-stage/nerf90a/selftest.XXXXXX)
    $RPY "$CODE/stub_server.py" $STUBPORT >/dev/null 2>&1 & STP=$!
    for i in $(seq 1 20); do curl -s -m 2 localhost:$STUBPORT/health | grep -q ok && break; sleep 0.5; done
    $BPY "$CODE/run_legs.py" --port $STUBPORT --build new --file q6k --tasks gpqa,he --out "$ST/results" --limit 3 > "$ST/legs.log" 2>&1; LRC=$?
    kill $STP 2>/dev/null; wait $STP 2>/dev/null
    NG=$(wc -l < "$ST/results/new-q6k/gpqa.jsonl" 2>/dev/null || echo 0); NH=$(wc -l < "$ST/results/new-q6k/he.jsonl" 2>/dev/null || echo 0)
    [ "$LRC" = 0 ] && [ "$NG" = 3 ] && [ "$NH" = 3 ] && nl_ok "run_legs plumbing vs a stub server (gpqa $NG, humaneval $NH items, rc 0)" || nl_fail "run_legs stub proof rc=$LRC gpqa=$NG he=$NH: $(tail -2 "$ST/legs.log" | tr '\n' ' ')"
    $BPY "$CODE/make_mock_results.py" "$ST/mock" same >/dev/null 2>&1 && $BPY "$CODE/nerf_summary.py" "$ST/mock" > "$ST/mock/summary.txt" 2>&1 && grep -q '^VERDICT SAME FILE, SAME SCORES' "$ST/mock/summary.txt" && \
      $RPY "$CODE/render_nerf.py" "$ST/mock" "$ST/mock/t.mp4" --ffmpeg "$FF" --scale 0.1 > "$ST/mock/render.log" 2>&1 && [ -s "$ST/mock/t.mp4" ] && nl_ok "mock render ok ($(stat -c %s "$ST/mock/t.mp4") bytes, MOCK-stamped, 0.1x time)" || nl_fail "mock summary/render failed: $(tail -2 "$ST/mock/render.log" 2>/dev/null | tr '\n' ' ')"
    rm -rf "$ST"
  fi
  if [ -z "${NERF_NO_SHA:-}" ]; then
    FD=$(mktemp -d /home/witcheer/voxel-stage/nerf90a/factsdry.XXXXXX)
    FL=$($BPY "$CODE/facts.py" "$FD" 2>&1 | tail -1); case "$FL" in "facts ok: q6k hash_ok=True q4km hash_ok=True"*) nl_ok "$FL";; *) nl_fail "file facts: $FL";; esac
    rm -rf "$FD"
  fi
  LK=$(grep -nE '/Users/|ghp_[A-Za-z0-9]{10}|hf_[A-Za-z0-9]{20}|100\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|\.ts\.net|@[a-z0-9-]+\.[a-z]{2,}' "$0" "$CODE"/*.py "$CODE/PREREG.md" 2>/dev/null | grep -v 'users.noreply' | grep -v 'LEAK = re.compile' | head -3 || true)
  [ -z "$LK" ] && nl_ok "leak scan clean" || nl_fail "leak scan: $LK"
fi
if nl_skip_exact "$ART"; then nl_say "nothing to run, item complete ($ART exists)"; exit 0; fi
DL=$(nl_deadline_epoch); LEFT=$(( (DL-$(date +%s))/60 ))
for p in $PASSES; do b=${p%%:*}; t=${p##*:}; nl_say "  RUN pass $b-$t -> $RES/$b-$t/{gpqa,he,arc,mmlu}.jsonl (literal artefacts; resumable per item; pending now: $($BPY "$CODE/nerf_common.py" pending "$RES" "$b" "$t" "$LIMIT" 2>/dev/null | tail -1) items)"; done
nl_say "  RUN verdict -> $ART ; clip -> $OUT/nerf-audit.mp4 ; planned 4 passes (~15-18 min each), hard stop $(date -d "@$DL" '+%a %H:%M') local, $LEFT min away"
if [ "$SMOKE" != 1 ] && [ -z "$STUB" ]; then
  [ "$LEFT" -ge 25 ] || { nl_say "  under 25 min left in the window: not starting (drain + load + a pass would be cut short; exit 3, resumable, Donald untouched)"; exit 3; }
fi
if [ "${DRYRUN:-0}" = "1" ] || [ ! -f "$MANIFEST" ]; then nl_manifest_init facts pass_new_q6k pass_old_q6k pass_new_q4km pass_old_q4km summary render; fi
nl_gate

# ---- phase 2 (real run) ----
mem_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
wait_mem_free() { local i v; for i in $(seq 1 90); do v=$(mem_used); [ "${v:-99999}" -lt 1024 ] && { nl_say "  card clear (${v} MiB)"; return 0; }; sleep 2; done; nl_say "  card NOT clear after 180 s (${v} MiB)"; return 1; }
SP=""; DONALD_STOPPED=0; DONALD_BACK=0
stop_srv() { [ -n "$SP" ] || return 0; kill -TERM "$SP" 2>/dev/null; for i in $(seq 1 30); do kill -0 "$SP" 2>/dev/null || break; sleep 2; done; kill -KILL "$SP" 2>/dev/null; sleep 3; SP=""; }
donald_restore() {
  [ "$DONALD_STOPPED" = 1 ] && [ "$DONALD_BACK" = 0 ] || return 0
  DONALD_BACK=1; nl_say "restoring Donald (sudo -n systemctl start llama-server.service)"
  sudo -n systemctl start llama-server.service
  for i in $(seq 1 60); do H=$(curl -s -m 5 localhost:8090/health); [[ "${H,,}" == *ok* ]] && { nl_say "Donald healthy again after ~$((i*5)) s"; return 0; }; sleep 5; done
  nl_say "WARNING: Donald /health did not answer within 5 min after the restore"
}
cleanup() { stop_srv; [ "$DONALD_STOPPED" = 1 ] && { wait_mem_free || true; }; donald_restore; }
trap cleanup EXIT
mkdir -p "$RES" "$OUT"
fail() { nl_mark "$1" failed "$2"; nl_say "FAIL: $2"; exit 1; }
DEADLINE=$(nl_deadline_epoch)
REMAIN=0
t0=$(date +%s)

# facts first, on CPU, BEFORE the card is touched: the two sha256 (page cache or NVMe), the HF oids, the build lines. mock-flagged under a stub.
nl_mark facts running
if [ -n "$STUB" ] || [ -n "${NERF_NO_SHA:-}" ]; then $BPY "$CODE/facts.py" "$RUN" --no-sha > "$RUN/facts.log" 2>&1; $BPY - "$RUN/facts.json" <<'PYEOF'
import json, sys
p = sys.argv[1]; d = json.load(open(p)); d["mock"] = True; json.dump(d, open(p, "w"), indent=1)
PYEOF
else $BPY "$CODE/facts.py" "$RUN" > "$RUN/facts.log" 2>&1; fi
FL=$(tail -1 "$RUN/facts.log"); nl_say "  $FL"
[ -s "$RUN/facts.json" ] || fail facts "facts.json not written: $FL"
nl_mark facts done "$FL"

if [ -n "$STUB" ]; then nl_say "STUB mode: no Donald drain, no server start (upstream $STUB)"
elif [ "$SMOKE" = 1 ] && [ "$NODRAIN" = 1 ]; then
  U=$(mem_used); [ "${U:-99999}" -lt 8000 ] || fail pass_new_q6k "smoke without drain needs the card under 8 GB used (now ${U} MiB)"
  nl_say "smoke without drain: card ${U} MiB used"
else
  FOREIGN=$(nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ' | tr '\n' ' ')
  nl_say "  rule 3, live: compute apps before the drain: ${FOREIGN:-none}; tmux: $(tmux ls -F '#{session_name}' 2>/dev/null | tr '\n' ' ')"
  nl_say "draining Donald (night-lib nl_donald_stop)"; DONALD_STOPPED=1; nl_donald_stop
  for i in $(seq 1 60); do v=$(mem_used); [ "${v:-99999}" -lt 1024 ] && { nl_say "drain clear (${v} MiB)"; break; }; sleep 2; done
  [ "${v:-99999}" -lt 1024 ] || fail pass_new_q6k "card memory still ${v:-?} MiB after the drain: another process owns the card, not loading"
fi

serve_pass() { # <build> <tag>: sets SP; writes fit-<build>-<tag>.json; returns 1 when the pass is unusable
  local b=$1 t=$2 t1=$(date +%s) i H SM
  "$(sb "$b")" -m "$(gg "$t")" --port $PORT --host 127.0.0.1 --n-gpu-layers 99 --jinja >> "$RUN/server-$b-$t.log" 2>&1 &
  SP=$!
  for i in $(seq 1 240); do
    H=$(curl -s -m 3 http://127.0.0.1:$PORT/health); [[ "${H,,}" == *ok* ]] && { LSTART=$(( $(date +%s)-t1 )); nl_say "  llama-server ($b $t) up after ${LSTART} s (VRAM $(mem_used) MiB)"; break; }
    kill -0 "$SP" 2>/dev/null || break; sleep 2; done
  if ! [[ "${H:-}" == *ok* || "${H,,}" == *ok* ]]; then echo '{"loaded":false,"why":"did not come up"}' > "$RUN/fit-$b-$t.json"; stop_srv; return 1; fi
  # template smoke: think-off must come back short and right (the board runs think-off through chat_template_kwargs)
  SM=$($BPY - <<PYEOF
import json, urllib.request
b = "http://127.0.0.1:$PORT"
body = {"model": "nerf", "messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number only."}], "max_tokens": 300, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False, "reasoning": False, "reasoning_effort": "none", "reasoning_strength": "low"}}
r = json.loads(urllib.request.urlopen(urllib.request.Request(b + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=120).read())
m = r["choices"][0]["message"]; tok = r["usage"]["completion_tokens"]
p = json.loads(urllib.request.urlopen(b + "/props", timeout=10).read())
ok = "391" in (m.get("content") or "") and tok < 120 and not (m.get("reasoning_content") or "").strip()
print(("SMOKE OK" if ok else "SMOKE FAIL"), "completion_tokens", tok, "n_ctx", (p.get("default_generation_settings") or {}).get("n_ctx"), "slots", p.get("total_slots"))
PYEOF
  ); SM=$(echo "$SM" | tail -1); nl_say "  template smoke ($b $t): $SM"
  case "$SM" in
    "SMOKE OK"*) echo "{\"loaded\":true,\"start_s\":${LSTART:-0},\"vram_mib\":$(mem_used),\"smoke\":\"$(echo "$SM" | cut -c1-100)\"}" > "$RUN/fit-$b-$t.json";;
    *) echo '{"loaded":false,"why":"template smoke failed"}' > "$RUN/fit-$b-$t.json"; stop_srv; return 1;;
  esac
  return 0
}

run_pass() { # <build> <tag>
  local b=$1 t=$2 ph="pass_${1}_${2}" lft pend rc
  pend=$($BPY "$CODE/nerf_common.py" pending "$RES" "$b" "$t" "$LIMIT" 2>/dev/null | tail -1)
  if [ "${pend:-1}" = 0 ]; then nl_say "  pass $b-$t: already complete (resume), kept"; nl_mark "$ph" done "already complete"; return 0; fi
  lft=$(( (DEADLINE-$(date +%s))/60 ))
  if [ "$SMOKE" != 1 ] && [ -z "$STUB" ] && [ "$lft" -lt "$PASS_MIN" ]; then nl_say "  pass $b-$t skipped: $lft min left < $PASS_MIN (resumable)"; nl_mark "$ph" skipped "window"; REMAIN=1; return 0; fi
  nl_mark "$ph" running
  if [ -n "$STUB" ]; then PORT_USE=${STUB##*:}; else
    serve_pass "$b" "$t" || { nl_say "  pass $b-$t BROKEN: did not load or template smoke failed"; nl_mark "$ph" failed "load/smoke"; REMAIN=1; return 0; }
    PORT_USE=$PORT
  fi
  local lim=(); [ "$LIMIT" != 0 ] && lim=(--limit "$LIMIT")
  $BPY "$CODE/run_legs.py" --port "$PORT_USE" --build "$b" --file "$t" --out "$RES" --deadline "$DEADLINE" "${lim[@]}" >> "$RUN/legs-$b-$t.log" 2>&1; rc=$?
  nl_say "  pass $b-$t run_legs rc=$rc: $(grep -E '^\[run_legs\] leg end .* all tasks' "$RUN/legs-$b-$t.log" | tail -1 | cut -c1-160)"
  [ "$rc" = 3 ] && REMAIN=1
  [ "$rc" = 1 ] && { nl_mark "$ph" failed "run_legs fault: $(tail -1 "$RUN/legs-$b-$t.log" | cut -c1-120)"; REMAIN=1; }
  [ -z "$STUB" ] && { curl -s -m 5 "http://127.0.0.1:$PORT/metrics" > "$RUN/metrics-end-$b-$t.txt" 2>/dev/null; stop_srv; wait_mem_free || fail "$ph" "card not clear after the $b-$t stop"; }
  [ "$rc" = 0 ] && nl_mark "$ph" done "all four tasks"
  return 0
}
for p in $PASSES; do run_pass "${p%%:*}" "${p##*:}"; done
# the card is free from here: Donald back, then the CPU-only legs
stop_srv; [ -n "$STUB" ] || wait_mem_free || true; donald_restore
nl_say "card released after $(( ($(date +%s)-t0)/60 )) min (drain to restore, incl. the CPU hashing), the rest is CPU"
if [ "$SMOKE" = 1 ]; then
  nl_say "SMOKE done: per-item results in $RES; no summary/render in smoke"
  for p in $PASSES; do for tk in gpqa he arc mmlu; do nl_say "  ${p/:/-} $tk: $(wc -l < "$RES/${p/:/-}/$tk.jsonl" 2>/dev/null || echo 0) items, $(grep -c '"ok": true' "$RES/${p/:/-}/$tk.jsonl" 2>/dev/null || echo 0) correct, mean s/item $($BPY -c "import json,sys;r=[json.loads(l) for l in open('$RES/${p/:/-}/$tk.jsonl')];print(round(sum(x['s'] for x in r)/max(len(r),1),3))" 2>/dev/null)"; done; done
  exit 0
fi
nl_mark summary running
$BPY "$CODE/nerf_summary.py" "$RUN" > "$OUT/summary.txt" 2>&1; src=$?
nl_say "  summary rc=$src: $(grep -m1 '^VERDICT' "$OUT/summary.txt" | cut -c1-200)"
for p in $PASSES; do b=${p%%:*}; t=${p##*:}; [ "$($BPY "$CODE/nerf_common.py" pending "$RES" "$b" "$t" 0 2>/dev/null | tail -1)" = 0 ] || REMAIN=1; done
if [ "$src" = 0 ] && [ "$REMAIN" = 0 ] && [ -f "$ART" ]; then
  nl_mark summary done "$(grep -m1 '^VERDICT' "$OUT/summary.txt" | cut -c9-60)"
  grep -m1 '^LINE1' "$OUT/summary.txt" | cut -c7- > "$OUT/line1.txt"
  { echo "# nerf audit readout"; echo; grep -E '^(VERDICT|ATTRIBUTION|SECONDARY)' "$OUT/summary.txt"; echo; grep '^SUMMARY' "$OUT/summary.txt"; echo; echo "regime: Q6_K (headline) and Q4_K_M, builds OLD $OLD_SHA / NEW $NEW_SHA, think-off, temperature 0, day-0 harness copy (benchmark-rig lib/evals @ 6c7ecd8), items = the day-0 items"; } > "$OUT/readout.md"
  if [ -z "${NERF_NO_RENDER:-}" ]; then
    nl_mark render running
    $RPY "$CODE/render_nerf.py" "$RUN" "$OUT/nerf-audit.mp4" --ffmpeg "$FF" --stills "$OUT/stills" >> "$OUT/render.log" 2>&1; rr=$?
    $FF -nostdin -hide_banner -loglevel error -y -ss 0.5 -i "$OUT/nerf-audit.mp4" -frames:v 1 "$OUT/contact-1.png" </dev/null
    $FF -nostdin -hide_banner -loglevel error -y -ss 40 -i "$OUT/nerf-audit.mp4" -frames:v 1 "$OUT/contact-2.png" </dev/null
    nl_judge_artifact render "$rr" "$OUT/nerf-audit.mp4" "$t0" "clip" || nl_say "  render failed (see $OUT/render.log); the verdict stands"
  fi
  nl_say "ANGLE: $(cat "$OUT/line1.txt")"
  nl_say "DONE $SLUG"
  exit 0
fi
nl_say "  a pass could not finish inside the window: exit 3, resumable, summary.json NOT written (PROVISIONAL: $RUN/summary.pre.json)"
exit 3
