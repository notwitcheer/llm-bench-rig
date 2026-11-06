#!/bin/bash
# 90-astra-stop-q4.sh: idle-queue item (STAGED in ~/voxel-stage, 2026-11-05, esoteric lane).
# WHAT: replay of the Astra-Stop weight edit on our card. HF jarrodwatts/Qwen3.8-27B-Astra-Stop (2026-11-04, apache-2.0): ONE rank edit, W' = W - alpha*outer(u, u@W), alpha -1, to
#   layers.63.mlp.down_proj of Qwen/Qwen3.8-27B (base rev 1d4bf0f). Card claim (reasoning effort xhigh, bf16, 16 tries per side): mean thinking tokens on "pelican on a bicycle" 28,474 -> 17,201 (-40%),
#   median 27,346 -> 17,156, 16/16 complete SVGs both; 694 maths/code problems -3% tokens, accuracy unchanged 95.2%. bf16 (55.6 GB) does not fit 32 GB, so BOTH arms were built with the identical recipe
#   (author's apply.py -> convert_hf_to_gguf.py bf16 -> llama-quantize Q4_K_M, llama.cpp fac3a894f) and are run here: unedited vs edited, same sampler, same 32 seeds, --parallel 4.
#   The edit's effect after quantising is verified on the files (blk.63.ffn_down u^T W norm x1.999), not assumed. "does not survive Q4" (no difference) is a valid, pre-registered result.
# MOJO: two pelicans on bicycles drawn side by side, a huge thinking-token counter per side, the edited one finishing first, then a gallery of 16 pelicans per arm | ahead (a 3-day-old weight-space edit replayed on our own card, checked not claimed as first) + funny (the pelican) | on a Q4 pair on one 5090, [the 20 KB edit made Qwen3.8-27B think N% less on the pelican prompt: median A vs B tokens, p=P, n=32 per arm]
# EV: private (projected post angle and reach bracket live in the ops vault status note)
#   (the news wave closes ~48 h after 11-04), 1-3k if NO DIFFERENCE (a null with a good picture), UNVERIFIED. call: X if the render reads at phone width and the verdict is not BROKEN/INVALID; else TG + repo.
#   RAILS: never "an LLM read the weights" (the discovery was the author's Codex run, we replay the effect); never "first" (three repos replicated the MIT experiment 8); say Q4 vs bf16, sampler ours, n.
# serves: experiment (a method question: does a one-matrix stop-signal edit survive Q4_K_M and shorten thinking) + content (esoteric lane, release-wave replay).
#
# PRE-REGISTERED (astra-stop/PREREG.md, sha256-frozen BEFORE any counted output in FROZEN.sha256, checked in the PLAN phase; astra_summary.py applies the rule, nothing is worded by hand):
#   headline = thinking tokens per try (the model's reasoning_content re-tokenised with its own tokenizer), edited vs unedited, two-sided Mann-Whitney, medians + bootstrap CI. SVG complete = finish_reason stop AND
#   closed <svg> that parses (lxml) AND renders non-blank (resvg). verdicts: BROKEN (edited arm failed to load/template smoke, or SVG completeness > 25 points under the unedited arm) | INVALID (unedited arm failed) |
#   REVERSED (p<0.05, edited thinks LONGER) | REPRODUCED (p<0.05 and median down >= 25%) | PARTIAL (p<0.05, median down < 25%) | NO DIFFERENCE (otherwise) | INCOMPLETE (< 32 tries on an arm: exit 3, resumes).
#   maths slice (24 fixed MATH-500 problems, string-match answer) is REPORTED ONLY, never gates a verdict.
# POSTABLE ARTEFACT (planned here, CPU only on capsule after the run): run/astra90/out/astra-stop.mp4 (16:9 1920x1080 ~65 s: five paired pelicans drawn side by side with token counters, the 4x4 galleries, the result card;
#   motion from frame 1, counters 150 px, text as captions) + out/contact.png + out/readout.md + summary.json. The arms ran one after the other on one card: the video says so; the playback speed is shared and is NOT a speed claim.
# HONEST LIMITS: Q4_K_M pair (the author tested bf16); KV q8_0 (card fit: 4 slots x 65,536 ctx); sampler is Qwen's thinking-mode default (the card states none, UNVERIFIED); one prompt, one author's wording ("Generate an SVG of a pelican riding a bicycle",
#   UNVERIFIED: the card does not print the exact prompt); n=32 per arm; thinking tokens counted by our tokenizer, not the author's harness; a try that hits the 60,000-token cap is counted at its length and flagged capped.
# CARD TIME (honest). smoke 2026-11-05 (card free, no drain): load 8-10 s from page cache (26.2 GB VRAM at 4 x 65,536 q8_0; cold ~60-90 s from NVMe), template smoke ok on both arms (xhigh renders 65 prompt tokens vs 53 for low),
#   4 slots decoding together at 43.4 tok/s EACH (~174 tok/s aggregate, near-empty context; single-stream elsewhere is 78 tok/s), a 4-try half-leg capped at 4,000 tokens took 92 s. maths smoke: 2 problems 4.8 s and 42.5 s.
#   ESTIMATE (UNVERIFIED until a real leg): tokens = 32 x 28,474 + 32 x 17,201 = 1.46 M at the card's own means; /174 tok/s = 140 min, +15-30% for decode slowing with a 20-30k context and slot tails = 160-185 min,
#   + 4 loads/stops ~6 min + maths slice ~10 min + drain/restore ~3 min + CPU render ~2 min => ~3 h of card. Expect the first block to hold the first one or two half-legs after 88 and 89 (they take 60-120 min of the 300), the rest spills to the next block (resumable per try). If the edit works, the edited half-legs are ~40% shorter.
# WINDOW RULE: needs >= 60 min left in the block at start (exit 3 before draining Donald otherwise, resumable); each of the four pelican half-legs (arm x 16 seeds) needs LEG_MIN min left or is skipped (exit 3 at the end);
#   no new try starts in the last 15 min and in-flight tries are dropped (not saved, re-run next block) in the last 5 min. resumable PER try (results/<arm>/pelican-<seed>.json). SORT: `90-` sorts after 89-ripple.
#
# test knobs (empty in production): ASTRA_SMOKE=1 (slug -smoke, no window rule, no summary/render, seeds ASTRA_SEEDS, cap ASTRA_MAXTOK, legs ASTRA_ARMS), ASTRA_NO_DRAIN=1 (smoke only, card must show < 8 GB used),
#   ASTRA_MATH=0 (skip the maths slice), ASTRA_NO_RENDER=1, MANIFEST_OVERRIDE (staging DRYRUN only), ASTRA_NL_NOW (unused).
MANIFEST=/home/witcheer/astra-stop-status.json
set -uo pipefail
NL_TAG=astra90
[ -n "${MANIFEST_OVERRIDE:-}" ] && MANIFEST=$MANIFEST_OVERRIDE
export MANIFEST
. /home/witcheer/night-lib.sh
AS=/home/witcheer/astra-stop
CODE=$AS/code
SRV=/home/witcheer/llama.cpp-qwen4/build/bin/llama-server
LL_SHA=fac3a894f
PORT=8265
CTX=262144; NP=4; KV=q8_0
RPY=$AS/rvenv/bin/python
FF=/home/witcheer/voxelbench/ffmpeg
SMOKE=${ASTRA_SMOKE:-0}; NODRAIN=${ASTRA_NO_DRAIN:-0}
SLUG=astra90; [ "$SMOKE" = 1 ] && SLUG=$SLUG-smoke
RUN=$AS/run/$SLUG; RES=$RUN/results; OUT=$RUN/out; ART=$RUN/summary.json
LEG_MIN=${ASTRA_LEG_MIN:-45}
MAXTOK=${ASTRA_MAXTOK:-60000}
ARMS=${ASTRA_ARMS:-base,edited}
G_BASE=$AS/gguf/base-Q4_K_M.gguf; G_EDIT=$AS/gguf/edited-Q4_K_M.gguf
B_BYTES=16547400256
SHA_BASE=9297d44ba5960f893ed122bb8aaf20b2138eec40613b0d4a1fc99cfc6beffe05
SHA_EDIT=c4c7acdc716ea09c4c4ed448855ec7a7d75030ac368349c9bc61f0d21d56e8e3
if [ "$SMOKE" = 1 ]; then SEEDS_A=${ASTRA_SEEDS:-1,2,3,4}; SEEDS_B=""; else SEEDS_A=$(seq -s, 1 16); SEEDS_B=$(seq -s, 17 32); fi
MATH_IDS=$($RPY -c "import sys; sys.path.insert(0,'$CODE'); from astra_common import math_ids; print(','.join(map(str,math_ids())))" 2>/dev/null || echo "")
MATH_FILE=$(ls /home/witcheer/.cache/huggingface/hub/datasets--HuggingFaceH4--MATH-500/snapshots/*/test.jsonl 2>/dev/null | head -1)

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0} smoke=$SMOKE arms=$ARMS nodrain=$NODRAIN leg_min=$LEG_MIN maxtok=$MAXTOK)"
nl_check_path "$SRV" "$G_BASE" "$G_EDIT" "$RPY" "$FF" "$CODE/astra_common.py" "$CODE/run_legs.py" "$CODE/astra_summary.py" "$CODE/render_astra.py" "$CODE/PREREG.md" "$CODE/FROZEN.sha256" "$MATH_FILE" /home/witcheer/night-lib.sh
nl_check_disk 120
for G in "$G_BASE" "$G_EDIT"; do
  SZ=$(stat -c %s "$G" 2>/dev/null || echo 0); [ "$SZ" = "$B_BYTES" ] && nl_ok "$(basename "$G") $SZ bytes" || nl_fail "$G size $SZ != $B_BYTES"
done
# frozen files (prereg, code, prompt) must be byte-identical to what was registered
FZ=$(cd "$CODE" && sha256sum -c FROZEN.sha256 2>&1 | grep -v ': OK$' || true)
[ -z "$FZ" ] && nl_ok "FROZEN.sha256: every frozen file matches" || nl_fail "frozen file changed: $FZ"
case "$($SRV --version 2>&1)" in *"$LL_SHA"*) nl_ok "llama.cpp build $LL_SHA";; *) nl_fail "llama.cpp build changed (expected $LL_SHA)";; esac
HELPL=$($SRV --help 2>&1 || true)
for f in --jinja --parallel --ctx-size --alias --n-gpu-layers --flash-attn --cache-type-k --cache-type-v --cache-ram --reasoning-format; do case "$HELPL" in *"$f"*) ;; *) nl_fail "llama-server flag $f not in this build";; esac; done
[ -n "$MATH_IDS" ] && nl_ok "maths ids: $(echo "$MATH_IDS" | cut -c1-60)..." || nl_fail "maths ids not computed"
ss -ltn 2>/dev/null | grep -q ":$PORT " && nl_fail "port $PORT already in use"
if [ "${DRYRUN:-0}" = "1" ]; then
  # CPU-only integrity proof of the two quantised files against the sha256 recorded when they were built (hash from page cache or NVMe, no card)
  for pair in "$G_BASE:$SHA_BASE" "$G_EDIT:$SHA_EDIT"; do
    H=$(sha256sum "${pair%%:*}" | cut -d' ' -f1); [ "$H" = "${pair##*:}" ] && nl_ok "sha256 $(basename "${pair%%:*}") ${H:0:12}" || nl_fail "sha256 of ${pair%%:*} is ${H:0:12}, expected ${pair##*:}"
  done
  # leak scan: this item + prereg + code must not carry paths, tokens or tailnet addresses
  LK=$(grep -nE '/Users/|ghp_[A-Za-z0-9]{10}|hf_[A-Za-z0-9]{20}|100\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|\.ts\.net|@[a-z0-9-]+\.[a-z]{2,}' "$0" "$CODE"/*.py "$CODE/PREREG.md" 2>/dev/null | grep -v 'users.noreply' | head -3 || true)
  [ -z "$LK" ] && nl_ok "leak scan clean" || nl_fail "leak scan: $LK"
  # CPU proofs of the scoring + render paths on mock data (never counted; MOCK marker puts a banner on every frame)
  if [ -z "${ASTRA_NO_SELFTEST:-}" ]; then
    ST=$(mktemp -d /home/witcheer/astra-stop/selftest.XXXXXX); okc=0
    for m in repro:REPRODUCED nodiff:"NO DIFFERENCE" partial:PARTIAL broken:BROKEN; do
      mkdir -p "$ST/${m%%:*}"; $RPY "$CODE/make_mock_results.py" "$ST/${m%%:*}" "${m%%:*}" >/dev/null 2>&1
      V=$($RPY "$CODE/astra_summary.py" "$ST/${m%%:*}" 2>&1 | grep -m1 '^VERDICT' | cut -d' ' -f2- | cut -d'|' -f1 | sed 's/ *$//')
      [ "$V" = "${m#*:}" ] && okc=$((okc+1)) || nl_fail "selftest ${m%%:*}: got '$V' want '${m#*:}'"
    done
    nl_say "  selftest: $okc/4 verdict branches"
    $RPY "$CODE/render_astra.py" "$ST/repro" "$ST/repro/t.mp4" --rounds 1 --ffmpeg "$FF" >/dev/null 2>&1 && [ -s "$ST/repro/t.mp4" ] && nl_ok "mock render ok ($(stat -c %s "$ST/repro/t.mp4") bytes)" || nl_fail "mock render failed"
    rm -rf "$ST"
  fi
fi
if nl_skip_exact "$ART"; then nl_say "nothing to run, item complete ($ART exists)"; exit 0; fi
DL=$(nl_deadline_epoch); LEFT=$(( (DL-$(date +%s))/60 ))
for arm in ${ARMS//,/ }; do for h in A B; do [ "$h" = B ] && [ -z "$SEEDS_B" ] && continue; nl_say "  RUN leg $arm-$h -> $RES/$arm/pelican-<seed>.json (literal artefacts)"; done; done
nl_say "  RUN verdict -> $ART ; clip -> $OUT/astra-stop.mp4 ; planned 4 pelican half-legs (+ maths when time allows), hard stop $(date -d "@$DL" '+%a %H:%M') local, $LEFT min away"
if [ "$SMOKE" != 1 ]; then
  [ "$LEFT" -ge 60 ] || { nl_say "  under 60 min left in the window: not starting (drain + load + a half-leg would be cut short; exit 3, resumable, Donald untouched)"; exit 3; }
fi
if [ "${DRYRUN:-0}" = "1" ] || [ ! -f "$MANIFEST" ]; then nl_manifest_init legA_base legA_edited legB_base legB_edited summary render; fi
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
REMAIN=0                      # set to 1 when a leg could not finish

if [ "$SMOKE" = 1 ] && [ "$NODRAIN" = 1 ]; then
  U=$(mem_used); [ "${U:-99999}" -lt 8000 ] || fail legA_base "smoke without drain needs the card under 8 GB used (now ${U} MiB)"
  nl_say "smoke without drain: card ${U} MiB used"
else
  FOREIGN=$(nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ' | tr '\n' ' ')
  nl_say "  rule 3, live: compute apps before the drain: ${FOREIGN:-none}; tmux: $(tmux ls -F '#{session_name}' 2>/dev/null | tr '\n' ' ')"
  nl_say "draining Donald (night-lib nl_donald_stop)"; DONALD_STOPPED=1; nl_donald_stop
  for i in $(seq 1 60); do v=$(mem_used); [ "${v:-99999}" -lt 1024 ] && { nl_say "drain clear (${v} MiB)"; break; }; sleep 2; done
  [ "${v:-99999}" -lt 1024 ] || fail legA_base "card memory still ${v:-?} MiB after the drain: another process owns the card, not loading"
fi

serve_arm() { # <arm> <gguf>: sets SP; writes fit-<arm>.json; returns 1 when the arm is unusable
  local arm=$1 g=$2 t1=$(date +%s) i
  "$SRV" -m "$g" --port $PORT --host 127.0.0.1 -ngl 99 -fa on --jinja -c $CTX -np $NP -ctk $KV -ctv $KV -cram 0 --reasoning-format deepseek --alias astra >> "$RUN/server-$arm.log" 2>&1 &
  SP=$!
  for i in $(seq 1 240); do
    H=$(curl -s -m 3 http://127.0.0.1:$PORT/health); [[ "${H,,}" == *ok* ]] && { LSTART=$(( $(date +%s)-t1 )); nl_say "  llama-server ($arm) up after ${LSTART} s (VRAM $(mem_used) MiB)"; break; }
    kill -0 "$SP" 2>/dev/null || break; sleep 2; done
  if ! [[ "${H,,}" == *ok* ]]; then echo '{"loaded":false,"why":"did not come up"}' > "$RES/../fit-$arm.json"; stop_srv; return 1; fi
  # template smoke: the kwargs must reach the template (xhigh vs low change the prompt) and the model must answer with reasoning_content
  SM=$($RPY - <<PYEOF
import sys, json; sys.path.insert(0, "$CODE")
from astra_common import *
b = "http://127.0.0.1:$PORT"
def ask(kw):
    o = chat_body("What is 17 * 23? Answer briefly.", 1, 1500, stream=False); o["chat_template_kwargs"] = kw
    return post_json(b + "/v1/chat/completions", o, 300)
x = ask({"enable_thinking": True, "reasoning_effort": "xhigh"}); l = ask({"enable_thinking": True, "reasoning_effort": "low"})
m = x["choices"][0]["message"]; px, pl = x["usage"]["prompt_tokens"], l["usage"]["prompt_tokens"]
ok = bool(m.get("reasoning_content")) and px > pl and "391" in (m.get("content") or "")
slots = json.loads(urllib.request.urlopen(b + "/slots", timeout=10).read()); nctx = slots[0].get("n_ctx") if slots else None
print(("SMOKE OK" if ok else "SMOKE FAIL"), "prompt_tokens xhigh", px, "low", pl, "think_chars", len(m.get("reasoning_content") or ""), "slots", len(slots), "n_ctx/slot", nctx)
PYEOF
  ); SM=$(echo "$SM" | tail -1); nl_say "  template smoke ($arm): $SM"
  case "$SM" in
    "SMOKE OK"*) echo "{\"loaded\":true,\"start_s\":${LSTART:-0},\"vram_mib\":$(mem_used),\"ctx\":$CTX,\"slots\":$NP,\"kv\":\"$KV\",\"smoke\":\"$(echo "$SM" | cut -c1-120)\"}" > "$RUN/fit-$arm.json";;
    *) echo "{\"loaded\":false,\"why\":\"template smoke failed\"}" > "$RUN/fit-$arm.json"; stop_srv; return 1;;
  esac
  return 0
}

run_leg() { # <half A|B> <arm> <gguf> <seeds>
  local h=$1 arm=$2 g=$3 seeds=$4 ph="leg${1}_${2}" lft
  if [ -n "${ALL_DONE_CHECK:-}" ]; then :; fi
  # nothing pending in this half-leg?
  local pend=0 s; for s in ${seeds//,/ }; do [ -f "$RES/$arm/pelican-$(printf %02d "$s").json" ] || pend=1; done
  if [ "$pend" = 0 ]; then
    # the maths slice (reported only) rides on the second half of an arm
    if [ "$h" = B ] || [ "$SMOKE" = 1 ]; then :; else nl_mark "$ph" done "already complete"; return 0; fi
  fi
  lft=$(( (DEADLINE-$(date +%s))/60 ))
  if [ "$pend" = 1 ] && [ "$SMOKE" != 1 ] && [ "$lft" -lt "$LEG_MIN" ]; then nl_say "  leg $ph skipped: $lft min left < $LEG_MIN (resumable)"; nl_mark "$ph" skipped "window"; REMAIN=1; return 0; fi
  nl_mark "$ph" running
  serve_arm "$arm" "$g" || { nl_say "  arm $arm BROKEN: did not load or template smoke failed"; nl_mark "$ph" failed "load/smoke"; return 0; }
  if [ "$pend" = 1 ]; then
    $RPY "$CODE/run_legs.py" --port $PORT --arm "$arm" --kind pelican --items "$seeds" --out "$RES" --deadline "$DEADLINE" --slots $NP --max-tokens "$MAXTOK" >> "$RUN/legs-$arm.log" 2>&1; rc=$?
    nl_say "  leg $ph run_legs rc=$rc: $(grep -E '^\[run_legs\] leg end' "$RUN/legs-$arm.log" | tail -1)"
    [ "$rc" = 3 ] && REMAIN=1
    [ "$rc" = 1 ] && { nl_mark "$ph" failed "run_legs errors"; REMAIN=1; }
  fi
  # maths slice: reported only, 24 problems, only on the second half-leg, only with >= 25 min left
  if [ "${ASTRA_MATH:-1}" = 1 ] && [ "$SMOKE" != 1 ] && [ "$h" = B ]; then
    lft=$(( (DEADLINE-$(date +%s))/60 ))
    if [ "$lft" -ge 25 ]; then
      $RPY "$CODE/run_legs.py" --port $PORT --arm "$arm" --kind maths --items "$MATH_IDS" --out "$RES" --deadline "$DEADLINE" --slots $NP --math-file "$MATH_FILE" >> "$RUN/legs-$arm.log" 2>&1
      nl_say "  maths slice ($arm): rc=$? $(ls "$RES/$arm"/maths-???.json 2>/dev/null | wc -l)/24 done (reported only)"
    else nl_say "  maths slice ($arm) skipped: $lft min left (reported only, never gates)"; fi
  fi
  curl -s -m 5 "http://127.0.0.1:$PORT/slots" > "$RUN/slots-end-$ph.json" 2>/dev/null
  stop_srv; wait_mem_free || fail "$ph" "card not clear after the $arm stop"
  nl_mark "$ph" done "$seeds"
}
gg() { [ "$1" = base ] && echo "$G_BASE" || echo "$G_EDIT"; }
for arm in ${ARMS//,/ }; do :; done
# interleaved halves so a cut block leaves a balanced partial: A(base) A(edited) B(base) B(edited)
for half in A B; do
  [ "$half" = B ] && [ -z "$SEEDS_B" ] && continue
  for arm in ${ARMS//,/ }; do
    sd=$SEEDS_A; [ "$half" = B ] && sd=$SEEDS_B
    run_leg "$half" "$arm" "$(gg "$arm")" "$sd"
  done
done
# the card is free from here: Donald back, then the CPU-only legs
stop_srv; wait_mem_free || true; donald_restore
if [ "$SMOKE" = 1 ]; then
  nl_say "SMOKE done: per-try results in $RES; no summary/render in smoke"
  for arm in ${ARMS//,/ }; do nl_say "  $arm: $(ls "$RES/$arm"/pelican-*.json 2>/dev/null | grep -vc tok.json) tries"; done
  exit 0
fi
nl_mark summary running
$RPY "$CODE/astra_summary.py" "$RUN" > "$OUT/summary.txt" 2>&1; src=$?
nl_say "  summary rc=$src: $(grep -m1 '^VERDICT' "$OUT/summary.txt" | cut -c1-200)"
if [ "$src" = 0 ] && [ -f "$ART" ]; then
  nl_mark summary done "$(grep -m1 '^VERDICT' "$OUT/summary.txt" | cut -c9-60)"
  grep -m1 '^LINE1' "$OUT/summary.txt" | cut -c7- > "$OUT/line1.txt"
  { echo "# astra-stop Q4 replay readout"; echo; cat "$OUT/summary.txt" | sed -n 1,3p; echo; echo "regime: llama.cpp $LL_SHA, Q4_K_M both arms, KV $KV, ctx $CTX/$NP slots, sampler temp 1.0 top-p 0.95 top-k 20, reasoning effort xhigh, seeds 1-32, cap $MAXTOK"; } > "$OUT/readout.md"
  if [ -z "${ASTRA_NO_RENDER:-}" ]; then
    nl_mark render running
    $RPY "$CODE/render_astra.py" "$RUN" "$OUT/astra-stop.mp4" --rounds 5 --ffmpeg "$FF" >> "$OUT/render.log" 2>&1 \
      && "$FF" -nostdin -y -loglevel error -ss 3 -i "$OUT/astra-stop.mp4" -frames:v 1 "$OUT/contact-1.png" </dev/null \
      && "$FF" -nostdin -y -loglevel error -ss 55 -i "$OUT/astra-stop.mp4" -frames:v 1 "$OUT/contact-2.png" </dev/null \
      && nl_judge_artifact render 0 "$OUT/astra-stop.mp4" "$(date -d '-1 hour' +%s)" "clip" || nl_say "  render failed (see $OUT/render.log); the verdict stands"
  fi
  nl_say "ANGLE: $(cat "$OUT/line1.txt")"
  exit 0
fi
nl_say "  a leg could not finish inside the window (verdict INCOMPLETE): exit 3, resumable, summary.json NOT written (PROVISIONAL: $RUN/summary.pre.json)"
exit 3
