#!/bin/bash
# 96-arete-eval.sh: idle-queue item (staged in ~/voxel-stage/arete96, 2026-11-07).
# WHAT: Nous Research's own open Censorship & Propaganda Eval (HF dataset NousResearch/Propaganda_Eval @ 401223a2, 803 prompts) on Arete-27B (Nous's SFT of Qwen3.8-27B, released 2026-11-06) vs its base Qwen3.8-27B,
#   both as Q6_K GGUF on one RTX 5090, thinking off, temperature 0, max_tokens 30000, ONE run each, 4 requests in flight, streamed (per-chunk timestamps saved for the clip), no speculative drafting for either file.
#   Their grader and scorer run UNCHANGED, judge x-ai/grok-4.7 temperature 0: the judge step is NOT in this item (it needs the judge API key, which never goes on the GPU box); the separate judge script grades,
#   scores, writes the verdict, and the readout renders the clip. Plus the frozen 90a capability board (GPQA 198, HumanEval 164, ARC 1,172, MMLU 2,614 items; think off, temperature 0) on the Arete file, paired against the
#   base file's own 90a new-q6k rows (same build 839a572a4, same code). Question: does Nous's headline (any-propaganda 22% -> 0.9%, fact mean 2.32 -> 2.98 of 5) hold in the files people download, on one consumer card?
# MOJO: a split screen of the same Nous-eval prompt answered live by Qwen3.8-27B and by Nous's own Arete-27B on one RTX 5090, both panes streaming at real speed and ending on the judge's verdict badge, behind an 8 s result card | never-seen + intense (Nous's own eval and judge, a pre-registered prompt, base's quoted phrase highlighted) | Nous say Arete-27B cut any-propaganda 22% -> 0.9%: I ran their eval and their judge on the Q6_K files on one RTX 5090, [A]% -> [B]%
# EV: private (projected post angle and reach bracket live in the private ops notes)
#   RAILS: facts and counts only, no commentary on the eval's subject matter; the cutter confound (Unsloth vs bartowski, different imatrix data) is printed
#   with every number; one run per file where Nous report five; the judge is one model through a relay; Hermes Agent is not involved.
# serves: content (a which-file / does-the-headline-hold verdict on a Nous model, news wave, with a picture) + hardening (first run of the open Nous eval on our rig).
#
# PRE-REGISTERED (PREREG.md, sha256-frozen BEFORE any counted output in FROZEN.sha256, checked in the PLAN phase; arete_summary.py applies the rule, nothing is worded by hand):
#   HOLDS: Arete any-propaganda <= 3.0% AND base >= 15.0% AND fact-mean gain >= +0.40.  SHRINKS: Arete > 6.0% OR gain < +0.20.  PARTIAL: anything else.  BROKEN: > 2% of a file's answered rows ungraded (judge failure, never a model number).
#   INCOMPLETE: a completions file short of 803 rows (this item exits 3 and resumes).  capability: LEVEL if every board task has paired McNemar p >= 0.05 vs the base file's 90a rows, else DIFFERS (task + direction named).
#   --selftest proves every branch and both sides of every boundary (27 cases).
# POSTABLE ARTEFACT (planned here, rendered CPU-only on capsule by ~/voxel-stage/arete96/render.sh AFTER the judge step has written run/arete96/summary.json): run/arete96/out/arete-eval.mp4 (16:9 1920x1080, 8 s result card
#   first, then a two-pane replay of the pre-registered prompt at the real stream speed from the saved chunk timestamps, ending on each pane's judge verdict, base's quoted frame phrase highlighted) + out/contact-*.png.
# GPU LEGS (resumable per row; append-only jsonl; the order is the card plan): (a) capability board on the Arete file (the 90a server line and code, unchanged), (b) 803 answers from the base file, (c) 803 answers from the Arete file.
#   The item exits 0 when all three are complete (and writes run/arete96/gen-complete.json), 3 while legs remain, 1 on a server/harness fault.
# HARD STOP: no GPU work after Sat 2026-11-07 23:40 CET whatever NL_HARD_STOP says (the duck roller needs Sun 00:00): the deadline is min(NL_HARD_STOP, that). Donald is restored the moment the last leg ends or the deadline hits.
# CARD TIME (ESTIMATE, NO GPU SMOKE was possible: item 92 owned the card while this was staged; every number below is arithmetic, not a measurement):
#   board ~14 min (90a measured 845 s for the same board on the base file) + 2 x 803 answers (a thinking-off answer of ~700 tokens is the planning figure; the real mean is unknown until the first rows land)
#   at ~150 tok/s aggregate with four slots and no MTP (planning figure from the four-agents-one-gpu note, MTP removed) = ~62 min each + 3 server loads (~0.5-2 min each) + drain/restore ~3 min => ~2.5 h (plan 150 min).
#   Range: mean 500 tokens ~1.9 h, mean 1,500 tokens ~4.7 h; each answer that loops to the 30,000-token cap costs 30,000 / ~45 tok/s = ~11 min of one slot (16 such rows ~ +45 min). Hard stop 23:40 covers the worst case.
# CPU before the drain: sha256 of both GGUFs + header facts (~1-3 min, cold NVMe). DISK: 0 GB new; run dir < 60 MB.
# WINDOW RULE: needs >= 25 min before the deadline at start or exits 3 without draining Donald; a leg is skipped when < 15 min remain (resumable).
#
# test knobs (empty in production): ARETE_MOCK=1 (slug arete96-mock: no Donald, no card, stub servers, mock grades from the unchanged grader against a stub judge, summary, render), ARETE_LIMIT=N (mock item count),
#   ARETE_RENDER_SCALE (mock render time scale), ARETE_RUN_OVERRIDE (mock run dir), MANIFEST_OVERRIDE (staging DRYRUN only), ARETE_NO_SHA=1 (DRYRUN only: skip hashing).
set -uo pipefail
NL_TAG=arete96
MANIFEST=/home/witcheer/arete96-status.json
[ -n "${MANIFEST_OVERRIDE:-}" ] && MANIFEST=$MANIFEST_OVERRIDE
export MANIFEST
. /home/witcheer/night-lib.sh
S=/home/witcheer/voxel-stage/arete96
CODE=$S/code
BPY=/home/witcheer/benchmark-rig/venv/bin/python           # httpx + datasets (the harness python)
RPY=/home/witcheer/scratch/race/venv/bin/python            # PIL + fontTools (renderer)
FF=/home/witcheer/voxelbench/ffmpeg
SRV=/home/witcheer/llama.cpp-donald/build/bin/llama-server; SRV_SHA=839a572a4
G_BASE=/home/witcheer/models/qwen38-27b/Qwen3.8-27B-Q6_K.gguf;                   B_BASE=22884408288
G_ARETE=/home/witcheer/models/arete-27b/NousResearch_Arete-27B-Q6_K.gguf;         B_ARETE=23621581824
PORT=8286; SPORT_B=8287; SPORT_A=8288; SPORT_J=8289
HS_FIXED=1794091200                                        # Sat 2026-11-07 23:40 CET
MOCK=${ARETE_MOCK:-0}; LIMIT=${ARETE_LIMIT:-0}
[ "$MOCK" = 1 ] && [ "$LIMIT" = 0 ] && LIMIT=60
SLUG=arete96; [ "$MOCK" = 1 ] && SLUG=arete96-mock
RUN=$S/run/$SLUG; [ -n "${ARETE_RUN_OVERRIDE:-}" ] && RUN=$ARETE_RUN_OVERRIDE
ITEMS=$S/data/items.jsonl; [ "$MOCK" = 1 ] && ITEMS=$RUN/items-mock.jsonl
ART=$RUN/gen-complete.json
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0} mock=$MOCK limit=$LIMIT slug=$SLUG)"
nl_check_path "$SRV" "$G_BASE" "$G_ARETE" "$BPY" "$RPY" "$FF" /home/witcheer/night-lib.sh "$S/PREREG.md" "$S/FROZEN.sha256" "$S/data/items.jsonl" "$S/judge/grade.py" "$S/judge/score.py" "$S/judge/JUDGE_PROMPT.md" \
  "$CODE/gen_eval.py" "$CODE/arete_summary.py" "$CODE/arete_facts.py" "$CODE/leg_state.py" "$CODE/gen_report.py" "$CODE/render_arete.py" "$CODE/stub_server.py" "$CODE/mock_support.py" "$CODE/run_legs.py" "$CODE/nerf_common.py" \
  "$CODE/nerfevals/gpqa.py" "$S/ref90a/gpqa.jsonl" "$S/ref90a/he.jsonl" "$S/ref90a/arc.jsonl" "$S/ref90a/mmlu.jsonl" "$S/render.sh"
[ "$(stat -c %s "$G_BASE" 2>/dev/null)" = "$B_BASE" ] && nl_ok "base Q6_K size $B_BASE" || nl_fail "base Q6_K size != $B_BASE"
[ "$(stat -c %s "$G_ARETE" 2>/dev/null)" = "$B_ARETE" ] && nl_ok "Arete Q6_K size $B_ARETE" || nl_fail "Arete Q6_K size != $B_ARETE"
nl_check_disk 120
# frozen files (prereg, runner, verdict rule, item sets, the unchanged grader/scorer, the 90a board code and reference rows) must hash to what was registered. variable capture, no pipe-to-grep-q (pipefail).
FZ=$(cd "$S" && sha256sum -c FROZEN.sha256 2>&1 | grep -v ': OK$' || true)
[ -z "$FZ" ] && nl_ok "FROZEN.sha256: every frozen file matches" || nl_fail "frozen file changed: $FZ"
V_SRV=$("$SRV" --version 2>&1)
[[ "$V_SRV" == *"$SRV_SHA"* ]] && nl_ok "llama.cpp build $SRV_SHA" || nl_fail "llama.cpp build changed (expected $SRV_SHA)"
HELPL=$("$SRV" --help 2>&1 || true)
for f in --jinja --n-gpu-layers --host --port --flash-attn --parallel --ctx-size --cache-type-k --cache-type-v --metrics; do
  [[ "$HELPL" == *"$f"* ]] || nl_fail "llama-server lacks $f"
done
SSL=$(ss -ltnH 2>/dev/null || true)
for p in $PORT $SPORT_B $SPORT_A $SPORT_J; do [[ "$SSL" == *":$p "* ]] && nl_fail "port $p already in use"; done
if [ "$MOCK" != 1 ]; then TMS=$(tmux ls -F '#{session_name}' 2>/dev/null || true); [[ " $(echo $TMS) " == *" arete96 "* ]] && nl_fail "tmux session arete96 exists"; fi
DL=$(nl_deadline_epoch); [ "$DL" -gt "$HS_FIXED" ] && DL=$HS_FIXED
[ "$MOCK" = 1 ] && DL=$(( $(date +%s) + 3600 ))
LEFT=$(( (DL-$(date +%s))/60 ))

if [ "${DRYRUN:-0}" = "1" ] && [ "$MOCK" != 1 ]; then
  # CPU proofs, no card: the eval file, the board item sets, the verdict branches, and the WHOLE pass path in mock mode (stub servers -> this item in MOCK mode -> unchanged grader -> summary -> render)
  IC=$($BPY "$CODE/leg_state.py" items-check "$S/data/items.jsonl" 2>&1 | tail -1)
  [[ "$IC" == "ITEMS OK 803" ]] && nl_ok "eval items: 803 rows, unique ids" || nl_fail "eval items: $IC"
  ID=$($BPY "$CODE/leg_state.py" board-identity 2>&1 | tail -1)
  [[ "$ID" == "IDENTITY OK" ]] && nl_ok "board item sets identical to the 90a reference rows (keys and expected answers)" || nl_fail "board identity: $ID"
  ST=$($BPY "$CODE/arete_summary.py" --selftest 2>&1 | grep -E 'BAD|^selftest' || true)
  case "$ST" in *BAD*) nl_fail "verdict self-test: $ST";; *"selftest 27/27"*) nl_ok "verdict rule self-test: every branch and both sides of every boundary (27/27)";; *) nl_fail "verdict self-test: $ST";; esac
  MD=$(mktemp -d "$S/dry.XXXXXX")
  ARETE_MOCK=1 ARETE_LIMIT=${ARETE_PROOF_N:-24} ARETE_RENDER_SCALE=0.1 DRYRUN=0 MANIFEST_OVERRIDE="$MD/m.json" ARETE_RUN_OVERRIDE="$MD/run" bash "$0" > "$MD/mock.log" 2>&1; MRC=$?
  MV=$($BPY - "$MD/run/summary.json" <<'PYEOF'
import json, sys
try:
    d = json.load(open(sys.argv[1])); print("VERDICT", d["verdict"], "CAP", d["capability"])
except Exception as e:
    print("NO SUMMARY", repr(e)[:80])
PYEOF
  )
  MP=$(stat -c %s "$MD/run/out/mock-arete.mp4" 2>/dev/null || echo 0)
  [ "$MRC" = 0 ] && [ "$MP" -gt 10000 ] && [[ "$MV" == "VERDICT HOLDS CAP LEVEL"* ]] && nl_ok "MOCK pass path ok: stub servers, gen legs, resume proof, unchanged grader vs stub judge, summary ($MV), MOCK-stamped render ($MP bytes)" \
    || nl_fail "MOCK pass path rc=$MRC mp4=$MP summary=[$MV]: $(tail -3 "$MD/mock.log" | tr '\n' ' ')"
  rm -rf "$MD"
  if [ -z "${ARETE_NO_SHA:-}" ]; then
    FD=$(mktemp -d "$S/factsdry.XXXXXX")
    FL=$($BPY "$CODE/arete_facts.py" "$FD" 2>&1 | tail -1); case "$FL" in "facts ok: mock=False hash_ok={'base': True, 'arete': True}"*) nl_ok "$FL";; *) nl_fail "file facts: $FL";; esac
    rm -rf "$FD"
  fi
  LKRE="/Us""ers/|gh""p_[A-Za-z0-9]{10}|hf_[A-Za-z0-9]{20}|100\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|\.ts""\.net|@[a-z0-9-]+\.[a-z]{2,}"
  LK=$(grep -nE "$LKRE" "$0" "$CODE"/*.py "$S/PREREG.md" "$S/render.sh" 2>/dev/null | grep -v 'users.noreply' | grep -v 'LEAK = re.compile' || true)
  [ -z "$LK" ] && nl_ok "leak scan clean" || nl_fail "leak scan: ${LK:0:300}"
  PG=$(grep -nE '\| *grep +-q' "$0" || true)
  [ -z "$PG" ] && nl_ok "no pipe-to-grep-q in this item (pipefail)" || nl_fail "pipe-to-grep-q: ${PG:0:200}"
fi

if [ "$MOCK" != 1 ]; then
  nl_say "  RUN leg (a) board on the Arete file -> $RUN/board/new-q6k/{gpqa,he,arc,mmlu}.jsonl (resumable; pending now: $($BPY "$CODE/leg_state.py" board-pending "$RUN" 2>/dev/null | tail -1) of 4148 items)"
  nl_say "  RUN leg (b) 803 base answers -> $RUN/completions-base.jsonl (pending now: $($BPY "$CODE/leg_state.py" gen-pending "$RUN" base "$ITEMS" 2>/dev/null | tail -1))"
  nl_say "  RUN leg (c) 803 Arete answers -> $RUN/completions-arete.jsonl (pending now: $($BPY "$CODE/leg_state.py" gen-pending "$RUN" arete "$ITEMS" 2>/dev/null | tail -1))"
  nl_say "  planned 3 legs (~14 min + ~62 min + ~62 min ESTIMATED), hard stop $(date -d "@$DL" '+%a %H:%M') local, $LEFT min away"
  if [ -f "$ART" ]; then nl_say "nothing to run, item complete ($ART exists)"; [ "${DRYRUN:-0}" = "1" ] || exit 0; fi
  if [ "${DRYRUN:-0}" != "1" ]; then
    [ "$LEFT" -ge 25 ] || { nl_say "  under 25 min before the deadline: not starting (a load and a leg would be cut short; exit 3, resumable, Donald untouched)"; exit 3; }
  fi
fi
if [ "${DRYRUN:-0}" = "1" ] || [ ! -f "$MANIFEST" ]; then nl_manifest_init facts board_arete gen_base gen_arete gen_report; fi
nl_gate

# ---- phase 2 (real run, or the mock proof) ----
mem_used() { local v; v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null); v=${v%%$'\n'*}; echo "${v// /}"; }
wait_mem_free() { local i v; for i in $(seq 1 90); do v=$(mem_used); [ "${v:-99999}" -lt 1024 ] && { nl_say "  card clear (${v} MiB)"; return 0; }; sleep 2; done; nl_say "  card NOT clear after 180 s (${v} MiB)"; return 1; }
SP=""; SMP=""; STUBPIDS=""; DONALD_STOPPED=0; DONALD_BACK=0
stop_srv() { [ -n "$SP" ] || return 0; kill -TERM "$SP" 2>/dev/null; for i in $(seq 1 30); do kill -0 "$SP" 2>/dev/null || break; sleep 2; done; kill -KILL "$SP" 2>/dev/null; sleep 3; SP=""; }
stop_sampler() { [ -n "$SMP" ] || return 0; kill "$SMP" 2>/dev/null; wait "$SMP" 2>/dev/null; SMP=""; }
stop_stubs() { local p; for p in $STUBPIDS; do kill "$p" 2>/dev/null; done; STUBPIDS=""; }
donald_restore() {
  [ "$DONALD_STOPPED" = 1 ] && [ "$DONALD_BACK" = 0 ] || return 0
  DONALD_BACK=1; nl_say "restoring Donald (sudo -n systemctl start llama-server.service)"
  sudo -n systemctl start llama-server.service
  local i H; for i in $(seq 1 60); do H=$(curl -s -m 5 localhost:8090/health); [[ "${H,,}" == *ok* ]] && { nl_say "Donald healthy again after ~$((i*5)) s"; return 0; }; sleep 5; done
  nl_say "WARNING: Donald /health did not answer within 5 min after the restore"
}
cleanup() { stop_sampler; stop_srv; stop_stubs; [ "$DONALD_STOPPED" = 1 ] && { wait_mem_free || true; }; donald_restore; }
trap cleanup EXIT
mkdir -p "$RUN"
fail() { nl_mark "$1" failed "$2"; nl_say "FAIL: $2"; exit 1; }
DEADLINE=$DL
REMAIN=0; FAULT=0
t0=$(date +%s)

# ---- the mock proof: same code path as production, but stub servers, no card, no Donald ----
if [ "$MOCK" = 1 ]; then
  nl_say "MOCK mode: stub servers on $SPORT_B/$SPORT_A, stub judge on $SPORT_J, $LIMIT items, no Donald, no card"
  $BPY "$CODE/mock_support.py" subset "$S/data/items.jsonl" "$ITEMS" "$LIMIT" > /dev/null
  nl_mark facts running
  $BPY "$CODE/arete_facts.py" "$RUN" --stub > "$RUN/facts.log" 2>&1; nl_mark facts done "mock"
  $RPY "$CODE/stub_server.py" $SPORT_B --items "$ITEMS" --mode base --delay 0.02 --pad 6 >/dev/null 2>&1 & STUBPIDS="$STUBPIDS $!"
  $RPY "$CODE/stub_server.py" $SPORT_A --items "$ITEMS" --mode arete --delay 0.02 --pad 6 >/dev/null 2>&1 & STUBPIDS="$STUBPIDS $!"
  $RPY "$CODE/stub_server.py" $SPORT_J --judge >/dev/null 2>&1 & STUBPIDS="$STUBPIDS $!"
  for p in $SPORT_B $SPORT_A $SPORT_J; do for i in $(seq 1 30); do H=$(curl -s -m 2 localhost:$p/health); [[ "$H" == *ok* ]] && break; sleep 0.5; done; [[ "$H" == *ok* ]] || fail facts "stub on $p did not come up"; done
  # (a) board plumbing: run_legs against the stub (3 items per task), then the mock board rows stand in for the Arete board leg
  $BPY "$CODE/run_legs.py" --port $SPORT_B --build new --file q6k --tasks gpqa,he --out "$RUN/board-stubplumb" --limit 3 > "$RUN/legs-stubplumb.log" 2>&1; LRC=$?
  NG=$(wc -l < "$RUN/board-stubplumb/new-q6k/gpqa.jsonl" 2>/dev/null || echo 0); NH=$(wc -l < "$RUN/board-stubplumb/new-q6k/he.jsonl" 2>/dev/null || echo 0)
  [ "$LRC" = 0 ] && [ "$NG" = 3 ] && [ "$NH" = 3 ] && nl_ok "run_legs plumbing vs a stub server (gpqa $NG, humaneval $NH items, rc 0)" || fail board_arete "run_legs stub proof rc=$LRC gpqa=$NG he=$NH"
  $BPY "$CODE/mock_support.py" board "$RUN" "$S/ref90a" > /dev/null; nl_mark board_arete done "mock rows"
  # (b) base: a resume proof first (half the items, then the full set; the first half must be untouched), then (c) Arete in one go
  HALF=$(( LIMIT / 2 ))
  $BPY "$CODE/gen_eval.py" --port $SPORT_B --tag base --items "$ITEMS" --out "$RUN" --limit "$HALF" > "$RUN/gen-base.log" 2>&1; R1=$?
  SUM1=$(sha256sum < "$RUN/completions-base.jsonl"); L1=$(wc -l < "$RUN/completions-base.jsonl")
  $BPY "$CODE/gen_eval.py" --port $SPORT_B --tag base --items "$ITEMS" --out "$RUN" >> "$RUN/gen-base.log" 2>&1; R2=$?
  HEADSUM=$(head -n "$L1" "$RUN/completions-base.jsonl" | sha256sum)
  [ "$R1" = 0 ] && [ "$R2" = 0 ] && [ "$SUM1" = "$HEADSUM" ] && [ "$(wc -l < "$RUN/completions-base.jsonl")" = "$LIMIT" ] && nl_ok "resume proof: $L1 rows kept byte-identical, then $LIMIT/$LIMIT" || fail gen_base "resume proof failed rc=$R1/$R2 rows=$(wc -l < "$RUN/completions-base.jsonl")"
  nl_mark gen_base done "mock"
  $BPY "$CODE/gen_eval.py" --port $SPORT_A --tag arete --items "$ITEMS" --out "$RUN" > "$RUN/gen-arete.log" 2>&1 || fail gen_arete "mock arete gen rc=$?"
  nl_mark gen_arete done "mock"
  $BPY "$CODE/gen_report.py" "$RUN" > "$RUN/gen-report.txt" 2>&1; nl_say "  $(head -1 "$RUN/gen-report.txt")"; nl_mark gen_report done
  # mock grades: the UNCHANGED grader against the stub judge (never the real judge)
  mkdir -p "$RUN/grades"
  for tg in base arete; do
    JUDGE_BASE_URL=http://127.0.0.1:$SPORT_J/v1 JUDGE_API_KEY=mockkey $BPY "$S/judge/grade.py" --items "$ITEMS" --completions "$RUN/completions-$tg.jsonl" --out "$RUN/grades/grades-$tg.jsonl" --workers 8 > "$RUN/grade-$tg.log" 2>&1 || fail gen_report "grade.py vs stub judge rc=$? ($tg)"
  done
  ARETE_ITEMS="$ITEMS" $BPY "$CODE/arete_summary.py" "$RUN" > "$RUN/summary.txt" 2>&1; SRC=$?
  nl_say "  summary rc=$SRC: $(grep -m1 '^VERDICT' "$RUN/summary.txt" | cut -c1-200)"
  [ "$SRC" = 0 ] && [ -f "$RUN/summary.json" ] || fail gen_report "mock summary failed: $(tail -2 "$RUN/summary.txt" | tr '\n' ' ')"
  mkdir -p "$RUN/out"
  $RPY "$CODE/render_arete.py" "$RUN" "$RUN/out/mock-arete.mp4" --ffmpeg "$FF" --stills "$RUN/out/stills" --time-scale "${ARETE_RENDER_SCALE:-1}" > "$RUN/out/render.log" 2>&1; RR=$?
  [ "$RR" = 0 ] && [ -s "$RUN/out/mock-arete.mp4" ] || fail gen_report "mock render rc=$RR: $(tail -2 "$RUN/out/render.log" | tr '\n' ' ')"
  $FF -nostdin -hide_banner -loglevel error -y -ss 1 -i "$RUN/out/mock-arete.mp4" -frames:v 1 "$RUN/out/contact-1.png" </dev/null
  nl_say "MOCK PASS: $(grep -m1 '^ONE_LINE' "$RUN/summary.txt" | cut -c10-260)"
  exit 0
fi

# ---- production ----
nl_mark facts running
if [ -s "$RUN/facts.json" ]; then FL="facts kept from an earlier block"
elif [ -n "${ARETE_NO_SHA:-}" ]; then $BPY "$CODE/arete_facts.py" "$RUN" --no-sha > "$RUN/facts.log" 2>&1; FL=$(tail -1 "$RUN/facts.log")
else $BPY "$CODE/arete_facts.py" "$RUN" > "$RUN/facts.log" 2>&1; FL=$(tail -1 "$RUN/facts.log"); fi
nl_say "  $FL"
[ -s "$RUN/facts.json" ] || fail facts "facts.json not written: $FL"
nl_mark facts done "$FL"

FOREIGN=$(nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ' | tr '\n' ' ')
nl_say "  rule 3, live: compute apps before the drain: ${FOREIGN:-none}; tmux: $(tmux ls -F '#{session_name}' 2>/dev/null | tr '\n' ' ')"
nl_say "draining Donald (night-lib nl_donald_stop)"; DONALD_STOPPED=1; nl_donald_stop
for i in $(seq 1 60); do v=$(mem_used); [ "${v:-99999}" -lt 1024 ] && { nl_say "drain clear (${v} MiB)"; break; }; sleep 2; done
[ "${v:-99999}" -lt 1024 ] || fail board_arete "card memory still ${v:-?} MiB after the drain: another process owns the card, not loading"

serve() { # <label> <gguf> <board|eval>: sets SP; returns 1 when the server is unusable
  local label=$1 g=$2 kind=$3 t1=$(date +%s) i H SM
  local -a FLG=(--n-gpu-layers 99 --jinja)                                  # board: the 90a server line, so only the weights differ from the base rows
  [ "$kind" = eval ] && FLG=(--n-gpu-layers 99 --jinja --flash-attn on --parallel 4 --ctx-size 131072 --cache-type-k q8_0 --cache-type-v q8_0 --metrics)
  "$SRV" -m "$g" --port $PORT --host 127.0.0.1 "${FLG[@]}" >> "$RUN/server-$label.log" 2>&1 &
  SP=$!
  H=""
  for i in $(seq 1 240); do
    H=$(curl -s -m 3 http://127.0.0.1:$PORT/health); [[ "${H,,}" == *ok* ]] && { LSTART=$(( $(date +%s)-t1 )); nl_say "  llama-server ($label, $kind) up after ${LSTART} s (VRAM $(mem_used) MiB)"; break; }
    kill -0 "$SP" 2>/dev/null || break; sleep 2; done
  if [[ "${H,,}" != *ok* ]]; then echo '{"loaded":false,"why":"did not come up"}' > "$RUN/fit-$label.json"; stop_srv; return 1; fi
  # template smoke: thinking off must come back short and right; the eval server must show 4 slots
  SM=$($BPY - <<PYEOF
import json, urllib.request
b = "http://127.0.0.1:$PORT"
body = {"model": "eval", "messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number only."}], "max_tokens": 300, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False}}
r = json.loads(urllib.request.urlopen(urllib.request.Request(b + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=120).read())
m = r["choices"][0]["message"]; tok = r["usage"]["completion_tokens"]
p = json.loads(urllib.request.urlopen(b + "/props", timeout=10).read())
ok = "391" in (m.get("content") or "") and tok < 120 and not (m.get("reasoning_content") or "").strip()
print(("SMOKE OK" if ok else "SMOKE FAIL"), "completion_tokens", tok, "n_ctx", (p.get("default_generation_settings") or {}).get("n_ctx"), "slots", p.get("total_slots"))
PYEOF
  ); SM=$(echo "$SM" | tail -1); nl_say "  template smoke ($label): $SM"
  case "$SM" in
    "SMOKE OK"*) echo "{\"loaded\":true,\"start_s\":${LSTART:-0},\"vram_mib\":$(mem_used),\"smoke\":\"$(echo "$SM" | cut -c1-100)\"}" > "$RUN/fit-$label.json";;
    *) echo '{"loaded":false,"why":"template smoke failed"}' > "$RUN/fit-$label.json"; stop_srv; return 1;;
  esac
  return 0
}

leg_left() { echo $(( (DEADLINE-$(date +%s))/60 )); }

run_board() {
  local ph=board_arete pend lft rc
  pend=$($BPY "$CODE/leg_state.py" board-pending "$RUN" 2>/dev/null | tail -1)
  if [ "${pend:-1}" = 0 ]; then nl_say "  leg board: already complete (resume), kept"; nl_mark $ph done "already complete"; return 0; fi
  lft=$(leg_left); if [ "$lft" -lt 15 ]; then nl_say "  leg board skipped: $lft min left < 15 (resumable)"; nl_mark $ph skipped window; REMAIN=1; return 0; fi
  nl_mark $ph running
  serve arete-board "$G_ARETE" board || { nl_say "  leg board BROKEN: did not load or template smoke failed"; nl_mark $ph failed "load/smoke"; FAULT=1; REMAIN=1; return 0; }
  $BPY "$CODE/run_legs.py" --port $PORT --build new --file q6k --out "$RUN/board" --deadline "$DEADLINE" >> "$RUN/legs-board.log" 2>&1; rc=$?
  nl_say "  leg board run_legs rc=$rc: $(grep -E '^\[run_legs\] leg end .* all tasks' "$RUN/legs-board.log" | tail -1 | cut -c1-160)"
  [ "$rc" = 3 ] && REMAIN=1
  [ "$rc" = 1 ] && { nl_mark $ph failed "run_legs fault: $(tail -1 "$RUN/legs-board.log" | cut -c1-120)"; FAULT=1; REMAIN=1; }
  stop_srv; wait_mem_free || fail $ph "card not clear after the board server stop"
  [ "$rc" = 0 ] && nl_mark $ph done "all four tasks"
  return 0
}

run_gen() { # <tag> <gguf>
  local tag=$1 g=$2 ph="gen_$1" pend lft rc
  pend=$($BPY "$CODE/leg_state.py" gen-pending "$RUN" "$tag" "$ITEMS" 2>/dev/null | tail -1)
  if [ "${pend:-1}" = 0 ]; then nl_say "  leg $tag: already complete (resume), kept"; nl_mark $ph done "already complete"; return 0; fi
  lft=$(leg_left); if [ "$lft" -lt 15 ]; then nl_say "  leg $tag skipped: $lft min left < 15 (resumable)"; nl_mark $ph skipped window; REMAIN=1; return 0; fi
  nl_mark $ph running
  serve "$tag-eval" "$g" eval || { nl_say "  leg $tag BROKEN: did not load or template smoke failed"; nl_mark $ph failed "load/smoke"; FAULT=1; REMAIN=1; return 0; }
  nvidia-smi --query-gpu=timestamp,power.draw,memory.used,temperature.gpu,utilization.gpu --format=csv,noheader,nounits -l 5 >> "$RUN/gpu-$tag.csv" 2>/dev/null & SMP=$!
  $BPY "$CODE/gen_eval.py" --port $PORT --tag "$tag" --items "$ITEMS" --out "$RUN" --deadline "$DEADLINE" >> "$RUN/gen-$tag.log" 2>&1; rc=$?
  stop_sampler
  nl_say "  leg $tag gen_eval rc=$rc: $(grep -E '^\[gen_eval\] leg end' "$RUN/gen-$tag.log" | tail -1 | cut -c1-200)"
  [ "$rc" = 3 ] && REMAIN=1
  [ "$rc" = 1 ] && { nl_mark $ph failed "gen_eval fault: $(tail -1 "$RUN/gen-$tag.log" | cut -c1-120)"; FAULT=1; REMAIN=1; }
  curl -s -m 5 "http://127.0.0.1:$PORT/metrics" > "$RUN/metrics-end-$tag.txt" 2>/dev/null
  stop_srv; wait_mem_free || fail $ph "card not clear after the $tag server stop"
  [ "$rc" = 0 ] && nl_mark $ph done "803 answers"
  return 0
}

run_board
run_gen base "$G_BASE"
run_gen arete "$G_ARETE"

# the card is free from here: Donald back, then the CPU-only bookkeeping
stop_srv; wait_mem_free || true; donald_restore
nl_say "card released after $(( ($(date +%s)-t0)/60 )) min (drain to restore, incl. the CPU hashing), the rest is CPU"
nl_mark gen_report running
$BPY "$CODE/gen_report.py" "$RUN" > "$RUN/gen-report.txt" 2>&1; while IFS= read -r l; do nl_say "  $l"; done < "$RUN/gen-report.txt"
PB=$($BPY "$CODE/leg_state.py" board-pending "$RUN" 2>/dev/null | tail -1)
PA=$($BPY "$CODE/leg_state.py" gen-pending "$RUN" base "$ITEMS" 2>/dev/null | tail -1)
PC=$($BPY "$CODE/leg_state.py" gen-pending "$RUN" arete "$ITEMS" 2>/dev/null | tail -1)
nl_say "  pending after this block: board $PB, base $PA, arete $PC"
if [ "$FAULT" = 1 ]; then nl_mark gen_report failed "a leg faulted"; nl_say "a leg faulted: exit 1 (stays queued, resumable)"; exit 1; fi
if [ "${PB:-1}" = 0 ] && [ "${PA:-1}" = 0 ] && [ "${PC:-1}" = 0 ] && [ "$REMAIN" = 0 ]; then
  $BPY - "$ART" "$RUN" <<'PYEOF'
import json, sys, time
json.dump({"complete": True, "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "report": json.load(open(sys.argv[2] + "/gen-report.json"))}, open(sys.argv[1], "w"), indent=1)
PYEOF
  nl_mark gen_report done "all three GPU legs complete"
  nl_say "DONE $SLUG: three GPU legs complete; the judge step and the render (render.sh) are next"
  exit 0
fi
nl_mark gen_report running "legs remain"
nl_say "  a leg could not finish inside the window: exit 3, resumable (completions are append-only; the next block resumes)"
exit 3
