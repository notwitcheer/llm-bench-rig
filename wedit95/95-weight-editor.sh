#!/bin/bash
# 95-weight-editor.sh: idle-queue item (STAGED in ~/voxel-stage, 2026-11-06, mercury subagent for the GPU thread, mojo-scan pick F; sorts LAST on purpose: bank-quality, high failure risk, time-boxed ~150 min).
# WHAT: can a LOCAL Qwen3.8-27B (Q6_K, llama.cpp), driving a real Hermes Agent session, find a weight edit that cures Qwen3.5-0.8B's thinking loops? Context (read 2026-11-06): MIT CSAIL's weight-space programming post (2026-11-02) reports a frontier-model editor
#   cutting the 0.8B model's loops with a rank-limited edit of two MLP matrices (layers 15 and 19 down_proj); a scripted replication with no editor model cures 15/18 problems (KidIkaros/wsp-reproduction), a blind frontier-editor search in another replay did not (jsaurabh/weight-space-lab).
#   The edit effect is therefore known; this item asks the open question: does a LOCAL editor find it? Arms on held-out GSM8K test problems (40, one try each): base | published edit (bit-exact, sha256-verified reference arm) | scripted pipeline (arm B, no model in the loop) | the editor's best-by-dev edit (arm A).
#   The editor sees only a sandbox command `wd` (forward passes of the 0.8B, base weights, candidate builder, a 20-problem dev set, 6 evaluations, ~55 min). It is NOT told the published layers, neurons or strength.
# MOJO: the 0.8B model's thinking text literally looping ("wait, let me re-check...") on the left, a real Hermes Agent session editing its weights in the centre, the same problem answered after the editor's edit on the right, with the loop-rate counter and the editor's dev scores arriving live | esoteric (a 2-day-old weight-space paper reproduced on our card with a LOCAL model as the editor; checked, never claimed as first) + intense (a 27B and a 0.8B on one 32 GB card) | a local Qwen3.8-27B in Hermes Agent edited a 0.8B model's weights to stop its thinking loops: held-out loops [N/40 base -> M/40 after the editor's edit], the scripted pipeline [S/40], the published edit [P/40]
# EV: private (projected post angle and reach bracket live in the ops vault status note)
#   bracket UNVERIFIED: 1.5k-6k, wide (mojo-scan-2026-11-06 pick 6: pheromone replay 822 views at T+78 min, the MIT thread at r/LocalLLaMA score 595). A failure of the editor is a valid, pre-registered, still-postable result ("could not match the scripted pipeline"), phrased without a dig at anyone.
#   RAILS: never "first"; never "an LLM read the weights" as discovery (the discovery is the MIT authors'); Hermes Agent is never framed as flawed (the subject of a failure is the editor model and the task); the scripted arm is OUR reading of the replication's pipeline, not the authors' code.
# serves: content (esoteric lane: an unseen local-model variant of a viral paper, with a picture) + experiment (can a local model do weight-space editing at all).
#
# PRE-REGISTERED (wedit95/code/PREREG.md, sha256-frozen BEFORE any counted output in FROZEN.sha256, checked in the PLAN phase; wd_summary.py applies the rule, nothing is worded by hand; `wd_summary.py --selftest` proves 11 branches):
#   loop = the 0.8B's thinking never closed inside the 16,384-token cap. cured(arm) = one-sided exact Fisher, loops LOWER than base, p<0.05 (40 held-out tries per arm).
#   verdicts: INCOMPLETE (exit 3) | INVALID (base loop rate < 40%) | EDIT DID NOT REPRODUCE (published edit not cured) | NO CANDIDATE | EDITOR BEAT THE SCRIPT | EDITOR MATCHED THE SCRIPT | EDITOR HELPED, SCRIPT BETTER | EDITOR CURED, SCRIPT DID NOT | EDITOR COULD NOT MATCH THE SCRIPT | NEITHER CURED IT.
#   the editor's final candidate = best by DEV loop rate among those it scored (ties: earlier). take flags (outside-workdir, network, no-wd-use) are printed with line 1.
# POSTABLE ARTEFACT (planned here, CPU only on capsule after the card is released): run/wedit95/out/weight-editor.mp4 (16:9 1920x1080 ~64 s: result card 8 s FIRST, three panes with the loop counters, the card again; the editor's Hermes pane is the recorded take time-compressed with a speed pill; leak mask on the pane)
#   + out/contact-*.png + out/readout.md + summary.json. mock-render proof: the PLAN phase (DRYRUN) renders a MOCK-stamped clip from a CPU mock take at 0.1x time.
# HONEST LIMITS: one 0.8B model, GSM8K, 40 held-out tries per arm (a gap under ~20 points is not detectable), ONE editing session of ONE local 27B (not a benchmark of Hermes Agent or of the 27B as an editor), 16,384 cap (the paper's was 32,768), the replication's sampler (presence penalty 1.5),
#   bf16 GGUF on llama.cpp (not vLLM), the editor chooses its own layers, the editor's Hermes pane is real but compressed in time (speed pill). the editor has a terminal: reading outside its workdir is flagged, not prevented (no network is configured for it; a numpy-only python).
# CARD TIME (honest). GPU SMOKES 2026-11-06 (card free, flock'd, never counted): (1) 0.8B bf16 GGUF, 24 tries of 8,192 tokens at 24 slots: 219 s (decode ~117 tok/s per slot at 8 slots, ~44 tok/s at 24), base loop rate 22/24 at cap 8,192 and 22/24 at 16,384 (INVALID rule: >=40% needed, so the failure reproduces here);
#   (2) the published edit, same set: 21/24 loops at 8,192 but 13/24 at 16,384 (396 s): the edit helps, the cap matters, which is why the cap is 16,384; (3) coexistence: 27B Q6_K ctx 65,536 q8_0 (23.8 GB) + 0.8B at 16 slots x 17.6k = 29.5 GB peak of 32.6 (20 slots ~30.8 GB).
#   ESTIMATE (UNVERIFIED until a full run): drain/load 3 + base dev ~8 + tables 1 + base held ~10 + published held ~10 + scripted held ~10 + 27B load 2 + the take 60 (wall cap) + up to 12 for an eval in flight + editor held ~10 = ~126 min, +CPU render 1 min. plan 150.
#   the take itself is the unknown: a 27B at ~50-60 tok/s thinking in Hermes with a 3-8 min tool call per dev evaluation. up to 6 evaluations fit; fewer is a result, not a fault.
# DISK: base 0.8B safetensors 1.7 GB + bf16 GGUF 1.6 GB (staged) + up to 10 candidate GGUFs x 1.6 GB during the take (non-best ones deleted after the gate; each apply refuses under 123 GB free) + published/scripted/best 4.8 GB; <= ~25 GB peak, ~8 GB kept. numpy venv 83 MB. runs/results < 100 MB.
# WINDOW RULE: each 0.8B arm needs >= 15 min left, the take needs >= 80 min left (it is not resumable: a cut take is moved aside and re-run from scratch next block), the editor-held arm needs >= 15 min; exit 3 when a phase was skipped (resumable per arm file). no new phase after the deadline minus 8 min.
#   SORT: `95-` sorts after every other bench item; do not enqueue before the Sat 11-07 drain has room for ~150 min.
#
# test knobs (empty in production): WE_SMOKE=1 (slug -smoke, 6 dev + 8 held problems, cap 2048, a 7-min tool clock, 2 evaluations, no held arms except the editor's, no summary/render; WE_NO_DRAIN=1 = card must show < 8 GB used),
#   WE_STUB=1 (CPU mock editor + stub tool service: no drain, no servers, DRYRUN-safe), WE_NO_SELFTEST=1, MANIFEST_OVERRIDE (staging DRYRUN only).
set -uo pipefail
NL_TAG=wedit95
MANIFEST=/home/witcheer/wedit95-status.json
[ -n "${MANIFEST_OVERRIDE:-}" ] && MANIFEST=$MANIFEST_OVERRIDE
export MANIFEST
. /home/witcheer/night-lib.sh
WD=/home/witcheer/voxel-stage/wedit95
CODE=$WD/code
BPY=/home/witcheer/benchmark-rig/venv/bin/python            # torch + transformers + datasets + numpy (acts, gen, summary)
RPY=/home/witcheer/scratch/race/venv/bin/python             # pyte + PIL + fontTools (recorder + renderer)
FF=/home/witcheer/voxelbench/ffmpeg
CHECKOUT=/home/witcheer/.hermes/hermes-agent; HBIN=$CHECKOUT/.hermes/bin/hermes; HERMES_PIN=9bda188
SRV=/home/witcheer/llama.cpp-donald/build/bin/llama-server; SRV_SHA=839a572a4
G27=/home/witcheer/models/qwen38-27b/Qwen3.8-27B-Q6_K.gguf; B27=22884408288
BASE_DIR=$WD/base; BASE_ST=$BASE_DIR/model.safetensors-00001-of-00001.safetensors; BASE_ST_SHA=04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696
BASE_GGUF=$WD/gguf/base-bf16.gguf; B_GGUF=1557662688
export WD_BASE_GGUF=$BASE_GGUF WD_BASE_DIR=$BASE_DIR WD_SRV=$SRV HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
GEN_PORT=8341; SVC_PORT=8342; EVAL_PORT=8343; PROXY_PORT=8344; ED_PORT=8345; MOCK_PORT=8346; MOCK_SVC=8347; MOCK_EVAL=8348; MOCK_PROXY=8349
SMOKE=${WE_SMOKE:-0}; NODRAIN=${WE_NO_DRAIN:-0}
SLUG=wedit95; [ "$SMOKE" = 1 ] && SLUG=$SLUG-smoke
RUN=$WD/runs/$SLUG; ARMS=$RUN/arms; TAKE=$RUN/take; OUT=$RUN/out; ART=$RUN/summary.json; CAND=$RUN/cand
CAP=16384; DEVLIM=0; HELDLIM=0; EVALS=6; MINUTES=55; WALL=3600; ARMMIN=15; TAKEMIN=80
if [ "$SMOKE" = 1 ]; then CAP=2048; DEVLIM=6; HELDLIM=8; EVALS=2; MINUTES=7; WALL=420; ARMMIN=0; TAKEMIN=0; fi

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0} smoke=$SMOKE stub=${WE_STUB:-0} nodrain=$NODRAIN)"
nl_check_path "$SRV" "$G27" "$BPY" "$RPY" "$FF" "$HBIN" "$BASE_ST" "$BASE_GGUF" "$BASE_DIR/config.json" "$WD/npvenv/bin/python3" "$WD/ref/materials/edit.npz" "$WD/ref/materials/manifest.json" "$CODE/wd_take.sh" "$CODE/wd_service.py" "$CODE/wd_gen.py" \
  "$CODE/wd_summary.py" "$CODE/wd_gate.py" "$CODE/wd_scripted.py" "$CODE/wd_ref.py" "$CODE/wd_acts.py" "$CODE/render_wedit.py" "$CODE/wd_make_mock.py" "$CODE/wd_mock.py" "$CODE/prompts/edit-task.txt" "$CODE/PREREG.md" "$CODE/FROZEN.sha256" \
  /home/witcheer/llama.cpp-qwen4/gguf-py/gguf/__init__.py /home/witcheer/night-lib.sh
[ "$(stat -c %s "$G27" 2>/dev/null)" = "$B27" ] && nl_ok "27B Q6_K size $B27" || nl_fail "27B Q6_K size != $B27"
[ "$(stat -c %s "$BASE_GGUF" 2>/dev/null)" = "$B_GGUF" ] && nl_ok "0.8B bf16 GGUF size $B_GGUF" || nl_fail "0.8B GGUF size != $B_GGUF"
nl_check_disk 125
FZ=$(cd "$CODE" && sha256sum -c FROZEN.sha256 2>&1 | grep -v ': OK$' || true)
[ -z "$FZ" ] && nl_ok "FROZEN.sha256: every frozen file matches (prereg, arms, verdict rule, tool service, task prompt, published edit)" || nl_fail "frozen file changed: $FZ"
case "$($SRV --version 2>&1)" in *"$SRV_SHA"*) nl_ok "llama.cpp build $SRV_SHA";; *) nl_fail "llama.cpp build changed (expected $SRV_SHA)";; esac
HELPL=$($SRV --help 2>&1 || true); for f in --jinja --parallel --ctx-size --n-gpu-layers --flash-attn --cache-type-k --presence-penalty --alias; do case "$HELPL" in *"$f"*) ;; *) nl_fail "llama-server flag $f not in this build";; esac; done
HV=$(cd "$CHECKOUT" && git rev-parse --short HEAD 2>/dev/null); case "$HV" in "$HERMES_PIN"*) nl_ok "Hermes checkout $HV (pin $HERMES_PIN)";; *) nl_fail "Hermes checkout $HV != pin $HERMES_PIN";; esac
for dd in installs tools; do [ -d "/home/witcheer/.hermes/$dd" ] || nl_fail "main Hermes root has no $dd/ to share with the take home"; done
for p in $GEN_PORT $SVC_PORT $EVAL_PORT $PROXY_PORT $ED_PORT; do ss -ltnH "sport = :$p" 2>/dev/null | grep -q . && nl_fail "port $p is already in use"; done
tmux has-session -t "=wd-editor" 2>/dev/null && nl_fail "tmux session wd-editor exists"
if [ "${DRYRUN:-0}" = "1" ]; then
  H=$(sha256sum "$BASE_ST" | cut -c1-64); [ "$H" = "$BASE_ST_SHA" ] && nl_ok "base 0.8B safetensors sha256 ${H:0:12} (rev 2fc06364)" || nl_fail "base safetensors sha256 ${H:0:12} != ${BASE_ST_SHA:0:12}"
  # CPU proofs, no card: the verdict rule, the published edit bit-exact on our GGUF, the sandbox plumbing end to end with a mock editor, the render from mock arms
  if [ -z "${WE_NO_SELFTEST:-}" ]; then
    ST=$($BPY "$CODE/wd_summary.py" --selftest 2>&1 | grep -E 'BAD|^selftest'); case "$ST" in *BAD*) nl_fail "verdict self-test: $ST";; *"selftest ALL OK"*) nl_ok "verdict rule self-test: 11 branches + line 1 OK";; *) nl_fail "verdict self-test: $ST";; esac
    DT=$(mktemp -d $WD/dry.XXXXXX)
    PUB=$($BPY "$CODE/wd_ref.py" "$WD/ref/materials/edit.npz" "$WD/ref/materials/manifest.json" "$DT/pub.gguf" 2>&1 | tail -1)
    case "$PUB" in *'"layer": 15, "changed": 65260, "sha_ok": true'*'"layer": 19, "changed": 64928, "sha_ok": true'*) nl_ok "published edit rebuilt bit-exact on the GGUF (65,260 + 64,928 entries, both tensor sha256 equal the manifest)";; *) nl_fail "published edit: $PUB";; esac
    rm -f "$DT/pub.gguf"
    mkdir -p "$DT/tables"; $BPY -c "import numpy as np; z=np.zeros((24,3584),dtype=np.float32); np.save('$DT/tables/loop_mean.npy', z); np.save('$DT/tables/prod_mean.npy', z)"
    $RPY "$CODE/wd_mock.py" $MOCK_PORT > "$DT/mock.log" 2>&1 & MP=$!
    for i in $(seq 1 20); do curl -s -m 2 localhost:$MOCK_PORT/health | grep -q ok && break; sleep 0.5; done
    ( export UPSTREAM=http://127.0.0.1:$MOCK_PORT ALIAS=mock-editor SVC_STUB=1 EVALS=3 MINUTES=20 DEV_LIMIT=6 WALL_S=200 FR_QUIET=15 BASE_ROWS=$CODE/mock_base_dev.json TABLES=$DT/tables ARENA=/tmp/arena95-dry FR_PORT=$MOCK_PROXY SVC_PORT=$MOCK_SVC EVAL_PORT=$MOCK_EVAL; bash "$CODE/wd_take.sh" "$DT/take" > "$DT/take.log" 2>&1 ); TRC=$?
    kill $MP 2>/dev/null; wait $MP 2>/dev/null
    GT=$($BPY "$CODE/wd_gate.py" "$DT/take" 2>&1 | tail -1)
    case "$GT" in "GATE scored=1 best=m1"*) nl_ok "sandbox plumbing end to end with a mock editor (take rc $TRC, real Hermes session, wd info/transcripts/apply/eval, gate: $(echo "$GT" | cut -c1-90))";; *) nl_fail "mock take: rc=$TRC $GT / $(tail -2 "$DT/take.log" | tr '\n' ' ' | cut -c1-200)";; esac
    mkdir -p "$DT/ra"; $BPY "$CODE/wd_make_mock.py" "$DT/ra" cured --take "$DT/take" > /dev/null 2>&1
    $BPY "$CODE/wd_summary.py" "$DT/ra" > "$DT/ra/summary.txt" 2>&1; grep -q '^VERDICT EDITOR MATCHED THE SCRIPT' "$DT/ra/summary.txt" && \
      $RPY "$CODE/render_wedit.py" "$DT/ra" "$DT/ra/t.mp4" --ffmpeg "$FF" --scale 0.1 > "$DT/ra/render.log" 2>&1 && [ -s "$DT/ra/t.mp4" ] && nl_ok "mock summary + render ok ($(stat -c %s "$DT/ra/t.mp4") bytes, MOCK-stamped, 0.1x time)" || nl_fail "mock summary/render failed: $(tail -2 "$DT/ra/render.log" 2>/dev/null | tr '\n' ' ')"
    rm -rf "$DT" /tmp/arena95-dry
  fi
  LK=$(grep -nE '/Users/|ghp_[A-Za-z0-9]{10}|hf_[A-Za-z0-9]{20}|100\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|\.ts\.net|@[a-z0-9-]+\.[a-z]{2,}' "$0" "$CODE"/*.py "$CODE/PREREG.md" "$CODE/prompts/edit-task.txt" "$CODE/wd_take.sh" 2>/dev/null | grep -v 'users.noreply' | grep -v 'LEAK = re.compile' | grep -v 'wd_record.py\|wd_proxy.py\|wd_mock.py' | head -3 || true)
  [ -z "$LK" ] && nl_ok "leak scan clean" || nl_fail "leak scan: $LK"
fi
if nl_skip_exact "$ART"; then nl_say "nothing to run, item complete ($ART exists)"; exit 0; fi
DL=$(nl_deadline_epoch); LEFT=$(( (DL-$(date +%s))/60 ))
for ph in "base-dev" "base-held" "published-held" "scripted-held" "take (editor session)" "editor-held"; do nl_say "  RUN $ph -> $ARMS or $TAKE (literal artefacts: arms/<arm>-<set>.json, take/rec/result.json; resumable per arm)"; done
nl_say "  RUN verdict -> $ART ; clip -> $OUT/weight-editor.mp4 ; planned ~126 min of card (hard stop $(date -d "@$DL" '+%a %H:%M') local, $LEFT min away)"
if [ "$SMOKE" != 1 ] && [ -z "${WE_STUB:-}" ]; then
  [ "$LEFT" -ge 40 ] || { nl_say "  under 40 min left in the window: not starting (drain + the arms would be cut short; exit 3, resumable, Donald untouched)"; exit 3; }
fi
if [ "${DRYRUN:-0}" = "1" ] || [ ! -f "$MANIFEST" ]; then nl_manifest_init base_dev tables base_held published_held scripted_held take editor_held summary render; fi
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
cleanup() { tmux kill-session -t "=wd-editor" 2>/dev/null; stop_srv; [ "$DONALD_STOPPED" = 1 ] && { wait_mem_free || true; }; donald_restore; }
trap cleanup EXIT
mkdir -p "$ARMS" "$OUT" "$CAND"
fail() { nl_mark "$1" failed "$2"; nl_say "FAIL: $2"; exit 1; }
DEADLINE=$(nl_deadline_epoch)
REMAIN=0
t0=$(date +%s)
left_min() { echo $(( (DEADLINE-$(date +%s))/60 )); }
free_g() { df --output=avail -BG / | tail -1 | tr -dc 0-9; }
if [ "$SMOKE" = 1 ] && [ "$NODRAIN" = 1 ]; then
  U=$(mem_used); [ "${U:-99999}" -lt 8000 ] || fail base_dev "smoke without drain needs the card under 8 GB used (now ${U} MiB)"; nl_say "smoke without drain: card ${U} MiB used"
else
  FOREIGN=$(nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ' | tr '\n' ' ')
  nl_say "  rule 3, live: compute apps before the drain: ${FOREIGN:-none}; tmux: $(tmux ls -F '#{session_name}' 2>/dev/null | tr '\n' ' ')"
  nl_say "draining Donald (night-lib nl_donald_stop)"; DONALD_STOPPED=1; nl_donald_stop
  for i in $(seq 1 60); do v=$(mem_used); [ "${v:-99999}" -lt 1024 ] && { nl_say "drain clear (${v} MiB)"; break; }; sleep 2; done
  [ "${v:-99999}" -lt 1024 ] || fail base_dev "card memory still ${v:-?} MiB after the drain: another process owns the card, not loading"
fi

gen_arm() { # <arm> <gguf> <set dev|held> <phase>
  local arm=$1 gg=$2 st=$3 ph=$4 lim=0 slots out lft
  out=$ARMS/$arm-$st.json
  if [ -f "$out" ]; then nl_say "  $arm-$st: already done (resume), kept"; nl_mark "$ph" done "kept"; return 0; fi
  lft=$(left_min)
  if [ "$lft" -lt "$ARMMIN" ]; then nl_say "  $arm-$st skipped: $lft min left < $ARMMIN (resumable)"; nl_mark "$ph" skipped "window"; REMAIN=1; return 1; fi
  [ "$(free_g)" -ge 125 ] || { nl_say "  $arm-$st skipped: only $(free_g) G free (< 125)"; nl_mark "$ph" skipped "disk"; REMAIN=1; return 1; }
  slots=20; [ "$st" = held ] && slots=40
  [ "$SMOKE" = 1 ] && { [ "$st" = dev ] && lim=$DEVLIM || lim=$HELDLIM; slots=8; }
  nl_mark "$ph" running
  $BPY "$CODE/wd_gen.py" --gguf "$gg" --set "$st" --out "$out.part" --tag "$arm" --slots $slots --cap $CAP --port $GEN_PORT --limit $lim >> "$RUN/gen.log" 2>&1; local rc=$?
  nl_say "  $arm-$st rc=$rc: $(tail -1 "$RUN/gen.log" | cut -c1-230)"
  if [ "$rc" = 0 ] && [ -s "$out.part" ]; then mv "$out.part" "$out"; nl_mark "$ph" done "$(tail -1 "$RUN/gen.log" | cut -c14-120)"; return 0; fi
  nl_mark "$ph" failed "gen rc=$rc"; REMAIN=1; return 1
}

# ---- A. the base model: dev rows (the editor's evidence) and held-out rows ----
gen_arm base "$BASE_GGUF" dev base_dev || true
if [ -f "$ARMS/base-dev.json" ]; then
  if [ ! -f "$RUN/tables/loop_mean.npy" ]; then
    nl_mark tables running
    $BPY "$CODE/wd_acts.py" tables --dev-rows "$ARMS/base-dev.json" --out "$RUN/tables" --device cuda >> "$RUN/tables.log" 2>&1; nl_say "  tables rc=$?: $(tail -1 "$RUN/tables.log" | cut -c1-200)"
    [ -f "$RUN/tables/loop_mean.npy" ] && nl_mark tables done || { nl_mark tables failed "no tables"; REMAIN=1; }
  else nl_mark tables done "kept"; fi
fi
[ "$SMOKE" = 1 ] || gen_arm base "$BASE_GGUF" held base_held || true
# ---- B. the reference and the baseline: published edit (bit-exact) and the scripted pipeline ----
if [ ! -f "$CAND/published.gguf" ]; then $BPY "$CODE/wd_ref.py" "$WD/ref/materials/edit.npz" "$WD/ref/materials/manifest.json" "$CAND/published.gguf" >> "$RUN/build.log" 2>&1 || { rm -f "$CAND/published.gguf"; nl_say "  published edit did NOT rebuild bit-exact: $(tail -1 "$RUN/build.log" | cut -c1-200)"; }; fi
if [ -f "$RUN/tables/loop_mean.npy" ] && [ ! -f "$CAND/scripted.gguf" ]; then
  $BPY "$CODE/wd_scripted.py" --tables "$RUN/tables" --out "$RUN/scripted-spec.json" >> "$RUN/build.log" 2>&1 && $BPY "$CODE/wd_edit.py" "$RUN/scripted-spec.json" "$CAND/scripted.gguf" >> "$RUN/build.log" 2>&1 || { rm -f "$CAND/scripted.gguf"; nl_say "  scripted candidate build failed: $(tail -1 "$RUN/build.log" | cut -c1-200)"; }
fi
if [ "$SMOKE" != 1 ]; then
  [ -f "$CAND/published.gguf" ] && { gen_arm published "$CAND/published.gguf" held published_held || true; } || { nl_mark published_held failed "no bit-exact gguf"; REMAIN=1; }
  [ -f "$CAND/scripted.gguf" ] && { gen_arm scripted "$CAND/scripted.gguf" held scripted_held || true; } || { nl_mark scripted_held failed "no scripted gguf"; REMAIN=1; }
fi

# ---- C. the editor take: Qwen3.8-27B in a real Hermes Agent session, driving the sandbox tool ----
serve_editor() { # sets SP
  local t1=$(date +%s) i
  "$SRV" -m "$G27" --port $ED_PORT --host 127.0.0.1 -ngl 99 -fa on --jinja -c 65536 --parallel 1 -ctk q8_0 -ctv q8_0 -cram 0 --alias editor >> "$RUN/server-editor.log" 2>&1 &
  SP=$!
  for i in $(seq 1 180); do curl -s -m 3 http://127.0.0.1:$ED_PORT/health | grep -qi ok && { nl_say "  editor llama-server up after $(( $(date +%s)-t1 )) s (VRAM $(mem_used) MiB)"; return 0; }; kill -0 "$SP" 2>/dev/null || break; sleep 2; done
  return 1
}
chat_smoke() { python3 - "$1" <<'PYEOF'
import json, sys, time, urllib.request
body = {"model": "editor", "messages": [{"role": "user", "content": "What time is it in Paris? Use the tool."}], "max_tokens": 3000, "temperature": 0.6, "stream": False,
        "tools": [{"type": "function", "function": {"name": "get_time", "description": "current time in a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]}
t = time.time()
try:
    r = json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions" % sys.argv[1], json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=240).read())
except Exception as e:
    print("SMOKE FAIL request: %s" % str(e)[:160]); sys.exit(1)
tc = r["choices"][0]["message"].get("tool_calls") or []
ok = bool(tc) and tc[0]["function"]["name"] == "get_time"
print(("SMOKE OK" if ok else "SMOKE FAIL") + " tool_calls=%d %.1fs" % (len(tc), time.time() - t)); sys.exit(0 if ok else 1)
PYEOF
}
TAKE_DONE=0
[ -f "$TAKE/gate.json" ] && TAKE_DONE=1
if [ "$TAKE_DONE" = 1 ]; then nl_say "  take: already done (resume), kept"; nl_mark take done "kept"
elif [ ! -f "$ARMS/base-dev.json" ] || [ ! -f "$RUN/tables/loop_mean.npy" ]; then nl_say "  take skipped: no base dev rows or tables yet"; nl_mark take skipped "prerequisites"; REMAIN=1
elif [ "$(left_min)" -lt "$TAKEMIN" ]; then nl_say "  take skipped: $(left_min) min left < $TAKEMIN (a cut take is not resumable; next block)"; nl_mark take skipped "window"; REMAIN=1
else
  nl_mark take running
  if serve_editor; then
    SM=$(chat_smoke $ED_PORT); nl_say "  editor tool-call smoke: $SM"
    case "$SM" in
      "SMOKE OK"*)
        [ -d "$TAKE" ] && mv "$TAKE" "$TAKE.partial-$(date +%s)"
        W=$WALL; l=$(( $(left_min) * 60 - 600 )); [ "$l" -lt "$W" ] && W=$l
        nl_say "=== editor take (Hermes Agent + Qwen3.8-27B, wall cap $W s, $(left_min) min left, evals $EVALS, tool clock $MINUTES min) ==="
        UPSTREAM=http://127.0.0.1:$ED_PORT ALIAS=editor WALL_S=$W CTXCFG=65536 FR_PORT=$PROXY_PORT SVC_PORT=$SVC_PORT EVAL_PORT=$EVAL_PORT EVALS=$EVALS MINUTES=$MINUTES DEV_LIMIT=$DEVLIM EVAL_CAP=$CAP EVAL_SLOTS=$([ "$SMOKE" = 1 ] && echo 8 || echo 20) \
          BASE_ROWS=$ARMS/base-dev.json TABLES=$RUN/tables ARENA=/tmp/arena95 ACTS_DEVICE=cpu bash "$CODE/wd_take.sh" "$TAKE" >> "$RUN/take.log" 2>&1; trc=$?
        nl_say "  take rc=$trc: $(tail -2 "$RUN/take.log" | tr '\n' ' ' | cut -c1-240)"
        curl -s -m 5 "http://127.0.0.1:$ED_PORT/slots" > "$RUN/slots-end-editor.json" 2>/dev/null
        $BPY "$CODE/wd_gate.py" "$TAKE" > "$RUN/gate.log" 2>&1; nl_say "  $(tail -1 "$RUN/gate.log" | cut -c1-240)"
        [ -f "$TAKE/gate.json" ] && nl_mark take done "$(tail -1 "$RUN/gate.log" | cut -c1-100)" || { nl_mark take failed "no gate.json"; REMAIN=1; };;
      *) nl_say "  editor tool-call smoke failed: take BROKEN, not run"; nl_mark take failed "tool-call smoke"; REMAIN=1;;
    esac
  else nl_say "  editor llama-server did not come up (see $RUN/server-editor.log)"; nl_mark take failed "no load"; REMAIN=1; fi
  stop_srv; wait_mem_free || fail take "card not clear after the editor stop"
fi

# ---- D. the editor's final candidate on the held-out problems ----
BEST=$($BPY -c "import json;g=json.load(open('$TAKE/gate.json'));print(g.get('best_gguf') or '')" 2>/dev/null)
if [ -n "$BEST" ] && [ -f "$BEST" ]; then
  [ -f "$ARMS/editor-held.json" ] || cp -a "$BEST" "$CAND/editor-best.gguf"
  gen_arm editor "$CAND/editor-best.gguf" held editor_held || true
elif [ -f "$TAKE/gate.json" ]; then nl_say "  no scored candidate: no editor-held arm (verdict NO CANDIDATE)"; nl_mark editor_held skipped "no candidate"
else nl_mark editor_held skipped "no take"; fi
# the card is free from here: Donald back, then the CPU-only legs
stop_srv; wait_mem_free || true; donald_restore
nl_say "card released after $(( ($(date +%s)-t0)/60 )) min (drain to restore), the rest is CPU"
# keep the specs and the best candidate, drop the other candidate GGUFs (disk)
if [ -f "$TAKE/gate.json" ]; then for g in "$TAKE"/state/cand/*.gguf; do [ -f "$g" ] && [ "$g" != "$BEST" ] && rm -f "$g"; done; fi
if [ "$SMOKE" = 1 ]; then
  nl_say "SMOKE done: $(ls "$ARMS" | tr '\n' ' '); gate: $(tail -1 "$RUN/gate.log" 2>/dev/null | cut -c1-200)"; exit 0
fi
nl_mark summary running
$BPY "$CODE/wd_summary.py" "$RUN" > "$OUT/summary.txt" 2>&1; src=$?
nl_say "  summary rc=$src: $(grep -m1 '^VERDICT' "$OUT/summary.txt" | cut -c1-220)"
if [ "$src" = 0 ] && [ "$REMAIN" = 0 ] && [ -f "$ART" ]; then
  nl_mark summary done "$(grep -m1 '^VERDICT' "$OUT/summary.txt" | cut -c9-70)"
  grep -m1 '^LINE1' "$OUT/summary.txt" | cut -c7- > "$OUT/line1.txt"
  { echo "# weight editor readout"; echo; grep -E '^(VERDICT)' "$OUT/summary.txt"; echo; grep '^SUMMARY' "$OUT/summary.txt"; echo; echo "regime: editor Qwen3.8-27B Q6_K (llama.cpp $SRV_SHA, ctx 65,536, KV q8_0) in Hermes Agent $HERMES_PIN; 0.8B Qwen3.5-0.8B rev 2fc06364 bf16 GGUF; GSM8K dev 20 (train) / held-out 40 (test), cap $CAP, thinking on, temp 0.6 top-p 0.95 top-k 20 presence 1.5; one try per problem and arm; take: $(cat "$TAKE/gate.json" | tr -d '\n ' | cut -c1-300)"; } > "$OUT/readout.md"
  if [ -z "${WE_NO_RENDER:-}" ]; then
    nl_mark render running
    $RPY "$CODE/render_wedit.py" "$RUN" "$OUT/weight-editor.mp4" --ffmpeg "$FF" --stills "$OUT/stills" >> "$OUT/render.log" 2>&1; rr=$?
    $FF -nostdin -hide_banner -loglevel error -y -ss 0.5 -i "$OUT/weight-editor.mp4" -frames:v 1 "$OUT/contact-1.png" </dev/null
    $FF -nostdin -hide_banner -loglevel error -y -ss 30 -i "$OUT/weight-editor.mp4" -frames:v 1 "$OUT/contact-2.png" </dev/null
    nl_judge_artifact render "$rr" "$OUT/weight-editor.mp4" "$t0" "clip" || nl_say "  render failed (see $OUT/render.log); the verdict stands"
  fi
  nl_say "ANGLE: $(cat "$OUT/line1.txt")"
  nl_say "DONE $SLUG"
  exit 0
fi
nl_say "  a phase could not finish inside the window: exit 3, resumable, summary.json NOT written"
exit 3
