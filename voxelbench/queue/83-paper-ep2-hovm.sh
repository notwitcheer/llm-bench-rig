#!/bin/bash
# 83-paper-ep2-hovm.sh: idle-queue item (STAGED in ~/voxel-stage, not enqueued; 2026-11-03, mercury subagent for the GPU thread; operator yes 2026-11-03 ~09:35 Paris: stage paper-to-demo episode 2).
# WHAT: paper-to-demo EPISODE 2 on the trees h4 harness. Qwen3.8-27B Q6_K gets ONE Hermes Agent /goal hour (60 min wall, 20 goal turns) and prompts/hovm-v1.txt: rebuild the flocking model of arXiv 2608.10844
#   (Masoumi, Kargaran, Jafari, 11 Aug 2026, "Impact of Higher-Order Interactions on Collective Motion") as a two-panel browser demo, standard Vicsek (pairwise) left, the paper's higher-order Vicsek model
#   (triadic) right, same birds and same noise. The paper's claim: the pairwise flock loses order smoothly as noise grows, the triadic flock stays ordered and then collapses abruptly (a first-order transition) at a lower noise.
# MOJO: two flocks of 1,000 birds side by side under the same noise, the pairwise one holds its shape and the triadic one falls apart into a rainbow gas on cue | esoteric (a 12-week-old preprint, reproduced by a local model in one hour; the authors' own text and code disagree on one rule, see the vault note) + intense (the collapse is visible in ~2 s of the clip) | line 1 = the pre-registered variants below
# EV: line 1 as it would post (gate PASS): "a local model on one RTX 5090 rebuilt a 2026 flocking paper in one Hermes Agent hour: same 1,000 birds, same noise, the pairwise flock holds and the triadic one falls apart [figure from the paper: order vs noise]".
#   gate FAIL variants (written before the run): "... the model's triadic flock never collapsed (the paper's claim did not show in its page): <first failed check>" / "... the page never ran". gpu-ev-gate-2026-10-25 section 2 rule 4 (reproducible, fully local Hermes piece)
#   + the esoteric line; comparables: first voxel post 5,549 views, voxel sequels ~2k, trees episode (the series anchor, posts Tue 2026-11-03 18:30). bracket 2-6k (a NEW picture and a new paper, but a series sequel in format).
#   X only on a final-gate PASS with a watchable clip, otherwise a board line / TG + repo. NOT blind: the model is told the paper's claim and checks it itself (operator pick A, as in trees h4); say so in every post and board line.
# serves: content (a Hermes-driven showcase, paper to demo, franchise F1 episode 2) + experiment (can a local model rebuild a checkable physics claim from a precise spec in one hour)
# PRE-REGISTERED (written 2026-11-03 before any agent run of this prompt): the FINAL gate lives in ~/.vxfinal and is named on no path the agent can read; the thresholds and their sources are in the vault note
#   99-personal/benchmark-rig/paper-ep2-status.md, never in this file. the in-run gate (scripts/quick_gate_flock.py) checks the DIRECTION of the claim only (pick A) plus liveness at other noise values.
#   extras 2 and 3 (other seeds, the transition window, random-start runs) are reported, not gating: the anti-tuning check.
# LINE-1 VARIANTS: V1 gate PASS: "<model> rebuilt a 2026 flocking paper in one Hermes /goal hour on one RTX 5090: same 1,000 birds, same noise, the pairwise flock holds (order <svm_hi>) and the triadic flock falls apart (order <hovm_hi>)".
#   V2 gate FAIL on the claim: "... its triadic flock <still holds / fell apart at the wrong noise>: <first failed check, numbers>". V3 no page: "... did not produce a page that runs: <reason>". the brackets are filled from gate.json by hand-free readout, never reworded.
# MODEL / SAMPLING / SERVER: identical to 72fa (Qwen3.8-27B Q6_K, temp 1.0 / top-p 0.95 / top-k 20 / min-p 0.05, MTP n-max 2, ctx 131072 fit ladder, q8_0 KV, RB 16384, llama.cpp 839a572 pin). 60 min wall, 20 goal turns,
#   VX_GATE_STRICT=1, VX_STALL_NUDGE=1, >= 90 min left in the window or exit 3. card time ~75 min (drain + load + 60 + gate/clip on CPU), the gate and the clip use no GPU.
# CLIP PLAN (planned in the item, no rescue recut): flock_clip.py drives the page's OWN hooks on an absolute step schedule (flockReset / setEta / flockStep), one screenshot per frame, fixed camera = the page's own two panels at the same
#   scale: eta 1.0 for 6 s then eta 2.2 for 14 s (the same noise jump on both flocks), 1920x1080, caption band, contact sheet + a full-res last frame for the HUD leak read. clip + sheet land in runs/<slug>/showcase-flock/.
# artifact: runs/<slug>/gate/gate.json (exact). bench profiles are NEVER deleted.
set -uo pipefail
NL_TAG=voxflock83
MANIFEST=/home/witcheer/voxel-voxflock83-status.json
export MANIFEST
. /home/witcheer/night-lib.sh
V=/home/witcheer/voxelbench
BIN=/home/witcheer/llama.cpp-donald/build/bin/llama-server
GPY=/home/witcheer/benchmark-rig/.venv/bin/python
QG=$V/scripts/quick_gate_flock.py            # the in-run gate the agent can read (direction of the claim at seed 1 by design, pick A; no thresholds beyond that)
FINAL=/home/witcheer/.vxfinal/gate_flock.py   # the final gate, outside the agent's tree (HARDENING L33)
RUNLOG=/home/witcheer/voxel-pilot-run.log
SLUG=flock-hovm-qwen38-27b-q6k
MODEL_FILE=/home/witcheer/models/qwen38-27b/Qwen3.8-27B-Q6_K.gguf
ALIAS=Qwen3.8-27B-Q6_K.gguf
CTX=131072; WALL=60; TURNS=20; KV=q8_0
# maker sampling (HARDENING L5/L6; the same row is in voxelbench/model-params.md): the flags pin the card values over the GGUF keys
SAMPLING_FLAGS="--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05"
PARAMS_SRC="GGUF general.sampling keys of the maker file (temp 1.0, top_p 0.95, top_k 20), read 2026-10-22, voxelbench/model-params.md row Qwen3.8-27B Q6_K; no min_p key, so llama.cpp default 0.05; RB 16384 (series value)"
EXTRA_ARR=(--spec-type draft-mtp --spec-draft-n-max 2)
# hardening 2026-10-21 14:30 CEST (after the 66v pilot spent 90k+ tokens of one thinking block drafting the whole
# program before any write_file): cap thinking per request, same value as the board's think-on 16k budget.
RB=16384
RB_MSG="Thinking budget reached. Stop planning and write the files now with your tools; refine in later turns."
ART=$V/runs/$SLUG/gate/gate.json

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0})"
nl_check_path "$BIN" "$GPY" "$MODEL_FILE" "$V/scripts/run_agent.sh" "$QG" "$FINAL" "$V/scripts/vx_stall.py" "$V/scripts/metrics.py" \
  "$V/prompts/hovm-v1.txt" "$V/scripts/flock_clip.py" "$V/lib/libasound.so.2" "$V/ffmpeg" /home/witcheer/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome
# L33 enforcement (h4): the final gate stays in ~/.vxfinal; its other check names must not sit on any agent path. the lambda criterion is visible by design (pick A).
# the patterns are assembled from pieces so this scan does not match its own file.
[ ! -e "$V/scripts/gate_flock.py" ] || nl_fail "final gate still in the agent's tree: $V/scripts/gate_flock.py (move it to ~/.vxfinal)"
LEAKS=$(grep -rIl --exclude-dir=runs --exclude-dir=deb --exclude-dir=extract --exclude-dir=lib --exclude-dir=previews --exclude-dir=__pycache__ --exclude="runs-*" -e "hovm_""collapsed" -e "svm_""smooth" -e "gap_at_""22" -e "hovm_""ordered_low" -e "svm_""ordered_low" "$V" /home/witcheer/idle-queue/*.sh 2>/dev/null || true)
[ -z "$LEAKS" ] || nl_fail "pre-registered criteria visible to the agent in: $(echo $LEAKS | tr '\n' ' ')"
nl_say "  ok: criteria not visible on the agent paths ($V, ~/idle-queue/*.sh); final gate $FINAL"
"$GPY" -c "import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$QG" "$FINAL" "$V/scripts/vx_stall.py" "$V/scripts/flock_clip.py" || nl_fail "gate script syntax"
bash -n "$V/scripts/run_agent.sh" || nl_fail "run_agent.sh syntax"
HELP=$($BIN --help 2>&1 || true); case "$HELP" in *--reasoning-budget-message*) nl_say "  ok: --reasoning-budget(-message) in build";; *) nl_fail "--reasoning-budget-message missing from build";; esac
for f in $SAMPLING_FLAGS "${EXTRA_ARR[@]}"; do case "$f" in --*) case "$HELP" in *"$f"*) ;; *) nl_fail "flag $f not in this build";; esac;; esac; done; nl_say "  ok: sampling/extra flags exist in build: $SAMPLING_FLAGS ${EXTRA_ARR[*]}"
[ -n "$PARAMS_SRC" ] || nl_fail "PARAMS_SRC empty: identify the maker parameters first (voxelbench/model-params.md)"
GG_SAMP=$($GPY - "$MODEL_FILE" <<'PYEOF' 2>/dev/null
import sys
try:
    from gguf import GGUFReader
except ImportError:
    sys.path.insert(0, "/home/witcheer/llama.cpp-donald/gguf-py"); from gguf import GGUFReader
r = GGUFReader(sys.argv[1]); out = []
for k, f in r.fields.items():
    if k.startswith("general.sampling."):
        try: v = f.parts[f.data[0]].tolist()
        except Exception: v = "?"
        out.append(k.split(".")[-1] + "=" + str(v[0] if isinstance(v, list) and v else v))
print(" ".join(out))
PYEOF
)
nl_say "  params: src='$PARAMS_SRC' gguf_sampling='${GG_SAMP:-none}' flags='$SAMPLING_FLAGS'"
[ -n "$SAMPLING_FLAGS" ] || nl_fail "SAMPLING_FLAGS empty"
$BIN --version 2>&1 | grep -q "839a572" && nl_say "  ok: llama.cpp 839a572" || { nl_say "  llama.cpp build changed or missing"; exit 2; }
nl_check_disk 20
if nl_skip_exact "$ART"; then nl_say "nothing to run, item complete"; exit 0; fi
DL=$(nl_deadline_epoch); LEFT=$(( (DL-$(date +%s))/60 ))
nl_say "  planned RUN: $SLUG (wall $WALL min + ~15 min gate+clips; hard stop $(date -d "@$DL" '+%a %H:%M') local, $LEFT min away)"
[ "$LEFT" -ge 90 ] || { nl_say "  under 90 min left in the window: not starting (60 min wall + ~15 min gate/clips; would cut the run short)"; exit 3; }
if [ "${DRYRUN:-0}" = "1" ] || [ ! -f "$MANIFEST" ]; then nl_manifest_init "$SLUG"; fi
nl_gate

restore() { tmux kill-session -t "=voxel-$SLUG" 2>/dev/null; pkill -9 -x llama-server 2>/dev/null; sleep 5; nl_say "EXIT trap: restoring Donald"; nl_donald_restore; }
trap restore EXIT
DEADLINE=$(nl_deadline_epoch)
nl_say "draining Donald"; nl_donald_stop
for i in $(seq 1 120); do v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${v:-99999}" -lt 4000 ] && { nl_say "drain clear (${v} MiB)"; break; }; sleep 2; done
[ "$(ls -A $V/runs/$SLUG/work 2>/dev/null)" ] && { nl_say "  stale work dir from an earlier attempt: moving aside"; mv $V/runs/$SLUG $V/runs/$SLUG.stale-$(date +%s); }

mkdir -p "$V/runs/$SLUG/work"   # the exact work dir exists before the agent starts (run_agent.sh also makes it)
t0=$(date +%s); nl_mark "$SLUG" running; UP=0; CTX_REQ=$CTX
# fit ladder copied from day0-voxel.sh: requested ctx first, then smaller, never under the 64K Hermes Agent minimum; the ctx used is logged + written to run-config.json
for c in $CTX_REQ 98304 65536; do
  [ "$c" -le "$CTX_REQ" ] || continue
  # shellcheck disable=SC2086
  $BIN --model "$MODEL_FILE" --host 127.0.0.1 --port 8090 --n-gpu-layers 99 --ctx-size $c --parallel 1 --flash-attn on \
    --cache-type-k $KV --cache-type-v $KV --alias "$ALIAS" --metrics --reasoning-budget $RB --reasoning-budget-message "$RB_MSG" $SAMPLING_FLAGS "${EXTRA_ARR[@]}" \
    > $V/runs-$SLUG-server.log 2>&1 &
  for i in $(seq 1 150); do curl -s 127.0.0.1:8090/health | grep -qi ok && { UP=1; break; }; kill -0 $! 2>/dev/null || break; sleep 2; done
  [ "$UP" = 1 ] && { CTX=$c; break; }
  nl_say "  server did not come up at ctx $c: $(grep -m1 -iE 'out of memory|failed|error|unknown' $V/runs-$SLUG-server.log | head -c 200)"
  pkill -9 -x llama-server 2>/dev/null; sleep 4
done
[ "$UP" = 1 ] || { nl_mark "$SLUG" failed "server did not come up at any ctx (see runs-$SLUG-server.log)"; exit 1; }
nl_say "server up ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader)), ctx $CTX (requested $CTX_REQ)"
mkdir -p $V/runs/$SLUG
curl -s 127.0.0.1:8090/props > $V/runs/$SLUG/props.json 2>/dev/null
nl_say "  server sampling defaults: $(python3 -c "import json;g=json.load(open('$V/runs/$SLUG/props.json')).get('default_generation_settings',{});p=g.get('params',g);print({k:p.get(k) for k in ('temperature','top_k','top_p','min_p','n_predict')})" 2>/dev/null)"
VER=$($BIN --version 2>&1 | grep -m1 -i 'version' | tr -d '"')
echo "{\"model_file\": \"$MODEL_FILE\", \"ctx\": $CTX, \"ctx_requested\": $CTX_REQ, \"kv\": \"$KV\", \"extra\": \"${EXTRA_ARR[*]}\", \"sampling\": \"$SAMPLING_FLAGS\", \"gguf_sampling\": \"${GG_SAMP:-}\", \"params_src\": \"$PARAMS_SRC\", \"reasoning_budget\": $RB, \"prompt\": \"hovm-v1.txt\", \"build\": \"$VER\", \"vram_used\": \"$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)\"}" > $V/runs/$SLUG/run-config.json
export DEADLINE_EPOCH=$(( DEADLINE - 300 ))
mkdir -p $V/runs/$SLUG
PROMPT_RUN=$V/runs/$SLUG/prompt.txt
{ cat $V/prompts/hovm-v1.txt; printf '\nThe working directory is %s. The final file must be exactly %s/index.html (copy this path, never retype it).\n' "$V/runs/$SLUG/work" "$V/runs/$SLUG/work"; } > "$PROMPT_RUN"
# h4 harness (h3 + L39-L40): socket-safe TMPDIR for the model's terminal (HARDENING L9) + Hermes /goal quality gate with the held-server setup (L3/L4)
VX_TMPFIX=1 VX_GATE_STRICT=1 VX_STALL_NUDGE=1 GATE_CMD="$GPY $QG $V/runs/$SLUG/work/index.html $V/runs/$SLUG/qgate --video-seconds 0" bash $V/scripts/run_agent.sh "$SLUG" "$ALIAS" $CTX $WALL $TURNS "$PROMPT_RUN" >> "$RUNLOG" 2>&1
nl_say "driver: $(cat $V/runs/$SLUG/driver.json 2>/dev/null)"
[ -f "$V/runs/$SLUG/path-miss.json" ] && nl_say "  PATH MISS flag present: $(cat $V/runs/$SLUG/path-miss.json)"
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
  setsid timeout 900 $GPY $FINAL "$HTML" $V/runs/$SLUG/gate --video-seconds 20 --budget 780 < /dev/null >> "$RUNLOG" 2>&1; grc=$?
else
  echo '{"pass": false, "no_html": true}' > "$ART"; grc=1
fi
python3 $V/scripts/metrics.py /home/witcheer/.hermes/profiles/$(python3 -c "import json;print(json.load(open('$V/runs/$SLUG/driver.json'))['profile'])" 2>/dev/null || echo vx${SLUG//[^a-z0-9]/}) $V/runs/$SLUG/metrics.json >> "$RUNLOG" 2>&1
if nl_judge_artifact "$SLUG" 0 "$ART" "$t0" "$(( ($(date +%s)-t0)/60 )) min, gate rc $grc"; then
  nl_say "  gate: $(python3 -c "import json;d=json.load(open('$ART'));print('pass' if d.get('pass') else 'FAIL', d.get('checks'), 'fps', d.get('fps'), 'bytes', d.get('bytes'), 'main_thread', json.dumps(d.get('main_thread')))")"
  nl_say "  flock: $(python3 -c "import json;d=json.load(open('$ART'));f=d.get('flock',{});print('eta1.0 [svm,hovm]', json.dumps(f.get('eta1.0')), 'eta2.2 [svm,hovm]', json.dumps(f.get('eta2.2')), 'determinism', json.dumps(f.get('determinism')), 'extra seeds holding', (d.get('extras') or {}).get('seeds_holding'), 'canvas diff L/R', d.get('frame_diff_left'), d.get('frame_diff_right'), 'elapsed_s', d.get('elapsed_s'))" 2>/dev/null)"
  # the clip is cut when the page renders and the hooks answer, even if a paper check failed (a failed claim is still the film)
  if python3 -c "import json,sys;d=json.load(open('$ART'));c=d.get('checks') or {};sys.exit(0 if c.get('no_page_errors') and c.get('canvas_present') and c.get('flockrun_hook') and c.get('main_thread_ok') else 1)"; then
    setsid timeout 420 $GPY $V/scripts/flock_clip.py "$HTML" $V/runs/$SLUG/showcase-flock --seed 1 --eta-lo 1.0 --eta-hi 2.2 --switch 6 --seconds 20 < /dev/null >> "$RUNLOG" 2>&1 \
      && nl_say "  flock clip: $V/runs/$SLUG/showcase-flock/flock.mp4 (sheet flock_sheet.png, last frame flock_last.png for the HUD leak read)" || nl_say "  flock clip failed (gate result stands)"
  else
    nl_say "  base checks or hooks failed: no flock clip"
  fi
fi
exit 0

