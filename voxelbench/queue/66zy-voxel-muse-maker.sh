#!/bin/bash
# 66zy-voxel-muse-maker.sh: idle-queue item (2026-10-22 ~18:40 CEST, mercury, GPU thread; operator: "that is also part of our bench no?").
# fairness rerun of 66zz (Muse Glimmer 30B, msm-v2, Hermes /goal quality gate) with ONE change: the MAKER's sampling from the model card
# (huggingface.co/meta-models/Muse-Glimmer-30B-GGUF, fetched 2026-10-22: "--temp 1.0 --top-p 0.95 --top-k 64", "Sampling defaults. temperature
# 1.0, top_p 0.95, top_k 64"); the GGUF has no general.sampling keys, so 66y/66zz ran llama.cpp defaults (temp 0.8, top-k 40, min-p 0.05).
# long agentic task pilot: Hermes v0.22.2 goal mode drives ONE local model through a long visual build (voxel diorama,
# single HTML file, WebGL), then a headless-chromium gate on the 5090 checks the result and records a 20 s clip.
# prompt: a community member's reply to the Muse Glimmer post (verbatim) + one harness line naming the output file.
# harness (all smoke-tested 2026-10-21 09:16-09:32 CEST): ~/voxelbench/scripts/run_agent.sh (tmux classic CLI, fresh
# bench profile per run, /goal, polls goal state in the profile state.db), gate.py (ANGLE GL/EGL on the 5090; ANGLE/Vulkan
# loses the context), metrics.py. smoke2: goal judged done in 1 turn / 96 s, file "verified" by the model (read back +
# node --check) yet rendered blank on GPU and SwiftShader alike; the gate failed it (not_blank, animated). that is the point.
# pilot: Qwen3.8-27B Q6_K (Donald's file), llama.cpp 839a572 (Donald's build since 2026-10-21), MTP n-max 2, q8_0 KV, fa,
# ctx 131072 (27,654 MiB at load, measured), goal budget 20 turns, wall 60 min.
# artifact: runs/<slug>/gate/gate.json (exact). bench profiles are NEVER deleted (host gateway holds their state.db).
set -uo pipefail
NL_TAG=voxmusem
MANIFEST=/home/witcheer/voxel-muse-maker-status.json
PVER=msm-v2
export MANIFEST
. /home/witcheer/night-lib.sh
V=/home/witcheer/voxelbench
BIN=/home/witcheer/llama.cpp-donald/build/bin/llama-server
GPY=/home/witcheer/benchmark-rig/.venv/bin/python
RUNLOG=/home/witcheer/voxel-pilot-run.log
SLUG=muse-glimmer-30b-q4km-$PVER-maker
SAMPLING="--temp 1.0 --top-p 0.95 --top-k 64"   # card values; min-p left at the llama.cpp default 0.05, exactly like the Qwen v2 reference (its GGUF carries temp 1.0 / top-p 0.95 / top-k 20, no min-p)
MODEL_FILE=/home/witcheer/models/muse-glimmer/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf
ALIAS=Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf
CTX=131072; WALL=60; TURNS=20
# hardening 2026-10-21 14:30 CEST (after the 66v pilot spent 90k+ tokens of one thinking block drafting the whole
# program before any write_file): cap thinking per request, same value as the board's think-on 16k budget.
RB=16384
RB_MSG="Thinking budget reached. Stop planning and write the files now with your tools; refine in later turns."
ART=$V/runs/$SLUG/gate/gate.json

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0})"
nl_check_path "$BIN" "$GPY" "$MODEL_FILE" "$V/scripts/run_agent.sh" "$V/scripts/gate.py" "$V/scripts/metrics.py" \
  "$V/prompts/$PVER.txt" "$V/scripts/orbit_clip.py" "$V/lib/libasound.so.2" "$V/ffmpeg" /home/witcheer/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome
HELP=$($BIN --help 2>&1 || true); case "$HELP" in *--reasoning-budget-message*) nl_say "  ok: --reasoning-budget(-message) in build";; *) nl_fail "--reasoning-budget-message missing from build";; esac
$BIN --version 2>&1 | grep -q "839a572" && nl_say "  ok: llama.cpp 839a572" || { nl_say "  llama.cpp build changed or missing"; exit 2; }
nl_check_disk 20
if nl_skip_exact "$ART"; then nl_say "nothing to run, item complete"; exit 0; fi
DL=$(nl_deadline_epoch); LEFT=$(( (DL-$(date +%s))/60 ))
nl_say "  planned RUN: $SLUG (wall $WALL min + ~3 min gate; hard stop $(date -d "@$DL" '+%a %H:%M') local, $LEFT min away)"
[ "$LEFT" -ge 75 ] || { nl_say "  under 75 min left in the window: not starting (would cut the run short)"; exit 3; }
if [ "${DRYRUN:-0}" = "1" ] || [ ! -f "$MANIFEST" ]; then nl_manifest_init "$SLUG"; fi
nl_gate

restore() { tmux kill-session -t "=voxel-$SLUG" 2>/dev/null; pkill -9 -x llama-server 2>/dev/null; sleep 5; nl_say "EXIT trap: restoring Donald"; nl_donald_restore; }
trap restore EXIT
DEADLINE=$(nl_deadline_epoch)
nl_say "draining Donald"; nl_donald_stop
for i in $(seq 1 120); do v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${v:-99999}" -lt 4000 ] && { nl_say "drain clear (${v} MiB)"; break; }; sleep 2; done
[ "$(ls -A $V/runs/$SLUG/work 2>/dev/null)" ] && { nl_say "  stale work dir from an earlier attempt: moving aside"; mv $V/runs/$SLUG $V/runs/$SLUG.stale-$(date +%s); }

t0=$(date +%s); nl_mark "$SLUG" running
$BIN --model "$MODEL_FILE" --host 127.0.0.1 --port 8090 --n-gpu-layers 99 --ctx-size $CTX --parallel 1 --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 --alias "$ALIAS" --metrics --reasoning-budget $RB --reasoning-budget-message "$RB_MSG" $SAMPLING \
  > $V/runs-$SLUG-server.log 2>&1 &
for i in $(seq 1 120); do curl -s 127.0.0.1:8090/health | grep -qi ok && break; sleep 2; done
curl -s 127.0.0.1:8090/health | grep -qi ok || { nl_mark "$SLUG" failed "server did not come up"; exit 1; }
curl -s 127.0.0.1:8090/props > $V/runs-$SLUG-props.json 2>/dev/null; nl_say "  sampling applied: $(python3 -c "import json;g=json.load(open('$V/runs-$SLUG-props.json')).get('default_generation_settings',{});p=g.get('params',g);print({k:p.get(k) for k in ('temperature','top_k','top_p','min_p')})" 2>/dev/null)"
nl_say "server up ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader)), ctx $CTX"
export DEADLINE_EPOCH=$(( DEADLINE - 300 ))
mkdir -p $V/runs/$SLUG
PROMPT_RUN=$V/runs/$SLUG/prompt.txt
{ cat $V/prompts/$PVER.txt; printf '\nThe working directory is %s. The final file must be exactly %s/index.html (copy this path, never retype it).\n' "$V/runs/$SLUG/work" "$V/runs/$SLUG/work"; } > "$PROMPT_RUN"
GATE_CMD="$GPY $V/scripts/gate.py $V/runs/$SLUG/work/index.html $V/runs/$SLUG/qgate --video-seconds 0" bash $V/scripts/run_agent.sh "$SLUG" "$ALIAS" $CTX $WALL $TURNS "$PROMPT_RUN" >> "$RUNLOG" 2>&1
nl_say "driver: $(cat $V/runs/$SLUG/driver.json 2>/dev/null)"
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
    setsid timeout 240 $GPY $V/scripts/orbit_clip.py "$HTML" $V/runs/$SLUG/showcase --seconds 30 --keys "n@11,n@22" < /dev/null >> "$RUNLOG" 2>&1 \
      && nl_say "  showcase: $V/runs/$SLUG/showcase/orbit.mp4" || nl_say "  showcase clip failed (gate result stands)"
  fi
fi
exit 0
