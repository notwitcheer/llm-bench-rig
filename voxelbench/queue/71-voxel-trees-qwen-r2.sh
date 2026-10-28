#!/bin/bash
# R2 (2026-10-28 ~13:15 CET, mercury; operator, GPU thread: "yes go with 1"): rerun of 71 after its harness fail (Wed 03:17 CET: the 6 KB goal was typed with a fixed 0.5 s wait,
# Hermes read the Enter as a pasted newline (<50 ms rule), "/goal gate add" got glued onto the goal text, no gate, 0 goal turns). fix = run_agent.sh type_line (HARDENING L31),
# proven on the CPU tiny probe with this exact prompt (old: glued, gates []; new: gate registered before turn 1). VX_GATE_STRICT=1: a failed gate setup now aborts in ~1 min instead
# of burning the hour. everything else (model, flags, prompt trees-v1, gate, wall, turns, pre-registered criteria) is unchanged from 71.
# 71-voxel-trees-qwen.sh: idle-queue item, TEST RUN (2026-10-26 ~11:40 CET, mercury subagent; operator, GPU thread: "let's set it up and see what the output is, as a test").
# Voxel-series episode with a new hook: paper to working demo. Hermes Agent on the series reference model (Qwen3.8-27B Q6_K) gets ONE prompt (prompts/trees-v1.txt)
# and one hour of /goal to implement the SIGGRAPH 2009 paper "Self-organizing tree models for image synthesis" (Palubicki et al., https://www.algorithmicbotany.org/papers/selforg.sig2009.html)
# as a single-page 3D browser demo where a tree grows season by season (metamers/buds, shadow-propagation voxel grid for light Q and direction V, extended Borchert-Honda
# resource allocation with lambda, shoot growth floor(v) with a cap, tropism + noise, phyllotaxis option, shedding, pipe-model girth, foliage). Gate = gate_trees.py (series base
# checks + treeStats() hooks + the paper's lambda crown-shape effect), clip = growth_clip.py (seasons 2 -> 30 timelapse, fixed camera, then a short orbit).
# EV: line 1 as it would post: "a local model on one GPU turned a 2009 SIGGRAPH tree-growth paper into a working 3D demo in an hour" (gate rule 4 fully-local Hermes piece, reproducible, plus a new hook: paper-to-working-demo; comparable: first voxel post 5,549 views, voxel sequels ~2k; a weak or failed result is a TG/repo line, not an X post).
# serves: content (Hermes-driven showcase, new hook)
# HARNESS: h2 (the most recent comparable voxel runs 66zz1 / 67e1 / 67e2 / 67f are all h2 on this model): VX_TMPFIX=1 (short TMPDIR) + a Hermes /goal quality gate registered before turn 1. The quality gate is gate_trees.py --quick
# (base checks + the treeStats() hook on a 5-season load only, so the model gets cheap feedback on its hooks and never sees the lambda criterion). Known h2 caveat (HARDENING L25/L30): one goal turn can eat the whole wall (91 api calls), so the
# quality gate may run only once; the final gate is the verdict. Same model file, same server flags, same 16k thinking cap, same path line, same 60 min wall / 20 goal turns as 67f.
# PRE-REGISTERED PASS CRITERIA (written before any run; docstring of gate_trees.py + vault 99-personal/benchmark-rig/voxel-trees-status.md): base five checks on the page opened without parameters; treestats_hook (finite season/metamers/buds/height/crownWidth/shed,
# season == requested); metamers_grow (lambda .46, seasons 5/15/30: m15 >= .95 m5, m30 >= .95 m15, m30 > m5); non_trivial (>= 100 metamers at season 30 for both lambdas); lambda_effect (seed 1, 30 seasons: crownWidth/height at lambda 0.46 > at lambda 0.56, both numbers reported);
# deterministic (seed 1, lambda .46, 30 seasons twice: same metamer count). Extra seeds 2 and 3 for the lambda effect are reported, not gating. The clips are cut when the base checks and the hook pass, even if a paper check fails.
# DEVIATIONS from the 67f template: prompt trees-v1 (6.1 KB vs 3.2 KB: longer prompts have split into files before, HARDENING L21; the v3b "work in index.html from the first write" line is carried), gate_trees.py instead of gate.py (final gate has ~9 page loads, timeout 900),
# the item needs >= 90 min left in the block (67f: 75), clip = growth_clip.py (two clips: lambda 0.46 and 0.56, seed 1), no orbit_clip.py.
# template lineage: 67f (msm3b side test) <- 66zz2 (msm-v3) <- 66x (msm-v2 reference) <- 66v (temple pilot); the generic harness notes follow.
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
NL_TAG=voxtrees2
MANIFEST=/home/witcheer/voxel-voxtrees2-status.json
export MANIFEST
. /home/witcheer/night-lib.sh
V=/home/witcheer/voxelbench
BIN=/home/witcheer/llama.cpp-donald/build/bin/llama-server
GPY=/home/witcheer/benchmark-rig/.venv/bin/python
RUNLOG=/home/witcheer/voxel-pilot-run.log
SLUG=qwen38-27b-q6k-trees2
MODEL_FILE=/home/witcheer/models/qwen38-27b/Qwen3.8-27B-Q6_K.gguf
ALIAS=Qwen3.8-27B-Q6_K.gguf
CTX=131072; WALL=60; TURNS=20
# hardening 2026-10-21 14:30 CEST (after the 66v pilot spent 90k+ tokens of one thinking block drafting the whole
# program before any write_file): cap thinking per request, same value as the board's think-on 16k budget.
RB=16384
RB_MSG="Thinking budget reached. Stop planning and write the files now with your tools; refine in later turns."
ART=$V/runs/$SLUG/gate/gate.json

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0})"
nl_check_path "$BIN" "$GPY" "$MODEL_FILE" "$V/scripts/run_agent.sh" "$V/scripts/gate_trees.py" "$V/scripts/metrics.py" \
  "$V/prompts/trees-v1.txt" "$V/scripts/growth_clip.py" "$V/lib/libasound.so.2" "$V/ffmpeg" /home/witcheer/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome
HELP=$($BIN --help 2>&1 || true); case "$HELP" in *--reasoning-budget-message*) nl_say "  ok: --reasoning-budget(-message) in build";; *) nl_fail "--reasoning-budget-message missing from build";; esac
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

t0=$(date +%s); nl_mark "$SLUG" running
$BIN --model "$MODEL_FILE" --host 127.0.0.1 --port 8090 --n-gpu-layers 99 --ctx-size $CTX --parallel 1 --flash-attn on \
  --spec-type draft-mtp --spec-draft-n-max 2 --cache-type-k q8_0 --cache-type-v q8_0 --alias "$ALIAS" --metrics --reasoning-budget $RB --reasoning-budget-message "$RB_MSG" \
  > $V/runs-$SLUG-server.log 2>&1 &
for i in $(seq 1 120); do curl -s 127.0.0.1:8090/health | grep -qi ok && break; sleep 2; done
curl -s 127.0.0.1:8090/health | grep -qi ok || { nl_mark "$SLUG" failed "server did not come up"; exit 1; }
nl_say "server up ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader)), ctx $CTX"
export DEADLINE_EPOCH=$(( DEADLINE - 300 ))
mkdir -p $V/runs/$SLUG
PROMPT_RUN=$V/runs/$SLUG/prompt.txt
{ cat $V/prompts/trees-v1.txt; printf '\nThe working directory is %s. The final file must be exactly %s/index.html (copy this path, never retype it).\n' "$V/runs/$SLUG/work" "$V/runs/$SLUG/work"; } > "$PROMPT_RUN"
# h2 harness: socket-safe TMPDIR for the model's terminal (HARDENING L9) + Hermes /goal quality gate with the held-server setup (L3/L4)
VX_TMPFIX=1 VX_GATE_STRICT=1 GATE_CMD="$GPY $V/scripts/gate_trees.py $V/runs/$SLUG/work/index.html $V/runs/$SLUG/qgate --quick --video-seconds 0" bash $V/scripts/run_agent.sh "$SLUG" "$ALIAS" $CTX $WALL $TURNS "$PROMPT_RUN" >> "$RUNLOG" 2>&1
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
  setsid timeout 900 $GPY $V/scripts/gate_trees.py "$HTML" $V/runs/$SLUG/gate --video-seconds 20 < /dev/null >> "$RUNLOG" 2>&1; grc=$?
else
  echo '{"pass": false, "no_html": true}' > "$ART"; grc=1
fi
python3 $V/scripts/metrics.py /home/witcheer/.hermes/profiles/$(python3 -c "import json;print(json.load(open('$V/runs/$SLUG/driver.json'))['profile'])" 2>/dev/null || echo vx${SLUG//[^a-z0-9]/}) $V/runs/$SLUG/metrics.json >> "$RUNLOG" 2>&1
if nl_judge_artifact "$SLUG" 0 "$ART" "$t0" "$(( ($(date +%s)-t0)/60 )) min, gate rc $grc"; then
  nl_say "  gate: $(python3 -c "import json;d=json.load(open('$ART'));print('pass' if d.get('pass') else 'FAIL', d.get('checks'), 'fps', d.get('fps'), 'bytes', d.get('bytes'))")"
  nl_say "  trees: $(python3 -c "import json;d=json.load(open('$ART'));t=d.get('trees',{});print('metamers 5/15/30', t.get('metamers_5_15_30'), 'lambda_test', json.dumps(t.get('lambda_test')), 'determinism', json.dumps(t.get('determinism')), 'extra seeds holding', t.get('lambda_effect_seeds_holding'), 'elapsed_s', d.get('elapsed_s'))" 2>/dev/null)"
  # clips are cut when the page renders and the hooks answer, even if a paper check failed (a failed-lambda tree is still the film)
  if python3 -c "import json,sys;d=json.load(open('$ART'));sys.exit(0 if d.get('base_pass') and (d.get('checks') or {}).get('treestats_hook') else 1)"; then
    for LAM in 0.46 0.56; do
      D=showcase-growth; [ "$LAM" = 0.56 ] && D=showcase-growth-l056
      setsid timeout 300 $GPY $V/scripts/growth_clip.py "$HTML" $V/runs/$SLUG/$D --seed 1 --lambda $LAM --seasons 30 < /dev/null >> "$RUNLOG" 2>&1 \
        && nl_say "  growth clip (lambda $LAM): $V/runs/$SLUG/$D/growth.mp4" || nl_say "  growth clip lambda $LAM failed (gate result stands)"
    done
  else
    nl_say "  base checks or treeStats hook failed: no growth clip"
  fi
fi
exit 0
