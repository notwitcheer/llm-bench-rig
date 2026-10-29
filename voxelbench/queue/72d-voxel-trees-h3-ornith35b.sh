#!/bin/bash
# 72d-voxel-trees-h3-ornith35b.sh: idle-queue item, trees-v1 rerun on harness h3 (2 of 3: Ornith 1.5 35B-A3B Q4_K_M (vendor GGUF, no MTP draft)).
# R-H3 (2026-10-29 ~03:30 CET, mercury subagent; operator, GPU thread, 2026-10-29 ~02:45 CET: "please fix the tree issues"): the three trees runs of 2026-10-28/29 (71-r2 Qwen pass with a path miss + a gate-source read, 72a Ornith page froze the gate, 72b Gemma tree never grew and the old in-run gate passed it) are rerun on ONE fixed harness, h3, so the three models stay comparable
# (HARDENING 'compare runs only on the same harness version'): 72c Qwen3.8-27B Q6_K, 72d Ornith 1.5 35B-A3B Q4_K_M, 72e Gemma 4 31B QAT Q4_0. the earlier results stay separate rows on harness h2.
# EV: line 1 as it would post: "three local models got the same 2009 SIGGRAPH tree-growth paper and one hour in Hermes Agent: here are their three trees" (gpu-ev-gate rule 4 fully local Hermes piece + a picture; mojo = three different trees from one paper on one GPU, side by side; comparable: first voxel post 5,549 views, voxel sequels ~2k, a new hook per rule 7; only gate PASSES make the side-by-side, a FAIL is a board line, not an X post).
# serves: content (hardening: harness h3 is what makes the three trees comparable)
# MODEL / SAMPLING / FIT: exactly as 72a (voxelbench/model-params.md row 'Ornith 1.5 35B-A3B'): card general-task values --temp 0.6 --top-p 0.95 --top-k 20 + min-p 0.05, RB 16384, no draft flags, fit ladder 131072 first (22,348 MiB measured in 72a).
# why a rerun: 72a's page froze the tab (a model bug: an infinite loop in its isLiveMetamer helper, HARDENING L35) and Hermes stopped a repetition loop at 01:39 leaving the run idle for 32 min (L37). both are now handled by the harness (main-thread feedback in the in-run gate, stall nudge); the model still has to write a working page.
# queue order: SECOND: the best test of the new in-run feedback (a frozen tab is now reported with the measured block time) and of the stall nudge.
# PROMPT: prompts/trees-v1.txt UNCHANGED (6,064 bytes + the exact-path line the item appends); none of the three failures was prompt-caused, so no trees-v2.
# SAME AS 71-r2 / 72a / 72b: Hermes /goal in the classic CLI, VX_TMPFIX=1, VX_GATE_STRICT=1, 60 min wall, 20 goal turns, ctx 131072 (fit ladder as day0-voxel.sh), q8_0 KV, flash-attn, --parallel 1, llama.cpp 839a572 pin, 16k thinking cap + budget message, >= 90 min left in the window or exit 3, final gate = the same pre-registered pass rule (unchanged), two growth_clip.py clips after.
# h3 CHANGES (HARDENING L32-L38; none of them touches the pass rule):
#   L32 the in-run /goal quality gate is scripts/quick_gate_trees.py: base checks + treeStats hook + a growth check (a non-growing tree fails in-run), no paper criteria in it (the old --quick passed Gemma's 2-metamer tree).
#   L33 the FINAL gate lives outside the agent's tree (~/.vxfinal, a dot directory, no copy in ~/voxelbench, none in this header); the PLAN scans the agent-visible paths for the criteria and aborts on a hit.
#   L34 growth + path feedback: the in-run gate names the exact expected path and lists strays under runs/ (r2's model retyped the path twice and read ERR_FILE_NOT_FOUND as an environment bug); the slug has no `q6k-<word>` hyphen trap; the PATH MISS fallback + flag stay.
#   L35 a page that freezes the tab is a failure with a name: main_thread_ok + measured block time in gate.json and in the in-run gate's feedback (72a: Page.screenshot timeout read as harness_error).
#   L37 stall nudge (VX_STALL_NUDGE=1): a 12 min idle (no new message, server idle) with the goal active types one 'continue' line, at most 3 times; driver.json records stall_nudges.
# artifact: runs/<slug>/gate/gate.json (exact). bench profiles are NEVER deleted (host gateway holds their state.db). watchers: voxel-state.py slug list is not updated by this item (add the slug by hand if a monitor cron is armed).
set -uo pipefail
NL_TAG=voxtrees72d
MANIFEST=/home/witcheer/voxel-voxtrees72d-status.json
export MANIFEST
. /home/witcheer/night-lib.sh
V=/home/witcheer/voxelbench
BIN=/home/witcheer/llama.cpp-donald/build/bin/llama-server
GPY=/home/witcheer/benchmark-rig/.venv/bin/python
QG=$V/scripts/quick_gate_trees.py            # the in-run gate the agent can read (no paper criteria in it)
FINAL=/home/witcheer/.vxfinal/gate_trees.py   # the final gate, outside the agent's tree (HARDENING L33)
RUNLOG=/home/witcheer/voxel-pilot-run.log
SLUG=trees-h3-ornith15-35b-q4km
MODEL_FILE=/home/witcheer/models/ornith-15-35b/Ornith-1.5-35B-Q4_K_M.gguf
ALIAS=Ornith-1.5-35B-Q4_K_M.gguf
CTX=131072; WALL=60; TURNS=20; KV=q8_0
# maker sampling (HARDENING L5/L6; the same row is in voxelbench/model-params.md): the flags pin the card values over the GGUF keys
SAMPLING_FLAGS="--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.05"
PARAMS_SRC="https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF fetched 2026-10-28 (general tasks: temperature=0.6, top_p=0.95, top_k=20; no output budget named, RB floor 16384)"
EXTRA_ARR=()
# hardening 2026-10-21 14:30 CEST (after the 66v pilot spent 90k+ tokens of one thinking block drafting the whole
# program before any write_file): cap thinking per request, same value as the board's think-on 16k budget.
RB=16384
RB_MSG="Thinking budget reached. Stop planning and write the files now with your tools; refine in later turns."
ART=$V/runs/$SLUG/gate/gate.json

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0})"
nl_check_path "$BIN" "$GPY" "$MODEL_FILE" "$V/scripts/run_agent.sh" "$QG" "$FINAL" "$V/scripts/vx_stall.py" "$V/scripts/metrics.py" \
  "$V/prompts/trees-v1.txt" "$V/scripts/growth_clip.py" "$V/lib/libasound.so.2" "$V/ffmpeg" /home/witcheer/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome
# L33 enforcement: the criteria text must not sit on any path the agent explores (the final gate is in ~/.vxfinal; the PLAN refuses to start otherwise).
# the patterns are assembled from pieces so this scan does not match its own file.
[ ! -e "$V/scripts/gate_trees.py" ] || nl_fail "final gate still in the agent's tree: $V/scripts/gate_trees.py (move it to ~/.vxfinal)"
LEAKS=$(grep -rIl --exclude-dir=runs --exclude-dir=deb --exclude-dir=extract --exclude-dir=lib --exclude-dir=previews --exclude-dir=__pycache__ --exclude="runs-*" -e "lambda_""effect" -e "metamers_""grow" -e "non_""trivial" "$V" /home/witcheer/idle-queue/*.sh 2>/dev/null || true)
[ -z "$LEAKS" ] || nl_fail "pre-registered criteria visible to the agent in: $(echo $LEAKS | tr '\n' ' ')"
nl_say "  ok: criteria not visible on the agent paths ($V, ~/idle-queue/*.sh); final gate $FINAL"
"$GPY" -c "import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$QG" "$FINAL" "$V/scripts/vx_stall.py" || nl_fail "gate script syntax"
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
echo "{\"model_file\": \"$MODEL_FILE\", \"ctx\": $CTX, \"ctx_requested\": $CTX_REQ, \"kv\": \"$KV\", \"extra\": \"${EXTRA_ARR[*]}\", \"sampling\": \"$SAMPLING_FLAGS\", \"gguf_sampling\": \"${GG_SAMP:-}\", \"params_src\": \"$PARAMS_SRC\", \"reasoning_budget\": $RB, \"prompt\": \"trees-v1.txt\", \"build\": \"$VER\", \"vram_used\": \"$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)\"}" > $V/runs/$SLUG/run-config.json
export DEADLINE_EPOCH=$(( DEADLINE - 300 ))
mkdir -p $V/runs/$SLUG
PROMPT_RUN=$V/runs/$SLUG/prompt.txt
{ cat $V/prompts/trees-v1.txt; printf '\nThe working directory is %s. The final file must be exactly %s/index.html (copy this path, never retype it).\n' "$V/runs/$SLUG/work" "$V/runs/$SLUG/work"; } > "$PROMPT_RUN"
# h3 harness (h2 + L32-L37): socket-safe TMPDIR for the model's terminal (HARDENING L9) + Hermes /goal quality gate with the held-server setup (L3/L4)
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
  setsid timeout 900 $GPY $FINAL "$HTML" $V/runs/$SLUG/gate --video-seconds 20 < /dev/null >> "$RUNLOG" 2>&1; grc=$?
else
  echo '{"pass": false, "no_html": true}' > "$ART"; grc=1
fi
python3 $V/scripts/metrics.py /home/witcheer/.hermes/profiles/$(python3 -c "import json;print(json.load(open('$V/runs/$SLUG/driver.json'))['profile'])" 2>/dev/null || echo vx${SLUG//[^a-z0-9]/}) $V/runs/$SLUG/metrics.json >> "$RUNLOG" 2>&1
if nl_judge_artifact "$SLUG" 0 "$ART" "$t0" "$(( ($(date +%s)-t0)/60 )) min, gate rc $grc"; then
  nl_say "  gate: $(python3 -c "import json;d=json.load(open('$ART'));print('pass' if d.get('pass') else 'FAIL', d.get('checks'), 'fps', d.get('fps'), 'bytes', d.get('bytes'), 'main_thread', json.dumps(d.get('main_thread')))")"
  nl_say "  trees: $(python3 -c "import json;d=json.load(open('$ART'));t=d.get('trees',{});print('metamers 5/15/30', t.get('metamers_5_15_30'), 'lambda_test', json.dumps(t.get('lambda_test')), 'determinism', json.dumps(t.get('determinism')), 'extra seeds holding', t.get('lambda_'+'effect_seeds_holding'), 'growth_flags', json.dumps(t.get('growth_flags')), 'elapsed_s', d.get('elapsed_s'))" 2>/dev/null)"
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

