#!/bin/bash
# 79-clover-hermes.sh: idle-queue item (STAGED in ~/voxel-stage, not enqueued; 2026-11-02, mercury subagent for the GPU thread, weekend-fills pick 1 + mojo slate F2).
# WHAT: Clover-35B-A3B-Preview (DiffuSpace, 2026-10-24, apache-2.0, block-diffusion MoE, ~3B active, Qwen3.6-35B-A3B base) Q4_K_M drives Hermes Agent /goal for one hour on the series prompt msm-v2
#   (Mont-Saint-Michel diorama), harness h3 (same as 74w), comparable with the Qwen3.8-27B Q6_K msm-v2-h2 reference (66zz1). AFTER the gate the same weights draft/verify a fixed prompt through
#   llama-speculative-simple with the CLOVER_TRACE patch and a renderer turns that log into the decode clip (whole blocks appearing, then accepted / bonus / rejected tokens).
# MOJO: a diffusion language model drafting its text a whole block at a time (grey block lands at once, then white accepted / red rejected) next to the Mont-Saint-Michel it builds through Hermes Agent on one 5090 | never-seen / esoteric: a block-diffusion LM as the agent brain, and its decode drawn token for token (checked 2026-11-02: web search finds a YouTube local test of Clover and the vendor blog, no Hermes /goal build and no draft/accept picture; never claimed as 'first') | line 1: "a diffusion language model built Mont-Saint-Michel through Hermes Agent on one RTX 5090, and here is how it writes: a whole block at once, then the checker throws out what it does not believe"
# EV: line 1 above; rules: gpu-ev-gate-2026-10-25 section 2 rule 4 (reproducible, fully local Hermes piece) + the franchise F2 never-seen mechanic (mojo-slate-2026-11-02). comparables: first voxel post 5,549 views, episode 1 8,030, voxel sequels 1.7-2.6k; a new
#   architecture on screen is the new hook (a sequel without one decays). bracket 2-6k. X only if the final gate PASSES and the page is watchable; a gate fail is the decode clip alone or a board line. n = 1 sample (greedy), so no 'beats Qwen' is ever claimed from this run.
# serves: content (a never-seen mechanic + a Hermes-driven showcase) + experiment (can a block-diffusion model carry a long Hermes tool loop at agent speed?).
#
# EVIDENCE BEHIND THE FLAGS (all 2026-11-02, capsule, CPU probes ~/voxel-stage/clover/cpuprobe*, GPU smoke ~/voxel-stage/clover/gpusmoke + hermes-smoke, vault 99-personal/benchmark-rig/clover-35b-status.md):
#   - runtime: the DiffuSpace llama.cpp PR (ggml-org/llama.cpp#31306) built at ~/llama.cpp-clover-pr HEAD cb09d4355, CUDA sm120. stock llama.cpp (Donald's 839a572) cannot load the model. the PR path serves Clover as SELF-SPECULATIVE decoding
#     (--spec-type draft-self: one bidirectional masked-block pass drafts n tokens, the causal pass verifies), NOT a native diffusion sampler: there are no intermediate denoising states to capture on this path.
#   - SAMPLING = GREEDY. maker generation_config.json (HF, fetched 2026-11-02): do_sample false, presence_penalty 0.9, repetition_penalty 1.05 (temperature 1.0 / top_p 1.0 / top_k null are ignored under do_sample false).
#     CPU probes 7b/10: at temp 1.0 top-k off the draft acceptance collapsed on long turns (0.0034 = 2/588 at a 300-token turn; 0.104 = 531/5090 over a 3,000-token write_file, the file degenerated into multilingual junk after ~2k tokens, 6.7 tok/s);
#     the same long turn at greedy: acceptance 0.752 (2052/2730), 15.3 tok/s on CPU, 98 tok/s on the 5090 (acceptance 0.782). that is the long-turn collapse fixed: it was our sampling, not the model. the earlier 'tool_calls 0' reply was a max_tokens-truncated turn.
#   - n_max 4 (--spec-draft-n-max 4): CPU probe 5 + trace: block 4 accepts 75/75 replay-adjusted on the trace prompt; the README demo uses 15 for the trace binary only.
#   - truncated tool call: llama-server answers HTTP 500 'Failed to parse tool call arguments as JSON' when a turn hits the token cap mid-argument (probes 8-11, the same on the GPU). Hermes sends no max_tokens, so the server cap -n 24576 (maker max_new_tokens) is the only
#     bound; a runaway turn would come back as that 500 and Hermes retries. UNVERIFIED: how often a real msm-v2 turn does that.
#   - the GPU fit (measured, free card): ctx 131072 q8_0 KV loads at 23,994 MiB (peak 24,064 over a 6-turn loop + a 3,000-token write_file), ctx 65536 at 23,122 MiB (peak 23,192): 8 GiB of headroom, so the Chromium gate fits.
#   - THE HERMES LOOP: a real Hermes Agent /goal session on this exact server is the 'hermes-smoke' in the vault note (result recorded there, not assumed here).
# REGIME vs the series reference (Qwen3.8-27B Q6_K msm-v2-h2, 66zz1): THESE ARE CONFOUNDS, label every clip and number with them:
#   1. model/file: Clover-35B-A3B-Preview Q4_K_M (22,998,009,248 bytes) not Qwen3.8-27B Q6_K; a diffusion-trained MoE (~3B active) vs a dense 27B.  2. decoding: draft-self n-max 4, GREEDY (reference temp 1.0 / top-p 0.95 / top-k 20 + min-p 0.05, MTP n-max 2).
#   3. ctx 131072 q8_0 KV (same as the reference), fit ladder 131072 > 98304 > 65536.  4. engine: the PR build cb09d4355 (reference llama.cpp 839a572).  5. RB 16384 same as the reference.  6. harness h3 (VX_GATE_STRICT, VX_STALL_NUDGE, quick_gate_msm.py), the reference was h2, same final gate.py 5 checks.
#   SAME: prompt msm-v2 (md5 logged) + the exact-path line, 60 min wall, 20 goal turns, --parallel 1, flash-attn, Hermes /goal classic CLI, VX_TMPFIX=1, final gate with --video-seconds 20, orbit clip recipe. speed (tok/s) is never a cross-model claim.
# DECODE TRACE (the clip picture): after the gate, llama-speculative-simple (CLOVER_TRACE=1; local side patch ~/voxel-stage/clover/clover-trace.patch applied in ~/llama.cpp-clover-pr, uncommitted) decodes the msm-v2 prompt (thinking off, 320 tokens, n-max 15) and
#   writes one JSON line per verify step; ~/voxelbench/scripts/clover_trace_render.py draws it (grey block at once, then white accepted / cyan bonus / red rejected). The clip is a draft-vs-accept log, honest label: 'self-speculative block drafting', never 'denoising steps'. HUD read: the frames carry only model text + step counters.
# resumable: needs >= 90 min left in the window (60 min wall + load + sha + gate + orbit clip + trace, ~75 min worst case) or exits 3 (stays queued).
# artifact: runs/<slug>/gate/gate.json (exact). bench profiles are NEVER deleted (host gateway holds their state.db). own-pid kills only (the server this item started); Donald drained with sudo -n systemctl stop, EXIT trap restores.
set -uo pipefail
NL_TAG=voxclover79
MANIFEST=/home/witcheer/voxel-voxclover79-status.json
# throwaway manifest for a staging DRYRUN only (the drain greps the literal ^MANIFEST= line above)
[ -n "${MANIFEST_OVERRIDE:-}" ] && MANIFEST=$MANIFEST_OVERRIDE
export MANIFEST
. /home/witcheer/night-lib.sh
V=/home/witcheer/voxelbench
BIN=/home/witcheer/llama.cpp-clover-pr/build/bin/llama-server
SPEC=/home/witcheer/llama.cpp-clover-pr/build/bin/llama-speculative-simple
GPY=/home/witcheer/benchmark-rig/.venv/bin/python
QG=$V/scripts/quick_gate_msm.py
GATE=$V/scripts/gate.py
RENDER=$V/scripts/clover_trace_render.py
RUNLOG=/home/witcheer/voxel-pilot-run.log
PVER=msm-v2
SLUG=msm2-h4-clover35b-q4km
MODEL_FILE=/home/witcheer/models/clover-35b/Clover-35B-A3B-Preview-Q4_K_M.gguf
MODEL_BYTES=22998009248
SUMS=/home/witcheer/models/clover-35b/SHA256SUMS
ALIAS=Clover-35B-A3B-Preview-Q4_K_M.gguf
PR_SHA=cb09d4355
CTX=131072; WALL=60; TURNS=20; KV=q8_0
# maker sampling: generation_config.json do_sample false = greedy, presence 0.9, repetition 1.05 (see the evidence block above)
SAMPLING_FLAGS="--temp 0 --top-k 1 --top-p 1.0 --min-p 0 --presence-penalty 0.9 --repeat-penalty 1.05"
PARAMS_SRC="https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview generation_config.json fetched 2026-11-02 (do_sample false, presence_penalty 0.9, repetition_penalty 1.05, max_new_tokens 24576; HF temperature 1.0/top_p 1.0 ignored under greedy); probes 7b/10 show temp 1.0 collapses the draft acceptance on long turns; RB 16384 (series value)"
EXTRA_ARR=(--spec-type draft-self --spec-draft-n-max 4 -n 24576)
RB=16384
RB_MSG="Thinking budget reached. Stop planning and write the files now with your tools; refine in later turns."
ART=$V/runs/$SLUG/gate/gate.json
EXPECT_RUN=1
REGIME="Clover-35B-A3B-Preview Q4_K_M (diffusion MoE, self-speculative draft-self n-max 4, PR build cb09d4355) vs Qwen3.8-27B Q6_K msm-v2-h2 (66zz1): greedy sampling (reference 1.0/0.95/20 + min-p 0.05), ctx CTXVAL q8_0 KV, RB 16384, harness h3 not h2, same prompt msm-v2, wall 60 min, 20 goal turns"

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0})"
nl_check_path "$BIN" "$SPEC" "$GPY" "$MODEL_FILE" "$SUMS" "$V/scripts/run_agent.sh" "$QG" "$GATE" "$RENDER" "$V/scripts/vx_stall.py" "$V/scripts/metrics.py" \
  "$V/prompts/$PVER.txt" "$V/scripts/orbit_clip.py" "$V/scripts/orbit_clip_multidrag.py" "$V/scripts/orbit_pick.py" "$V/lib/libasound.so.2" "$V/ffmpeg" /home/witcheer/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome
SZ=$(stat -c %s "$MODEL_FILE" 2>/dev/null || echo 0)
[ "$SZ" = "$MODEL_BYTES" ] && nl_say "  ok: model file $SZ bytes" || nl_fail "model file size $SZ != $MODEL_BYTES"
grep -q "Clover-35B-A3B-Preview-Q4_K_M.gguf" "$SUMS" 2>/dev/null && nl_say "  ok: SHA256SUMS lists the file (full sha256 -c runs in the RUN path before the drain)" || nl_fail "SHA256SUMS does not list the Q4_K_M file"
CHEAD=$(git -C /home/witcheer/llama.cpp-clover-pr rev-parse --short=9 HEAD 2>/dev/null || echo none)
[ "$CHEAD" = "$PR_SHA" ] && nl_say "  ok: Clover PR clone at $CHEAD" || nl_fail "Clover PR clone HEAD '$CHEAD' != $PR_SHA"
$BIN --version 2>&1 | grep -q "$PR_SHA" && nl_say "  ok: llama-server build reports $PR_SHA" || nl_fail "llama-server binary is not the $PR_SHA build"
LD_LIBRARY_PATH=$(dirname $BIN) strings "$(dirname $BIN)/llama-speculative-simple" | grep -q CLOVERTRACE && nl_say "  ok: llama-speculative-simple carries the CLOVER_TRACE patch" || nl_fail "llama-speculative-simple lacks CLOVERTRACE (rebuild: apply clover-trace.patch)"
"$GPY" -c "import PIL, ast, sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$QG" "$GATE" "$V/scripts/vx_stall.py" "$RENDER" || nl_fail "gate/renderer syntax or PIL import"
bash -n "$V/scripts/run_agent.sh" || nl_fail "run_agent.sh syntax"
HELP=$($BIN --help 2>&1 || true)
case "$HELP" in *--reasoning-budget-message*) nl_say "  ok: --reasoning-budget(-message) in build";; *) nl_fail "--reasoning-budget-message missing from build";; esac
case "$HELP" in *draft-self*) nl_say "  ok: --spec-type draft-self in build";; *) nl_fail "draft-self missing from this build";; esac
for f in --temp --top-k --top-p --min-p --presence-penalty --repeat-penalty --spec-type --spec-draft-n-max; do case "$HELP" in *"$f"*) ;; *) nl_fail "flag $f not in this build";; esac; done; nl_say "  ok: sampling/extra flags exist in build: $SAMPLING_FLAGS ${EXTRA_ARR[*]}"
[ -n "$PARAMS_SRC" ] || nl_fail "PARAMS_SRC empty"
nl_say "  params: src='$PARAMS_SRC' flags='$SAMPLING_FLAGS' gguf_sampling=none-read (maker values from the HF generation_config)"
nl_check_disk 20
nl_say "  regime: ${REGIME//CTXVAL/$CTX}"
if nl_skip_exact "$ART"; then nl_say "  SKIP: $SLUG artifact exists: $ART"; PLAN_RUN=0; else nl_say "  RUN: $SLUG artifact $ART"; PLAN_RUN=1; fi
[ "$PLAN_RUN" -eq "$EXPECT_RUN" ] || { nl_say "PLAN: planned RUN $PLAN_RUN != EXPECT_RUN $EXPECT_RUN (item complete or mismatch); nothing to do"; [ "$PLAN_RUN" = 0 ] && exit 0; exit 2; }
DL=$(nl_deadline_epoch); LEFT=$(( (DL-$(date +%s))/60 ))
nl_say "  planned RUN: $SLUG (wall $WALL min + ~15 min load, sha, gate, clips, trace; hard stop $(date -d "@$DL" '+%a %H:%M') local, $LEFT min away)"
[ "$LEFT" -ge 90 ] || { nl_say "  under 90 min left in the window: not starting (60 min wall + gate + clips would be cut short; exit 3, resumable)"; exit 3; }
if [ "${DRYRUN:-0}" = "1" ] || [ ! -f "$MANIFEST" ]; then nl_manifest_init "$SLUG"; fi
nl_gate

SRV=""
killsrv() { [ -n "$SRV" ] && kill -9 "$SRV" 2>/dev/null; SRV=""; }   # only the server THIS item started
VRAMLOG=""; TRACER=""
restore() { [ -n "$TRACER" ] && kill "$TRACER" 2>/dev/null; [ -n "$VRAMLOG" ] && kill "$VRAMLOG" 2>/dev/null; tmux kill-session -t "=voxel-$SLUG" 2>/dev/null; killsrv; sleep 5; nl_say "EXIT trap: restoring Donald"; nl_donald_restore; }
trap restore EXIT
DEADLINE=$(nl_deadline_epoch)
nl_say "sha256 check of the weights (before the drain)"
( cd "$(dirname $MODEL_FILE)" && grep "Clover-35B-A3B-Preview-Q4_K_M.gguf" "$SUMS" | sha256sum -c - ) > /tmp/clover79-sha.txt 2>&1 || { nl_say "  FAIL sha256: $(cat /tmp/clover79-sha.txt)"; nl_mark "$SLUG" failed "weights sha256 mismatch"; exit 1; }
nl_say "  sha256 ok"
nl_say "draining Donald"; nl_donald_stop
for i in $(seq 1 120); do v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${v:-99999}" -lt 1024 ] && { nl_say "drain clear (${v} MiB)"; break; }; sleep 2; done
[ "${v:-99999}" -lt 1024 ] || { nl_mark "$SLUG" failed "Donald drain never cleared (${v} MiB)"; exit 1; }
[ "$(ls -A $V/runs/$SLUG/work 2>/dev/null)" ] && { nl_say "  stale work dir from an earlier attempt: moving aside"; mv $V/runs/$SLUG $V/runs/$SLUG.stale-$(date +%s); }

mkdir -p "$V/runs/$SLUG/work"
t0=$(date +%s); nl_mark "$SLUG" running; UP=0; CTX_REQ=$CTX
for c in $CTX_REQ 98304 65536; do
  [ "$c" -le "$CTX_REQ" ] || continue
  # shellcheck disable=SC2086
  LD_LIBRARY_PATH=$(dirname $BIN) $BIN --model "$MODEL_FILE" --host 127.0.0.1 --port 8090 --n-gpu-layers 99 --ctx-size $c --parallel 1 --flash-attn on --jinja --no-warmup \
    --cache-type-k $KV --cache-type-v $KV --alias "$ALIAS" --metrics --reasoning-budget $RB --reasoning-budget-message "$RB_MSG" $SAMPLING_FLAGS "${EXTRA_ARR[@]}" \
    > $V/runs-$SLUG-server.log 2>&1 &
  SRV=$!
  for i in $(seq 1 150); do curl -s 127.0.0.1:8090/health | grep -qi ok && { UP=1; break; }; kill -0 $SRV 2>/dev/null || break; sleep 2; done
  [ "$UP" = 1 ] && { CTX=$c; break; }
  nl_say "  server did not come up at ctx $c: $(grep -m1 -iE 'out of memory|failed|error|unknown' $V/runs-$SLUG-server.log | head -c 200)"
  killsrv; sleep 4
done
[ "$UP" = 1 ] || { nl_mark "$SLUG" failed "server did not come up at any ctx (see runs-$SLUG-server.log)"; exit 1; }
nl_say "server up ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader)), ctx $CTX (requested $CTX_REQ)"
mkdir -p $V/runs/$SLUG
( while :; do echo "$(date +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" >> $V/runs/$SLUG/vram-samples.csv; sleep 20; done ) &
VRAMLOG=$!
curl -s 127.0.0.1:8090/props > $V/runs/$SLUG/props.json
SCHK=$(python3 - "$V/runs/$SLUG/props.json" <<'PYEOF'
import json, sys
g = json.load(open(sys.argv[1])).get("default_generation_settings", {}); p = g.get("params", g)
want = {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0, "presence_penalty": 0.9, "repeat_penalty": 1.05}
got = {k: p.get(k) for k in want}
ok = all(got[k] is not None and abs(float(got[k]) - v) < 1e-3 for k, v in want.items())
print(("OK " if ok else "BAD ") + json.dumps(got))
PYEOF
)
nl_say "  server sampling defaults: $SCHK"
case "$SCHK" in OK*) ;; *) nl_mark "$SLUG" failed "server sampling defaults wrong: $SCHK"; exit 1;; esac
VER=$($BIN --version 2>&1 | grep -m1 -i 'version' | tr -d '"')
PMD5=$(md5sum $V/prompts/$PVER.txt | cut -c1-12)
python3 - "$V/runs/$SLUG/run-config.json" "$MODEL_FILE" "$CTX" "$CTX_REQ" "$KV" "$SAMPLING_FLAGS" "${EXTRA_ARR[*]}" "$RB" "$VER" "$PVER" "$PMD5" "$PARAMS_SRC" "${REGIME//CTXVAL/$CTX}" "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" <<'PYEOF'
import json, sys
a = sys.argv
json.dump({"model_file": a[2], "ctx": int(a[3]), "ctx_requested": int(a[4]), "kv": a[5], "sampling": a[6], "extra": a[7],
           "reasoning_budget": int(a[8]), "build": a[9], "prompt": a[10] + ".txt", "prompt_md5": a[11], "params_src": a[12], "regime": a[13],
           "vram_at_load": a[14], "harness": "h3 (VX_TMPFIX, VX_GATE_STRICT, VX_STALL_NUDGE, quick_gate_msm.py)", "min_ctx_hermes": 64000}, open(a[1], "w"), indent=2)
PYEOF
( while :; do curl -s -m5 127.0.0.1:8090/slots 2>/dev/null | python3 -c "
import json,sys,time
try:
    d=json.load(sys.stdin); s=d[0] if isinstance(d,list) else d; p=s.get('params',{})
    print(json.dumps({'t':int(time.time()),'processing':s.get('is_processing'),'id_task':s.get('id_task'),**{k:p.get(k) for k in ('temperature','top_k','top_p','min_p','n_predict')}}))
except Exception as e: print(json.dumps({'t':int(time.time()),'err':str(e)[:80]}))
" >> $V/runs/$SLUG/sampling-trace.jsonl; sleep 60; done ) &
TRACER=$!
export DEADLINE_EPOCH=$(( DEADLINE - 600 ))
PROMPT_RUN=$V/runs/$SLUG/prompt.txt
{ cat $V/prompts/$PVER.txt; printf '\nThe working directory is %s. The final file must be exactly %s/index.html (copy this path, never retype it).\n' "$V/runs/$SLUG/work" "$V/runs/$SLUG/work"; } > "$PROMPT_RUN"
VX_TMPFIX=1 VX_GATE_STRICT=1 VX_STALL_NUDGE=1 GATE_CMD="$GPY $QG $V/runs/$SLUG/work/index.html $V/runs/$SLUG/qgate --video-seconds 0" bash $V/scripts/run_agent.sh "$SLUG" "$ALIAS" $CTX $WALL $TURNS "$PROMPT_RUN" >> "$RUNLOG" 2>&1
nl_say "driver: $(cat $V/runs/$SLUG/driver.json 2>/dev/null)"
[ -f "$V/runs/$SLUG/path-miss.json" ] && nl_say "  PATH MISS flag present: $(cat $V/runs/$SLUG/path-miss.json)"
kill "$TRACER" 2>/dev/null; TRACER=""
curl -s 127.0.0.1:8090/metrics | grep -E "^llamacpp:(tokens_predicted_total|prompt_tokens_total|tokens_predicted_seconds_total|prompt_seconds_total)" > $V/runs/$SLUG/server-metrics.txt 2>/dev/null
nl_say "  server metrics: $(tr '\n' ' ' < $V/runs/$SLUG/server-metrics.txt | head -c 400)"
nl_say "  draft acceptance over the run (server log, last 5 task lines): $(grep -E 'draft acceptance' $V/runs-$SLUG-server.log | tail -5 | sed 's/.*draft acceptance/acc/' | tr '\n' ';' | head -c 500)"
kill "$VRAMLOG" 2>/dev/null; VRAMLOG=""
nl_say "  vram during the run (MiB, peak of $(wc -l < $V/runs/$SLUG/vram-samples.csv 2>/dev/null) samples): $(sort -t, -k2 -n $V/runs/$SLUG/vram-samples.csv 2>/dev/null | tail -1 | cut -d, -f2) (card total $(nvidia-smi --query-gpu=memory.total --format=csv,noheader))"
killsrv; sleep 3
HTML=$V/runs/$SLUG/work/index.html
[ -f "$HTML" ] || HTML=$(ls -t $V/runs/$SLUG/work/*.html 2>/dev/null | head -1)
if [ -z "$HTML" ] || [ ! -f "$HTML" ]; then
  STRAY=$(find $V/runs -name '*.html' -newermt "@$t0" -size +1k -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
  if [ -n "$STRAY" ]; then HTML=$STRAY; echo "{\"path_miss\": true, \"found\": \"$STRAY\"}" > $V/runs/$SLUG/path-miss.json; nl_say "  PATH MISS: no html in work/, grading the fresh file at $STRAY (flagged)"; fi
fi
mkdir -p $V/runs/$SLUG/gate
if [ -n "$HTML" ] && [ -f "$HTML" ]; then
  setsid timeout 300 $GPY $GATE "$HTML" $V/runs/$SLUG/gate --video-seconds 20 < /dev/null >> "$RUNLOG" 2>&1; grc=$?
else
  echo '{"pass": false, "no_html": true}' > "$ART"; grc=1
fi
python3 $V/scripts/metrics.py /home/witcheer/.hermes/profiles/$(python3 -c "import json;print(json.load(open('$V/runs/$SLUG/driver.json'))['profile'])" 2>/dev/null || echo vx${SLUG//[^a-z0-9]/}) $V/runs/$SLUG/metrics.json >> "$RUNLOG" 2>&1
if nl_judge_artifact "$SLUG" 0 "$ART" "$t0" "$(( ($(date +%s)-t0)/60 )) min, gate rc $grc"; then
  nl_say "  gate: $(python3 -c "import json;d=json.load(open('$ART'));print('pass' if d.get('pass') else 'FAIL', d.get('checks'), 'fps', d.get('fps'), 'bytes', d.get('bytes'))")"
  if python3 -c "import json,sys;sys.exit(0 if json.load(open('$ART')).get('pass') else 1)"; then
    mkdir -p $V/runs/$SLUG/showcase
    setsid timeout 120 $GPY $V/scripts/orbit_pick.py "$HTML" --clip-seconds 30 < /dev/null 2>>"$RUNLOG" | tail -1 > $V/runs/$SLUG/showcase/orbit-choice.json
    OPX=$(python3 -c "import json;print(json.load(open('$V/runs/$SLUG/showcase/orbit-choice.json'))['orbit_px'])" 2>/dev/null || echo 1257)
    OSC=$(python3 -c "import json;print(json.load(open('$V/runs/$SLUG/showcase/orbit-choice.json'))['script'])" 2>/dev/null || echo orbit_clip.py)
    case "$OSC" in orbit_clip.py|orbit_clip_multidrag.py) ;; *) OSC=orbit_clip.py;; esac
    nl_say "  orbit choice: $OSC --orbit-px $OPX"
    setsid timeout 240 $GPY $V/scripts/$OSC "$HTML" $V/runs/$SLUG/showcase --seconds 30 --orbit-px "$OPX" --keys "n@11,n@22" < /dev/null >> "$RUNLOG" 2>&1 \
      && nl_say "  showcase: $V/runs/$SLUG/showcase/orbit.mp4" || nl_say "  showcase clip failed (gate result stands)"
  fi
fi
# DECODE TRACE for the clip (runs whatever the gate said: the picture is the model's decode, not the page). server is down, the card is ours.
TD=$V/runs/$SLUG/trace; mkdir -p $TD
{ printf '<|im_start|>user\n'; cat $V/prompts/$PVER.txt; printf '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'; } > $TD/prompt.txt
CLOVER_TRACE=1 LD_LIBRARY_PATH=$(dirname $BIN) timeout 240 $SPEC -m "$MODEL_FILE" -f $TD/prompt.txt --spec-type draft-self --spec-draft-n-max 15 --temp 0 --top-k 1 --seed 42 \
  -ctk bf16 -ctv bf16 -c 8192 -n 320 -b 1024 -ub 1024 -ngl 99 --flash-attn on --color off > $TD/out.txt 2> $TD/err.txt
nl_say "  trace: $(grep -c CLOVERTRACE $TD/err.txt) verify steps logged, $(grep -E 'decoded .* tokens' $TD/err.txt | tail -1 | cut -c1-120)"
if [ "$(grep -c CLOVERTRACE $TD/err.txt)" -ge 5 ]; then
  setsid timeout 300 $GPY $RENDER $TD/err.txt $TD/clover-decode.mp4 --ffmpeg $V/ffmpeg < /dev/null >> "$RUNLOG" 2>&1 \
    && nl_say "  decode clip: $TD/clover-decode.mp4 (label: self-speculative block drafting, not denoising steps)" || nl_say "  decode clip render failed (trace log stands: $TD/err.txt)"
else
  nl_say "  decode trace too short, no clip"
fi
nl_say "DONE $SLUG"
exit 0
