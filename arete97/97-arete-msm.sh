#!/bin/bash
# 97-arete-msm.sh: idle-queue item (staged in ~/voxel-stage/arete97, 2026-11-07).
# operator idea, 2026-11-07 ~09:25 CET: "mont saint michel voxel comparison between qwen 3.8 27b and arete 27b"
# WHAT: the Mont-Saint-Michel voxel duel. Qwen3.8-27B Q6_K and Arete-27B Q6_K (Nous Research's SFT of it) each get the series prompt (prompts/msm-v2.txt) and one hour of /goal in
#   Hermes Agent, harness h3, no speculative drafting for either, TWO takes per model interleaved Q1, A1, Q2, A2. Per passing take an orbit clip; after all four a landscape two-pane vs clip
#   (pre-registered pair: take 1 of each, else the model's first passing take, labelled) built on this box. Verdict = the frozen rule in PREREG.md, applied by arete97_summary.py.
# MOJO: same prompt, same card, same hour: Qwen3.8-27B and Nous's own fine-tune of it each build Mont-Saint-Michel in Hermes Agent, side by side | intense (a two-pane race) + brand pull (Nous release this week) + arguable (did the values fine-tune cost any skill?) | line 1: drafted at readout from the pre-registered verdict, never before it exists
# EV: voxel franchise (first voxel post 5,549 views, episode 1 8,030, voxel sequels 1.7-2.6k, 40-content/posted-ledger.md) + the Nous Arete release wave (model card https://huggingface.co/NousResearch/Arete-27B, fetched 2026-11-07); rule gpu-ev-gate-2026-10-25 section 2 rule 7 (challenger vs champion on the same prompt) + rule 4 (reproducible fully local Hermes piece). bracket 3-7k views UNVERIFIED. n = 2 takes per model at temperature 1.0: "better" is never claimed from this run; a 1-take difference is noise-level by the frozen rule.
#   what is published, and in which shape, is decided at readout from the frozen verdict and the 8-element read.
# serves: content (a model-pair verdict on the series prompt, a Hermes-driven showcase) + experiment (does the fine-tune change what the model can build?).
#
# REGIME, identical for both models (the plan, PREREG.md): prompt msm-v2 (md5 30a4b777e510) + the exact-path line; harness h3 = VX_TMPFIX=1 VX_GATE_STRICT=1 VX_STALL_NUDGE=1 + in-run gate quick_gate_msm.py (HARDENING L31-L39);
#   60 min wall, 20 goal turns, ctx 131072, q8_0 KV, --parallel 1, flash-attn, llama.cpp-donald 839a572a4 (pin-checked), NO --spec flags (Arete has no MTP head; the Qwen file's nextn layer stays unused: both logs checked),
#   RB 16384 + the series budget message, sampling --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 (both GGUFs carry exactly temp 1.0 / top-p 0.95 / top-k 20 in general.sampling.*; Arete's generation_config.json says the same),
#   thinking on (series default: both templates open <think>), final gate gate.py --video-seconds 20, orbit clip per passing take (orbit_pick.py, keys n@11,n@22).
# CONFOUNDS (label every clip and number): (1) different cutters and imatrix: Qwen file = Unsloth Q6_K 22,884,408,288 B (imatrix 45 chunks), Arete file = bartowski Q6_K 23,621,581,824 B (imatrix 583 chunks);
#   (2) chat templates: the Qwen file carries the Unsloth-fixed template (merged system messages, stricter tool-call checks), Arete the maker template; a probe conversation (system + tool call + reasoning history) renders byte-identically on both (CPU, 2026-11-07);
#   (3) vs the older h2 Qwen rows (66zz1 + r2/r3, MTP drafting, min-p 0.05, harness h2): context only, never a comparison, because both arms here run no drafting, min-p 0 and h3;
#   (4) both arms are served under ONE alias, the reference alias Qwen3.8-27B-Q6_K.gguf: Hermes picks prompt blocks by model-name substring (tool-use enforcement + execution guidance fire for "qwen"; checked on capsule 2026-11-07), so an Arete-named alias would hand the two arms different system prompts;
#   (5) one server rung only (ctx 131072): a smaller fallback would make the arms differ, so a failed load stops the item (exit 1) rather than shrink one arm. fit is an ESTIMATE until the first load: the Qwen reference loaded at 27,654 MiB with MTP, Arete adds ~0.7 GB of weights.
# TIME: no new take starts after Sat 2026-11-07 22:00 CET and GPU work ends by 23:40 CET whatever NL_HARD_STOP says (09z30 is NOT_BEFORE Sun 00:00); from Sun 00:00 only NL_HARD_STOP applies. needs >= 90 min left for every take or exits 3 (resumable, stays queued).
#   ~65 min of card per take (2 load + 61 wall + ~1.5 gate/orbit) => <= ~4.4 h for four takes + ~5 min drain/restore; less when a model ends its goal early. one Donald drain for the whole item, EXIT trap restores.
# resumable: a take whose gate.json exists is skipped; a take that died without one is retried under a fresh slug (t1b, t1c: the bench profile name must be new).
# artifacts: ~/voxelbench/runs/msm2-h3-t<n>[b|c]-{qwen38-27b-q6k,arete27b-q6k}/gate/gate.json ; summary ~/voxel-stage/arete97/run/{summary.json,pair.json,summary.md} ; clip ~/voxelbench/runs/vs-qwen-arete97/. bench profiles are NEVER deleted.
set -uo pipefail
NL_TAG=vox97arete
MANIFEST=/home/witcheer/voxel-vox97arete-status.json
# throwaway manifest for a staging DRYRUN only (the drain greps the literal ^MANIFEST= line above)
[ -n "${MANIFEST_OVERRIDE:-}" ] && MANIFEST=$MANIFEST_OVERRIDE
export MANIFEST
. /home/witcheer/night-lib.sh
S=/home/witcheer/voxel-stage/arete97
V=/home/witcheer/voxelbench
BIN=/home/witcheer/llama.cpp-donald/build/bin/llama-server
GPY=/home/witcheer/benchmark-rig/.venv/bin/python
QG=$V/scripts/quick_gate_msm.py       # in-run gate = gate.py + the exact-path feedback (HARDENING L39), no new criterion
GATE=$V/scripts/gate.py               # final gate = the series gate, unchanged
RUNLOG=/home/witcheer/voxel-pilot-run.log
SUMMARY=$S/code/arete97_summary.py
VSBUILD=$S/code/vs-arete97-build.sh
PVER=msm-v2
SLUGPFX=msm2-h3                       # take slugs: <SLUGPFX>-t<n>[b|c]-<qwen38-27b-q6k|arete27b-q6k>; nothing hyphenated after q6k (the retyped-path trap, HARDENING L34)
QFILE=/home/witcheer/models/qwen38-27b/Qwen3.8-27B-Q6_K.gguf
QBYTES=22884408288
QSHA=562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727      # = the August oid, byte-identical per item 90a
AFILE=/home/witcheer/models/arete-27b/NousResearch_Arete-27B-Q6_K.gguf
ABYTES=23621581824
ASHA=5b4588c250b941d8dcbbfc610eb43d8303b83270229fc6da5b5ae174f52a23c3      # = the HF LFS oid of bartowski/NousResearch_Arete-27B-GGUF, fetched 2026-11-07
ALIAS=Qwen3.8-27B-Q6_K.gguf           # ONE alias for both arms (confound 4)
PORT=8090; export PORT              # run_agent.sh reads PORT (default 8090): the server, the SIGSTOP hold and the Hermes profile all use this one port
CTX=131072; WALL=60; TURNS=20; KV=q8_0
VS_SECS=30; VS_PRESET=slow
SAMPLING_FLAGS="--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0"
PARAMS_SRC="GGUF general.sampling keys of BOTH files (temp 1.0, top_p 0.95, top_k 20), read 2026-11-07 from the two Q6_K files; Arete generation_config.json (https://huggingface.co/NousResearch/Arete-27B, fetched 2026-11-07) says the same; no min_p anywhere, so --min-p 0 for both (vLLM/SGLang default; the older Qwen rows ran llama.cpp's 0.05); RB 16384 (series value)"
RB=16384
RB_MSG="Thinking budget reached. Stop planning and write the files now with your tools; refine in later turns."
SAT_CUT=1794085200       # Sat 2026-11-07 22:00 CET: no new take after this
SAT_END=1794091200       # Sat 2026-11-07 23:40 CET: GPU work over (09z30 duck roller is NOT_BEFORE Sun 00:00)
SUN0=1794092400          # Sun 2026-11-08 00:00 CET: the Saturday limits no longer apply (resume after the duck items)
REGIME="duel regime, identical for both arms: ctx CTXVAL q8_0 KV, no drafting (no --spec flags), sampling temp 1.0 top-p 0.95 top-k 20 min-p 0, RB 16384, thinking on, one alias, harness h3, prompt msm-v2, wall 60 min, 20 goal turns, llama.cpp 839a572a4; files: Unsloth Q6_K vs bartowski Q6_K (different imatrix); the older h2/MTP Qwen rows are context only"

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0})"
nl_check_path "$BIN" "$GPY" "$QFILE" "$AFILE" "$V/scripts/run_agent.sh" "$QG" "$GATE" "$V/scripts/vx_stall.py" "$V/scripts/metrics.py" \
  "$V/prompts/$PVER.txt" "$V/scripts/orbit_clip.py" "$V/scripts/orbit_clip_multidrag.py" "$V/scripts/orbit_pick.py" "$V/lib/libasound.so.2" "$V/ffmpeg" \
  "$SUMMARY" "$VSBUILD" "$S/PREREG.md" "$S/FROZEN.sha256" /home/witcheer/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome
# frozen set (captured into a variable and tested with [[ ]], never a pipe into grep -q under pipefail): prompt, quick gate, gate, summary script, PREREG, driver + stall detector, vs build
FZ=$(cd / && sha256sum -c "$S/FROZEN.sha256" 2>&1) && FZRC=0 || FZRC=1
if [ "$FZRC" = 0 ]; then nl_say "  ok: frozen set verified ($(echo "$FZ" | wc -l) files): prompt, quick gate, gate, summary script, PREREG, driver, stall detector, vs build"; else nl_fail "FROZEN.sha256 mismatch: $(echo "$FZ" | grep -v ': OK' | head -3 | tr '\n' ' ')"; fi
for pair in "$QFILE:$QBYTES" "$AFILE:$ABYTES"; do
  f=${pair%%:*}; b=${pair##*:}; SZ=$(stat -c %s "$f" 2>/dev/null || echo 0)
  if [ "$SZ" = "$b" ]; then nl_say "  ok: $(basename "$f") $SZ bytes (matches the recorded size)"; else nl_fail "$(basename "$f") size $SZ != $b"; fi
done
INC=$(ls "$(dirname "$AFILE")"/.cache/huggingface/download/*Q6_K*.incomplete 2>/dev/null || true)
[ -z "$INC" ] || nl_fail "an .incomplete download marker exists for the Arete Q6_K file"
"$GPY" -c "import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$QG" "$GATE" "$V/scripts/vx_stall.py" "$SUMMARY" || nl_fail "gate/summary script syntax"
bash -n "$V/scripts/run_agent.sh" || nl_fail "run_agent.sh syntax"
bash -n "$VSBUILD" || nl_fail "vs build script syntax"
HELP=$($BIN --help 2>&1 || true)
for f in --reasoning-budget --reasoning-budget-message --temp --top-p --top-k --min-p --cache-type-k --cache-type-v --flash-attn --parallel --alias --metrics --ctx-size --n-gpu-layers; do
  case "$HELP" in *"$f"*) ;; *) nl_fail "flag $f not in this build";; esac
done
nl_say "  ok: server flags exist in build: $SAMPLING_FLAGS --reasoning-budget(-message) --cache-type-k/v $KV --flash-attn --parallel 1 (and no --spec-type: neither arm drafts)"
[ -n "$PARAMS_SRC" ] || nl_fail "PARAMS_SRC empty: identify the maker parameters first"
nl_say "  params: src='$PARAMS_SRC' flags='$SAMPLING_FLAGS'"
VER=$($BIN --version 2>&1 || true)
if [[ "$VER" == *839a572* ]]; then nl_say "  ok: llama.cpp 839a572"; else nl_say "  llama.cpp build changed or missing: ${VER:0:120}"; exit 2; fi
nl_check_disk 20
nl_say "  regime: ${REGIME//CTXVAL/$CTX}"
NOW=${NL_NOW:-$(date +%s)}
HARD=$(nl_deadline_epoch)
if [ "$NOW" -lt "$SUN0" ]; then DL=$(( HARD < SAT_END ? HARD : SAT_END )); CUT=$SAT_CUT; else DL=$HARD; CUT=$HARD; fi
LEFT=$(( (DL-NOW)/60 ))
TAKES=("1:q" "1:a" "2:q" "2:a")
arm_vars() { # q|a -> ASLUG ALABEL AFILE_CUR (the arm's file)
  case "$1" in
    q) ASLUG=qwen38-27b-q6k; ALABEL="Qwen3.8-27B Q6_K"; AFILE_CUR=$QFILE;;
    a) ASLUG=arete27b-q6k;   ALABEL="Arete-27B Q6_K";   AFILE_CUR=$AFILE;;
  esac
}
choose_slug() { # n arm -> CH_STATE (done|new|exhausted), CH_SLUG
  local n=$1 arm=$2 suf s prof
  arm_vars "$arm"; CH_STATE=exhausted; CH_SLUG=""
  for suf in "" b c; do
    s="$SLUGPFX-t${n}${suf}-$ASLUG"
    if [ -f "$V/runs/$s/gate/gate.json" ]; then CH_STATE=done; CH_SLUG=$s; return 0; fi
  done
  for suf in "" b c; do
    s="$SLUGPFX-t${n}${suf}-$ASLUG"; prof=vx$(echo "$s" | tr -cd 'a-z0-9')
    if [ ! -e "$V/runs/$s" ] && [ ! -d "/home/witcheer/.hermes/profiles/$prof" ]; then CH_STATE=new; CH_SLUG=$s; return 0; fi
  done
  return 0
}
NRUN=0
for t in "${TAKES[@]}"; do
  n=${t%%:*}; arm=${t##*:}; choose_slug "$n" "$arm"
  case "$CH_STATE" in
    done) nl_say "  SKIP: take $n $arm done ($CH_SLUG): $V/runs/$CH_SLUG/gate/gate.json";;
    new) NRUN=$((NRUN+1)); nl_say "  RUN: take $n $arm -> $CH_SLUG (artifact $V/runs/$CH_SLUG/gate/gate.json)";;
    *) nl_fail "take $n $arm: three attempts (t$n, t${n}b, t${n}c) already used without a gate.json; look at ~/voxelbench/runs/$SLUGPFX-t$n*-$ASLUG by hand";;
  esac
done
VSDIR=$V/runs/vs-qwen-arete97
VSOUT=$VSDIR/vsa97-discord.mp4
NEED_VS=0; [ -s "$VSOUT" ] || NEED_VS=1
nl_say "  planned: $NRUN take(s) to run, vs clip $([ $NEED_VS = 1 ] && echo pending || echo done), summary rewritten at every exit; hard stop $(date -d "@$DL" '+%a %H:%M') local, $LEFT min away; no new take after $(date -d "@$CUT" '+%a %H:%M')"
if [ "$NRUN" -eq 0 ] && [ "$NEED_VS" = 0 ]; then nl_say "nothing to run, item complete"; exit 0; fi
if [ "$NRUN" -gt 0 ]; then
  [ "$LEFT" -ge 90 ] || { nl_say "  under 90 min left in the window: not starting (a take needs ~65 min plus gate and clips; exit 3, resumable)"; exit 3; }
  [ "$NOW" -lt "$CUT" ] || { nl_say "  past the no-new-take cutoff ($(date -d "@$CUT" '+%a %H:%M')): not starting (exit 3, resumable)"; exit 3; }
fi
if [ "${DRYRUN:-0}" = "1" ] || [ ! -f "$MANIFEST" ]; then nl_manifest_init t1q t1a t2q t2a vsclip summary; fi
nl_gate

# ---------------------------------------------------------------- RUN path (card work starts below this line) ----------
SRV=""; TRACER=""; VRAMLOG=""; DSTOP=0; T_CARD0=0
mem_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }
killsrv() { [ -n "$SRV" ] && kill -9 "$SRV" 2>/dev/null; SRV=""; }   # only the server THIS item started
stop_helpers() { [ -n "$TRACER" ] && kill "$TRACER" 2>/dev/null; TRACER=""; [ -n "$VRAMLOG" ] && kill "$VRAMLOG" 2>/dev/null; VRAMLOG=""; }
wait_card_clear() { # <label>: memory under 1 GiB or false
  local i v=""; for i in $(seq 1 120); do v=$(mem_used); [ "${v:-99999}" -lt 1024 ] && { nl_say "  card clear (${v} MiB) $1"; return 0; }; sleep 2; done
  nl_say "  card memory still ${v:-?} MiB after 240 s $1"; return 1
}
donald_back() { # restore + health poll (rule 2: grep ok, here a captured match)
  nl_donald_restore; local i h=""; for i in $(seq 1 60); do h=$(curl -s -m 3 127.0.0.1:$PORT/health || true); [[ "${h,,}" == *ok* ]] && { nl_say "  Donald health ok"; return 0; }; sleep 2; done
  nl_say "  WARN Donald health not ok after 120 s: $h"; return 1
}
release_card() { # idempotent: helpers, server, Donald
  stop_helpers; tmux kill-session -t "=voxel-${CUR_SLUG:-none}" 2>/dev/null; killsrv
  if [ "$DSTOP" = 1 ]; then sleep 5; nl_say "releasing the card: restoring Donald"; DSTOP=0; donald_back; nl_say "card released after $(( ($(date +%s)-T_CARD0)/60 )) min (drain to restore), the rest is CPU"; fi
}
trap release_card EXIT
write_summary() { "$GPY" "$SUMMARY" --runs "$V/runs" --prefix "$SLUGPFX" --out "$S/run" 2>&1 | tail -1 | cut -c1-400; }
rule3_check() { # rule 3, live: nothing but Donald's own process tree may hold the card when we drain him
  local dpid apps p q own foreign=""
  dpid=$(systemctl show -p MainPID --value llama-server.service 2>/dev/null || echo 0)
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | tr '\n' ' ')
  for p in $apps; do
    own=0; q=$p
    while [ -n "$q" ] && [ "$q" != 1 ] && [ "$q" != 0 ]; do [ "$q" = "$dpid" ] && { own=1; break; }; q=$(ps -o ppid= -p "$q" 2>/dev/null | tr -d ' '); done
    [ "$own" = 1 ] || foreign="$foreign $p"
  done
  nl_say "  rule 3, live: compute apps [${apps:-none}], Donald pid ${dpid:-0}, foreign [${foreign:- none}]; tmux: $(tmux ls -F '#{session_name}' 2>/dev/null | tr '\n' ' ')"
  [ -z "$foreign" ] || { nl_say "  a foreign process holds the card: not draining (exit 3, resumable)"; exit 3; }
}
can_start() { # a new take needs the cutoff not passed and >= 90 min left
  local now left; now=${NL_NOW:-$(date +%s)}; left=$(( (DL-now)/60 ))
  [ "$now" -lt "$CUT" ] || { nl_say "  past the no-new-take cutoff: pausing here"; return 1; }
  [ "$left" -ge 90 ] || { nl_say "  only $left min left in the window: pausing here"; return 1; }
  return 0
}

run_take() { # n arm -> 0 ok (gate.json written, pass or fail), 1 harness problem
  local n=$1 arm=$2 ph="t${n}${arm}" up=0 i h SCHK SPEC t0 HTML grc OPX OSC
  arm_vars "$arm"; choose_slug "$n" "$arm"
  [ "$CH_STATE" = new ] || return 0
  local SLUG=$CH_SLUG MODEL_FILE=$AFILE_CUR; CUR_SLUG=$SLUG
  local ART=$V/runs/$SLUG/gate/gate.json
  nl_say "=== take $n $arm: $ALABEL -> $SLUG"
  mem=$(mem_used); [ "${mem:-99999}" -lt 4000 ] || { wait_card_clear "before the $ph load" || { nl_mark "$ph" failed "card not clear before load"; return 1; }; }
  mkdir -p "$V/runs/$SLUG/work"; t0=$(date +%s); nl_mark "$ph" running "$SLUG"
  # shellcheck disable=SC2086
  $BIN --model "$MODEL_FILE" --host 127.0.0.1 --port $PORT --n-gpu-layers 99 --ctx-size $CTX --parallel 1 --flash-attn on \
    --cache-type-k $KV --cache-type-v $KV --alias "$ALIAS" --metrics --reasoning-budget $RB --reasoning-budget-message "$RB_MSG" $SAMPLING_FLAGS \
    > "$V/runs-$SLUG-server.log" 2>&1 &
  SRV=$!
  for i in $(seq 1 240); do h=$(curl -s -m 3 127.0.0.1:$PORT/health || true); [[ "${h,,}" == *ok* ]] && { up=1; break; }; kill -0 "$SRV" 2>/dev/null || break; sleep 2; done
  if [ "$up" != 1 ]; then
    nl_say "  server did not come up at ctx $CTX: $(grep -a -m1 -iE 'out of memory|failed|error|unknown' "$V/runs-$SLUG-server.log" | head -c 200)"
    killsrv; nl_mark "$ph" failed "server did not come up at ctx $CTX (one rung only, see runs-$SLUG-server.log)"; return 1
  fi
  nl_say "  server up ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)), ctx $CTX"
  mkdir -p "$V/runs/$SLUG"
  ( while :; do echo "$(date +%s),$(mem_used)" >> "$V/runs/$SLUG/vram-samples.csv"; sleep 20; done ) &
  VRAMLOG=$!
  curl -s 127.0.0.1:$PORT/props > "$V/runs/$SLUG/props.json"
  SCHK=$(python3 - "$V/runs/$SLUG/props.json" <<'PYEOF'
import json, sys
g = json.load(open(sys.argv[1])).get("default_generation_settings", {}); p = g.get("params", g)
want = {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0}
got = {k: p.get(k) for k in want}
ok = all(got[k] is not None and abs(float(got[k]) - v) < 1e-3 for k, v in want.items())
print(("OK " if ok else "BAD ") + json.dumps(got))
PYEOF
)
  nl_say "  server sampling defaults: $SCHK"
  case "$SCHK" in OK*) ;; *) stop_helpers; killsrv; nl_mark "$ph" failed "server sampling defaults wrong: $SCHK"; return 1;; esac
  SPEC=$(grep -a -c -E 'MTP draft context|draft-mtp|--spec-type|common_speculative_init' "$V/runs-$SLUG-server.log" || true)
  nl_say "  drafting lines in the server log: ${SPEC:-0} (want 0: no drafting for either arm)"
  [ "${SPEC:-0}" = 0 ] || { stop_helpers; killsrv; nl_mark "$ph" failed "drafting context created ($SPEC log lines)"; return 1; }
  local VERS PMD5 SHAV; VERS=$($BIN --version 2>&1 | grep -m1 -i 'version' | tr -d '"' || true); PMD5=$(md5sum "$V/prompts/$PVER.txt" | cut -c1-12)
  if [ "$arm" = q ]; then SHAV=$QSHA; else SHAV=$ASHA; fi
  python3 - "$V/runs/$SLUG/run-config.json" "$MODEL_FILE" "$CTX" "$KV" "$SAMPLING_FLAGS" "$RB" "$VERS" "$PVER" "$PMD5" "$PARAMS_SRC" "${REGIME//CTXVAL/$CTX}" "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)" "$ALIAS" "$SHAV" "$arm" "$n" <<'PYEOF'
import json, sys
a = sys.argv
json.dump({"model_file": a[2], "ctx": int(a[3]), "kv": a[4], "sampling": a[5], "reasoning_budget": int(a[6]), "build": a[7], "prompt": a[8] + ".txt", "prompt_md5": a[9],
           "params_src": a[10], "regime": a[11], "vram_at_load": a[12], "alias": a[13], "file_sha256": a[14], "arm": a[15], "take": int(a[16]), "drafting": "none",
           "harness": "h3 (VX_TMPFIX, VX_GATE_STRICT, VX_STALL_NUDGE, quick_gate_msm.py)", "item": "97-arete-msm"}, open(a[1], "w"), indent=2)
PYEOF
  ( while :; do curl -s -m5 127.0.0.1:$PORT/slots 2>/dev/null | python3 -c "
import json,sys,time
try:
    d=json.load(sys.stdin); s=d[0] if isinstance(d,list) else d; p=s.get('params',{})
    print(json.dumps({'t':int(time.time()),'processing':s.get('is_processing'),'id_task':s.get('id_task'),**{k:p.get(k) for k in ('temperature','top_k','top_p','min_p','n_predict')}}))
except Exception as e: print(json.dumps({'t':int(time.time()),'err':str(e)[:80]}))
" >> "$V/runs/$SLUG/sampling-trace.jsonl"; sleep 60; done ) &
  TRACER=$!
  export DEADLINE_EPOCH=$(( DL - 600 ))
  local PROMPT_RUN=$V/runs/$SLUG/prompt.txt
  { cat "$V/prompts/$PVER.txt"; printf '\nThe working directory is %s. The final file must be exactly %s/index.html (copy this path, never retype it).\n' "$V/runs/$SLUG/work" "$V/runs/$SLUG/work"; } > "$PROMPT_RUN"
  VX_TMPFIX=1 VX_GATE_STRICT=1 VX_STALL_NUDGE=1 GATE_CMD="$GPY $QG $V/runs/$SLUG/work/index.html $V/runs/$SLUG/qgate --video-seconds 0" \
    bash "$V/scripts/run_agent.sh" "$SLUG" "$ALIAS" $CTX $WALL $TURNS "$PROMPT_RUN" >> "$RUNLOG" 2>&1
  nl_say "  driver: $(cat "$V/runs/$SLUG/driver.json" 2>/dev/null)"
  DEND=$(python3 -c "import json;print(json.load(open('$V/runs/$SLUG/driver.json')).get('end',''))" 2>/dev/null || true)
  if [ "$DEND" = gate_setup_failed ]; then   # the driver ended the run itself (VX_GATE_STRICT): a harness fault, never a model result
    stop_helpers; killsrv; sleep 3; mv "$V/runs/$SLUG" "$V/runs/$SLUG.harness-fail-$(date +%s)"
    nl_mark "$ph" failed "gate setup failed (VX_GATE_STRICT), run moved aside, take not counted"; return 2
  fi
  sha256sum "$V/scripts/run_agent.sh" "$V/scripts/vx_stall.py" "$V/scripts/quick_gate_msm.py" "$V/scripts/gate.py" "$V/scripts/orbit_pick.py" "$V/scripts/orbit_clip.py" "$V/scripts/orbit_clip_multidrag.py" > "$V/runs/$SLUG/harness-sha256.txt" 2>/dev/null || true
  [ -f "$V/runs/$SLUG/path-miss.json" ] && nl_say "  PATH MISS flag present: $(cat "$V/runs/$SLUG/path-miss.json")"
  stop_helpers
  nl_say "  sampling trace: $(python3 -c "
import json
rows=[json.loads(l) for l in open('$V/runs/$SLUG/sampling-trace.jsonl') if l.strip()]
live=[r for r in rows if r.get('temperature') is not None]
sets=sorted({(r['temperature'],r['top_k'],round(r['top_p'],3),r['min_p']) for r in live})
print(len(rows),'samples,',len(live),'with params, distinct (temp,top_k,top_p,min_p):',sets)
" 2>&1 | head -c 300)"
  curl -s 127.0.0.1:$PORT/metrics | grep -E "^llamacpp:(tokens_predicted_total|prompt_tokens_total|tokens_predicted_seconds_total|prompt_seconds_total)" > "$V/runs/$SLUG/server-metrics.txt" 2>/dev/null
  nl_say "  vram during the run (MiB, peak of $(wc -l < "$V/runs/$SLUG/vram-samples.csv" 2>/dev/null) samples): $(sort -t, -k2 -n "$V/runs/$SLUG/vram-samples.csv" 2>/dev/null | tail -1 | cut -d, -f2) (card total $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null))"
  killsrv; sleep 3
  HTML=$V/runs/$SLUG/work/index.html
  [ -f "$HTML" ] || HTML=$(ls -t "$V/runs/$SLUG"/work/*.html 2>/dev/null | head -1 || true)
  if [ -z "$HTML" ] || [ ! -f "$HTML" ]; then
    local STRAY; STRAY=$(find "$V/runs" -name '*.html' -newermt "@$t0" -size +1k -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2- || true)
    if [ -n "$STRAY" ]; then HTML=$STRAY; echo "{\"path_miss\": true, \"found\": \"$STRAY\"}" > "$V/runs/$SLUG/path-miss.json"; nl_say "  PATH MISS: no html in work/, grading the fresh file at $STRAY (flagged)"; fi
  fi
  mkdir -p "$V/runs/$SLUG/gate"
  if [ -n "$HTML" ] && [ -f "$HTML" ]; then
    setsid timeout 300 $GPY $GATE "$HTML" "$V/runs/$SLUG/gate" --video-seconds 20 < /dev/null >> "$RUNLOG" 2>&1; grc=$?
    if [ "$grc" = 2 ]; then nl_say "  gate harness error (rc 2): one retry"; sleep 5; setsid timeout 300 $GPY $GATE "$HTML" "$V/runs/$SLUG/gate" --video-seconds 20 < /dev/null >> "$RUNLOG" 2>&1; grc=$?; fi
  else
    echo '{"pass": false, "no_html": true}' > "$ART"; grc=1
  fi
  python3 "$V/scripts/metrics.py" /home/witcheer/.hermes/profiles/$(python3 -c "import json;print(json.load(open('$V/runs/$SLUG/driver.json'))['profile'])" 2>/dev/null || echo "vx$(echo "$SLUG" | tr -cd 'a-z0-9')") "$V/runs/$SLUG/metrics.json" >> "$RUNLOG" 2>&1
  if nl_judge_artifact "$ph" 0 "$ART" "$t0" "$(( ($(date +%s)-t0)/60 )) min, gate rc $grc, $SLUG"; then
    nl_say "  gate: $(python3 -c "import json;d=json.load(open('$ART'));print('pass' if d.get('pass') else 'FAIL', d.get('checks'), 'fps', d.get('fps'), 'bytes', d.get('bytes'), d.get('error') or d.get('harness_error') or '')")"
    if python3 -c "import json,sys;sys.exit(0 if json.load(open('$ART')).get('pass') else 1)"; then
      mkdir -p "$V/runs/$SLUG/showcase"
      setsid timeout 120 $GPY "$V/scripts/orbit_pick.py" "$HTML" --clip-seconds 30 < /dev/null 2>>"$RUNLOG" | tail -1 > "$V/runs/$SLUG/showcase/orbit-choice.json"
      OPX=$(python3 -c "import json;print(json.load(open('$V/runs/$SLUG/showcase/orbit-choice.json'))['orbit_px'])" 2>/dev/null || echo 1257)
      OSC=$(python3 -c "import json;print(json.load(open('$V/runs/$SLUG/showcase/orbit-choice.json'))['script'])" 2>/dev/null || echo orbit_clip.py)
      case "$OSC" in orbit_clip.py|orbit_clip_multidrag.py) ;; *) OSC=orbit_clip.py;; esac
      nl_say "  orbit choice: $OSC --orbit-px $OPX ($(python3 -c "import json;d=json.load(open('$V/runs/$SLUG/showcase/orbit-choice.json'));print(d.get('method'),d.get('var'),d.get('units'))" 2>/dev/null))"
      setsid timeout 240 $GPY "$V/scripts/$OSC" "$HTML" "$V/runs/$SLUG/showcase" --seconds 30 --orbit-px "$OPX" --keys "n@11,n@22" < /dev/null >> "$RUNLOG" 2>&1 \
        && nl_say "  showcase: $V/runs/$SLUG/showcase/orbit.mp4" || nl_say "  showcase clip failed (gate result stands; the vs clip needs another passing take of this model)"
    fi
  else
    nl_say "  no fresh gate.json after the take: harness problem, not a model result"; return 1
  fi
  return 0
}

DRAINED=0; PAUSED=0; FAILED=0; first=1
for t in "${TAKES[@]}"; do
  n=${t%%:*}; arm=${t##*:}; choose_slug "$n" "$arm"
  [ "$CH_STATE" = done ] && { nl_mark "t${n}${arm}" done "already done: $CH_SLUG"; continue; }
  [ "$CH_STATE" = new ] || { nl_say "take $n $arm: attempts exhausted"; FAILED=1; break; }
  can_start || { PAUSED=1; break; }
  if [ "$DRAINED" = 0 ]; then
    nl_say "weights sha256 (CPU, before the drain, Donald still serving)"
    if [ "${A97_NO_SHA:-0}" != 1 ]; then
      QH=$(sha256sum "$QFILE" | cut -d' ' -f1); AH=$(sha256sum "$AFILE" | cut -d' ' -f1)
      if [ "$QH" = "$QSHA" ] && [ "$AH" = "$ASHA" ]; then nl_say "  sha256 ok: both files match the recorded oids"; else nl_say "  FAIL sha256: qwen $QH arete $AH"; nl_mark "t${n}${arm}" failed "weights sha256 mismatch"; exit 1; fi
    fi
    rule3_check
    T_CARD0=$(date +%s); nl_say "draining Donald (night-lib nl_donald_stop)"; nl_donald_stop; DSTOP=1; DRAINED=1
    wait_card_clear "after the drain" || { nl_mark "t${n}${arm}" failed "Donald drain never cleared: another process may own the card"; exit 1; }
  fi
  rc=0
  for attempt in 1 2; do
    run_take "$n" "$arm"; rc=$?
    [ "$rc" = 2 ] || break
    nl_say "take $n $arm: harness gate-setup failure on attempt $attempt, retrying under a fresh slug"
    killsrv; wait_card_clear "before the retry" || { rc=1; break; }
    choose_slug "$n" "$arm"; [ "$CH_STATE" = new ] || { rc=1; break; }
    can_start || { rc=3; break; }
  done
  [ "$rc" = 3 ] && { PAUSED=1; break; }
  [ "$rc" = 0 ] || { FAILED=1; break; }
  killsrv; wait_card_clear "after take $n $arm" || { FAILED=1; break; }
done
release_card
CUR_SLUG=none
nl_say "summary: $(write_summary)"
if [ "$FAILED" = 1 ]; then nl_mark summary failed "a take hit a harness problem, see the lines above"; exit 1; fi
if [ "$PAUSED" = 1 ]; then nl_mark summary pending "paused: out of window, takes remain"; nl_say "paused with takes remaining (exit 3, resumable)"; exit 3; fi
nl_mark summary done "$S/run/summary.json"

# ---- CPU only from here: the two-pane vs clip (pre-registered pair), Donald already back ----
PAIROK=$(python3 -c "import json;print(json.load(open('$S/run/pair.json')).get('ok'))" 2>/dev/null || echo False)
if [ -s "$VSOUT" ]; then nl_mark vsclip done "$VSOUT"; nl_say "vs clip already built: $VSOUT"
elif [ "$PAIROK" = True ]; then
  nl_mark vsclip running
  PAIR_JSON=$S/run/pair.json OUT=$VSDIR SECS=$VS_SECS PRESET=$VS_PRESET bash "$VSBUILD" >> "$RUNLOG" 2>&1 </dev/null; vrc=$?
  if [ "$vrc" = 0 ] && [ -s "$VSOUT" ]; then nl_mark vsclip done "$VSOUT"; nl_say "vs clip: $VSOUT ($(md5sum "$VSOUT" | cut -c1-12))"; else nl_mark vsclip failed "vs build rc $vrc"; nl_say "vs clip build failed rc $vrc (rerun: PAIR_JSON=$S/run/pair.json bash $VSBUILD)"; fi
else
  nl_mark vsclip skipped "no passing take with an orbit clip for both models (see summary.json pair)"; nl_say "no vs clip: a model has no passing take with an orbit clip "
fi
nl_say "verdict: $(python3 -c "import json;print(json.load(open('$S/run/summary.json'))['verdict']['text'])" 2>/dev/null)"
nl_say "DONE 97-arete-msm"
exit 0
