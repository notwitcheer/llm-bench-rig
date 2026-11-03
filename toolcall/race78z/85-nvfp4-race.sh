#!/bin/bash
# 85-nvfp4-race.sh (staged as 78z, renamed 2026-11-03 so it sorts after 84): idle-queue item (enqueued 2026-11-03 on the operator yes; 2026-11-03, mercury subagent for the GPU thread). The RACE LEG of item 78: operator, GPU thread, 2026-11-02 ~19:35 CET, decision A a:
#   "keep 78 and add a short race leg, the two files side by side on one Hermes task at real speed, so the post has a picture" (vault 99-personal/benchmark-rig/mojo-slate-2026-11-02.md); restaged
#   2026-11-03 ~09:35 CET after the first staging child died and left nothing. Item 78 itself finished 2026-11-03 05:26 CET: Qwen3.8-27B NVFP4 (vLLM 0.32.0) vs UD-Q4_K_M (llama.cpp b10371), tool calls
#   TIE 85 vs 80 of 100 (exact McNemar p=0.125), decode 56 vs 78 tok/s (vault 99-personal/benchmark-rig/toolcall-nvfp4-status.md, readout).
# MOJO: two Hermes Agent terminals side by side, the same task on Qwen3.8-27B NVFP4 (vLLM) and UD-Q4_K_M (llama.cpp), live tok/s on each pane, one RTX 5090, two takes with the start synced | never-seen / intense: Blackwell's native 4-bit file racing the file everyone downloads, in the agent you actually use | on an RTX 5090, NVFP4 and the Q4_K_M of Qwen3.8-27B tie on multi-turn tool calls (85 vs 80 of 100, p=0.125): 56 vs 78 tok/s decode (the clip: the same Hermes task, [M:SS] vs [M:SS])
# EV: self-quote of the 10-26 which-file post (11,302 views / 207 bookmarks, 21:40 CET, the week's only breakout and bookmark leader), rule 1 of gpu-ev-gate-2026-10-25 section 2 (a which-file verdict on a brand-pull model),
#   the picture added by decision A a (mojo rule 9a0: the item plans its own postable artefact). bracket 3-6k (item 78's: EAGLE-4 3,261, Humming 3,060, which-file median 4,595 n41). HONEST LIMITS: a race clip
#   on a which-file post has no measured comparable of its own (UNVERIFIED; capsule-rig-ops rule 9a: a strong visual with no brand pull lands near the lane floor, so the which-file verdict is the lead and the
#   clip rides under it); n=1 take per file in the clip (two takes recorded, the clip uses take 1 of each unless a recorder fault, the rule is fixed here before any output exists); the result is the TIE of item 78,
#   the race never turns it into a win; no cross-engine speed multiplier (item 78's pre-registered rail): the card shows 56 vs 78, two numbers labelled vLLM stream vs llama.cpp timings.
# serves: content (a which-file verdict with a picture, extends the 10-26 which-file post) + hardening (a vLLM-backed Hermes harness: the first Hermes session on vLLM here, off the main gateway).
#
# WHAT: ONE real Hermes Agent session per take (own HERMES_HOME, minted launcher, HERMES_DISABLE_LAZY_INSTALLS=1, never the main hermes-gateway.service, never :8090) types and submits the SAME prompt
#   (prompts/voxel-cottage-v1.txt: one animated isometric voxel cottage as a single canvas page, about 5-10 minutes) against a server that holds ONE file at a time: A = unsloth Qwen3.8-27B NVFP4 on stock
#   vLLM 0.32.0 (humming-env, --linear-backend cutlass, the 78 serve flags + ctx 65536 + qwen3_coder tool parser + qwen3 reasoning parser), B = unsloth UD-Q4_K_M on llama.cpp b10371 (--jinja, ctx 65536,
#   one slot). The two servers cannot be resident together (27.9 GB + about 18 GB do not fit 32 GB), so the takes run back to back and are composited with the START SYNCED on each take's own Enter mark:
#   the clip says "two takes, start synced" on its title band. Both takes send identical sampling through the provider's extra_body (temperature 1.0, top_p 0.95, top_k 20, min_p 0.05, seed 721: the Qwen3.8
#   generation_config.json values; the mock-model run logs what Hermes really sends). thinking on (the regime item 78 scored). HUD tok/s = the SERVER's own token count on both panes (llama-server: exact
#   predicted_n of finished requests + n_decoded of the busy slot; vLLM: vllm:generation_tokens_total), the proxy's stream deltas only a cross-check.
# TAKES: R78Z_TAKES per file (default 2), run A,A then B,B (one server start per file). Clip = take 1 of each file (fixed here, before any output), take 2 reported next to it; a take-1 recorder fault
#   (end reason other than done/wall, or no events) falls back to take 2 and the summary says so.
# CLIP PLAN (planned here, not rescued later): runs/<slug>/out/nvfp4-vs-q4km-final.mp4 = race (1920x1080, two 76x22 panes, per-pane live tok/s from the server counter + tokens so far + sparkline on a shared 0-100 scale +
#   clock from each take's own Enter + done chip, real speed when the longer take is <= 240 s, otherwise open 30 s and close 40 s real and the middle sped up with a visible pill) + a 12 s results segment (the two
#   pages side by side, same camera) + an 8 s end card (85 vs 80, 804 vs 802, 56 vs 78, this race, small print: n=1, two takes, noise unmeasured). HUD LEAK READ: the status-bar row is blanked wherever it sits, any
#   cell matching the account name / home path / tailnet address / hostname / e-mail shape is drawn as a block (counts in out/race.mp4.summary.json and each take's leakscan.json), work dirs are /tmp/arena78z/<tag> and HOME is a
#   throwaway dir. out/contact.png = a 3x3 sheet of stills + the results still + the card, for the sceptical read before anything is offered.
# CARD TIME (honest; GPU smoke 2026-11-03 10:05-10:14 CET = 8.7 min of card, logs in ~/voxelbench/runs/race78z-gpu-smoke-2026-11-03/): measured drain 5 s, vLLM start 80 s (item 78: 70-85 s), tool-call smoke 1-2 s, llama.cpp load 10 s
#   (page cache warm), restore 10 s. a take = boot 12 s + the agent + 20 s quiet tail. the agent's length is UNMEASURED: in a 3 min wall-capped take neither file had finished its first (thinking) request, NVFP4 at
#   9,373 server tokens (51.5 tok/s while generating) and UD-Q4_K_M at 13,835 (75.8), no page yet; item 80 saw ~18.5k-token first requests, so a take is GUESSED at 8 min on NVFP4 and 5 min on UD-Q4_K_M.
#   typical ~31 min for 2+2 takes, worst case (every take at the 12 min wall cap) ~58 min. CPU after the card is released: gates ~1.5 min, render + results + card 2-10 min (1.2 min for a 3 min clip).
# WINDOW RULE: needs >= 75 min left in the block (58 worst + 17 slack) or exits 3 without draining Donald (stays queued, resumable). a take is skipped when under 9 min remain, and its wall is cut to leave 6 min of margin.
#   SORT: 78z sorts BEFORE 79 in `ls | sort`, so in the next HC block it runs FIRST. rename to a higher prefix (e.g. 83z-) at enqueue if 79/80/82 should keep their slots (the status note says what each costs).
#
# test knobs (empty in production): R78Z_SMOKE=1 (slug -smoke, 1 take per file, wall 3 min, no window rule, run by hand in tmux with NL_HARD_STOP_EPOCH set), R78Z_TAKES, R78Z_WALL_MIN, R78Z_THINK=on|off, R78Z_QUIET (20),
#   R78Z_STUB_VLLM=http://127.0.0.1:PORT and R78Z_STUB_LLAMA=http://127.0.0.1:PORT (CPU mocks: no Donald drain, no server start, DRYRUN-safe), R78Z_VLLM_NOPREFIX=1 (adds --no-enable-prefix-caching as 78 did),
#   R78Z_MODEL_Q4 / R78Z_MODEL_NV (red-path tests), MANIFEST_OVERRIDE (staging DRYRUN only).
set -uo pipefail
NL_TAG=race78z
MANIFEST=/home/witcheer/voxel-race78z-status.json
[ -n "${MANIFEST_OVERRIDE:-}" ] && MANIFEST=$MANIFEST_OVERRIDE
export MANIFEST
. /home/witcheer/night-lib.sh
V=/home/witcheer/voxelbench
RA=$V/race78z
RES=/home/witcheer/benchmark-rig/results/toolcall-v1
GPY=/home/witcheer/benchmark-rig/.venv/bin/python          # playwright (r78z_gate.py, the CPU liveness gate)
RPY=/home/witcheer/scratch/race/venv/bin/python             # pyte + PIL + fontTools (recorder + renderer)
FF=$V/ffmpeg
CHECKOUT=/home/witcheer/.hermes/hermes-agent
HBIN=$CHECKOUT/.hermes/bin/hermes
HERMES_PIN=9bda188
MAIN=/home/witcheer/llama.cpp-nl/build/bin/llama-server
VENV=/home/witcheer/humming-env
NVDIR=${R78Z_MODEL_NV:-/home/witcheer/models/qwen38-27b-nvfp4}
Q4=${R78Z_MODEL_Q4:-/home/witcheer/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_M.gguf}
NV_BYTES=22568192096; Q4_BYTES=16464440224
NV_NAME=qwen38-27b-nvfp4; Q4_NAME=Qwen3.8-27B-UD-Q4_K_M
LPORT=8197; VPORT=8198; PPORT=8291                         # own ports: never :8090 (llama-swap / Donald), never 8097/8098 (item 78) or 819x (item 80)
CTXCFG=65536                                               # Hermes needs >= 64,000; vLLM KV capacity measured 144,725 tokens at fp8 (78 log), llama.cpp f16 KV 65536 fits next to the 16.5 GB file
PROMPT=$RA/prompts/voxel-cottage-v1.txt
SMOKE=${R78Z_SMOKE:-0}
TAKES=${R78Z_TAKES:-2}; WALLMIN=${R78Z_WALL_MIN:-12}; THINK=${R78Z_THINK:-on}; QUIET=${R78Z_QUIET:-20}
[ "$SMOKE" = 1 ] && { TAKES=${R78Z_TAKES:-1}; WALLMIN=${R78Z_WALL_MIN:-3}; }
SLUG=race78z-qwen38-nvfp4-vs-udq4km; [ "$SMOKE" = 1 ] && SLUG=$SLUG-smoke
SVL=${R78Z_STUB_VLLM:-}; SLL=${R78Z_STUB_LLAMA:-}
STUB=0; { [ -n "$SVL" ] || [ -n "$SLL" ]; } && STUB=1
RUN=$V/runs/$SLUG
ART=$RUN/summary.json
REGIME="thinking $THINK, temp 1.0 / top-p 0.95 / top-k 20 / min-p 0.05 / seed 721 via extra_body on both; Hermes terminal+file toolsets, approvals off, memory off, ctx 65536; vLLM 0.32.0 stock (humming-env) --linear-backend cutlass, FP8 KV as shipped, prefix caching at the engine default, qwen3_coder + qwen3 parsers; llama.cpp b10371 --jinja, one slot, f16 KV; one agent per server, NOT the series speed regime"

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0} smoke=$SMOKE takes=$TAKES wall=${WALLMIN}min stub=${STUB})"
nl_check_path "$GPY" "$RPY" "$HBIN" "$FF" "$PROMPT" "$MAIN" "$VENV/bin/python" "$NVDIR/config.json" "$NVDIR/model.safetensors" "$NVDIR/tokenizer.json" "$NVDIR/chat_template.jinja" "$Q4" \
  "$RA/r78z_take.sh" "$RA/r78z_proxy.py" "$RA/r78z_record.py" "$RA/r78z_mkhome.py" "$RA/r78z_render.py" "$RA/r78z_results.py" "$RA/r78z_gate.py" "$RA/r78z_facts.py" "$RA/r78z_summary.py" \
  "$RES/nvfp4-vs-q4km-paired.json" "$RES/qwen3-8-27b-nvfp4-vllm-s5-on/summary.json" "$RES/qwen3-8-27b-ud-q4-k-m-s5-on/summary.json" "$V/lib/libasound.so.2" /home/witcheer/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome
if [ -z "${R78Z_MODEL_NV:-}" ]; then
  [ "$(stat -c %s "$NVDIR/model.safetensors" 2>/dev/null)" = "$NV_BYTES" ] && nl_ok "NVFP4 weights size $NV_BYTES" || nl_fail "NVFP4 model.safetensors size != $NV_BYTES"
else nl_say "  NVFP4 dir override (test): $NVDIR"; fi
if [ -z "${R78Z_MODEL_Q4:-}" ]; then
  [ "$(stat -c %s "$Q4" 2>/dev/null)" = "$Q4_BYTES" ] && nl_ok "UD-Q4_K_M size $Q4_BYTES" || nl_fail "UD-Q4_K_M size != $Q4_BYTES"
else nl_say "  UD-Q4_K_M override (test): $Q4"; fi
"$GPY" -c "import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$RA/r78z_gate.py" "$RA/r78z_proxy.py" "$RA/r78z_record.py" "$RA/r78z_mkhome.py" "$RA/r78z_facts.py" "$RA/r78z_summary.py" || nl_fail "harness syntax"
"$RPY" -c "import ast,sys,pyte,PIL,fontTools; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$RA/r78z_render.py" "$RA/r78z_results.py" || nl_fail "renderer deps/syntax (pyte, PIL, fontTools in $RPY)"
bash -n "$RA/r78z_take.sh" || nl_fail "r78z_take.sh syntax"
# item 78's numbers must be the ones the end card quotes (r78z_facts.py exits 2 otherwise); this is also the card's source read, done now so a moved result file fails the plan, not the render
FACTS_CHECK=$($RPY - "$RES" <<'PYEOF'
import json, sys, os
res = sys.argv[1]
p = json.load(open(os.path.join(res, "nvfp4-vs-q4km-paired.json")))
s5 = p["pairs"]["s5-on: nvfp4 vs q4km"]
nv = json.load(open(os.path.join(res, "qwen3-8-27b-nvfp4-vllm-s5-on/summary.json")))["speed"]["decode_tok_s_median"]
q4 = json.load(open(os.path.join(res, "qwen3-8-27b-ud-q4-k-m-s5-on/summary.json")))["speed"]["decode_tok_s_median"]
ok = (s5["a_pass"], s5["b_pass"], round(nv), round(q4)) == (85, 80, 56, 78) and s5["p_exact"] == 0.125
print(("OK " if ok else "CHANGED ") + "s5 %s vs %s p=%s decode %s vs %s" % (s5["a_pass"], s5["b_pass"], s5["p_exact"], nv, q4))
PYEOF
)
case "$FACTS_CHECK" in OK*) nl_ok "item-78 numbers for the end card: $FACTS_CHECK";; *) nl_fail "item-78 numbers differ from what this item was staged against: $FACTS_CHECK";; esac
HELPL=$($MAIN --help 2>&1 || true)
for f in --jinja --parallel --ctx-size --slots --alias --n-gpu-layers; do case "$HELPL" in *"$f"*) ;; *) nl_fail "llama-server flag $f not in this build";; esac; done
case "$($MAIN --version 2>&1)" in *5d16e81*) nl_ok "mainline b10371 5d16e81";; *) nl_fail "llama.cpp build changed (expected b10371 5d16e81)";; esac
VLLM_V=$($VENV/bin/python -c "import importlib.metadata as m; print(m.version('vllm'))" 2>/dev/null)
[ "$VLLM_V" = "0.32.0" ] && nl_ok "stock vLLM $VLLM_V in humming-env" || nl_fail "vLLM version '$VLLM_V' != 0.32.0"
# NB: cwd must not hold an inspect.py / typing shadow (a /tmp cwd broke `vllm serve --help` on 2026-11-03): run from /
HELPV=$(cd / && PATH="$VENV/bin:$PATH" $VENV/bin/python -m vllm.entrypoints.cli.main serve --help=all 2>&1 || true)
for want in --linear-backend --language-model-only --max-num-seqs --max-num-batched-tokens --gpu-memory-utilization --served-model-name --max-model-len --trust-remote-code --seed \
            --enable-auto-tool-choice --tool-call-parser --reasoning-parser --enable-prefix-caching; do
  case "$HELPV" in *"$want"*) ;; *) nl_fail "vllm serve flag $want absent";; esac; done
case "$HELPV" in *qwen3_coder*) nl_ok "vLLM tool parser qwen3_coder listed";; *) nl_fail "vLLM has no qwen3_coder tool parser";; esac
RP=$(cd / && $VENV/bin/python -c "from vllm.reasoning import ReasoningParserManager as M; print(','.join(sorted(M.list_registered())))" 2>/dev/null)
case ",$RP," in *,qwen3,*) nl_ok "vLLM reasoning parser qwen3 registered";; *) nl_fail "vLLM reasoning parser qwen3 not registered (got: ${RP:0:80})";; esac
case "$(grep -c -E '<function=|<parameter=' "$NVDIR/chat_template.jinja" 2>/dev/null)" in 0|"") nl_fail "NVFP4 chat template has no <function=/<parameter= tool format (qwen3_coder parser would not match)";; *) nl_ok "template tool format = qwen3_coder XML";; esac
HV=$(cd "$CHECKOUT" && git rev-parse --short HEAD 2>/dev/null)
case "$HV" in "$HERMES_PIN"*) nl_ok "Hermes checkout $HV (pin $HERMES_PIN)";; *) nl_fail "Hermes checkout $HV != pin $HERMES_PIN";; esac
for dd in installs tools; do [ -d "/home/witcheer/.hermes/$dd" ] || nl_fail "main Hermes root has no $dd/ to share with the take homes"; done
nl_say "  regime: $REGIME"
nl_check_disk 20
for p in $LPORT $VPORT $PPORT; do ss -ltnH "sport = :$p" 2>/dev/null | grep -q . && nl_fail "port $p is already in use"; done
for t in r78z-nvfp4 r78z-q4km; do tmux has-session -t "=$t" 2>/dev/null && nl_fail "tmux session $t already exists"; done
grep -qE '/home/|/Users/|witcheer' "$PROMPT" && nl_fail "prompt file names a home path or the account"
if nl_skip_exact "$ART"; then nl_say "nothing to run, item complete ($ART exists)"; exit 0; fi
DL=$(nl_deadline_epoch); LEFT=$(( (DL-$(date +%s))/60 ))
for e in nvfp4 q4km; do for n in $(seq 1 "$TAKES"); do nl_say "  RUN take $e-$n -> $RUN/takes/$e-$n (literal artefact $RUN/takes/$e-$n/rec/result.json)"; done; done
nl_say "  RUN clip -> $RUN/out/nvfp4-vs-q4km-final.mp4 ; RUN artefact -> $ART"
nl_say "  planned: $((2*TAKES)) takes (wall cap ${WALLMIN} min each; card typical ~31 min for 2+2 takes (guess), worst ~58; hard stop $(date -d "@$DL" '+%a %H:%M') local, $LEFT min away)"
if [ "$SMOKE" != 1 ] && [ "$STUB" != 1 ]; then
  [ "$LEFT" -ge 75 ] || { nl_say "  under 75 min left in the window: not starting (drain + 2 starts + $((2*TAKES)) takes would be cut short; exit 3, resumable)"; exit 3; }
fi
if [ "${DRYRUN:-0}" = "1" ] || [ ! -f "$MANIFEST" ]; then nl_manifest_init serve_nvfp4 takes_nvfp4 serve_q4km takes_q4km gate render; fi
nl_gate

# ---- phase 2 ----
SP=""; WD=""; DONALD_STOPPED=0; DONALD_BACK=0
mem_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
wait_mem_free() { local i v; for i in $(seq 1 90); do v=$(mem_used); [ "${v:-99999}" -lt 1024 ] && { nl_say "  card clear (${v} MiB)"; return 0; }; sleep 2; done; nl_say "  card NOT clear after 180 s (${v} MiB)"; return 1; }
stop_srv() { # own server only: the pid (vLLM: its whole process group, started with setsid) this item started
  [ -n "$WD" ] && { kill "$WD" 2>/dev/null; WD=""; }
  [ -n "$SP" ] || return 0
  kill -TERM -- "-$SP" 2>/dev/null || kill -TERM "$SP" 2>/dev/null
  for i in $(seq 1 30); do kill -0 "$SP" 2>/dev/null || break; sleep 2; done
  kill -KILL -- "-$SP" 2>/dev/null || kill -KILL "$SP" 2>/dev/null
  sleep 3; SP=""; }
donald_stop() { nl_say "draining Donald (sudo -n systemctl stop llama-server.service)"; sudo -n systemctl stop llama-server.service; DONALD_STOPPED=1; sleep 5; }
donald_restore() {
  [ "$DONALD_STOPPED" = 1 ] && [ "$DONALD_BACK" = 0 ] || return 0
  DONALD_BACK=1; nl_say "restoring Donald (sudo -n systemctl start llama-server.service)"
  sudo -n systemctl start llama-server.service
  for i in $(seq 1 60); do curl -s -m 5 localhost:8090/health | grep -qi ok && { nl_say "Donald healthy again after ~$((i*5)) s"; return 0; }; sleep 5; done
  nl_say "WARNING: Donald /health did not answer within 5 min after the restore"
}
cleanup() { for t in r78z-nvfp4 r78z-q4km; do tmux kill-session -t "=$t" 2>/dev/null; done; stop_srv; [ "$STUB" = 1 ] || wait_mem_free || true; donald_restore; }
trap cleanup EXIT
mkdir -p "$V/runs"
[ -d "$RUN" ] && mv "$RUN" "$RUN.stale-$(date +%s)"
mkdir -p "$RUN/out" "$RUN/takes"
t0=$(date +%s)
fail() { nl_mark "$1" failed "$2"; nl_say "FAIL: $2"; exit 1; }

serve_vllm() { # sets SP; one NVFP4 server, ctx 65536, parsers on
  local d=$RUN t1=$(date +%s) i m
  local PFX=(); [ "${R78Z_VLLM_NOPREFIX:-0}" = 1 ] && PFX=(--no-enable-prefix-caching)
  ( export FLASHINFER_CUDA_ARCH_LIST=12.0f CUDA_HOME=/usr/local/cuda-13 MAX_JOBS=4 NVCC_THREADS=1 HUMMING_MAX_PARALLEL_BUILD_WORKERS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
    export PATH="$VENV/bin:/usr/local/cuda-13/bin:$PATH"
    cd /
    exec setsid nice -n 19 "$VENV/bin/python" -m vllm.entrypoints.cli.main serve "$NVDIR" --served-model-name $NV_NAME \
      --max-model-len $CTXCFG --max-num-seqs 16 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.92 \
      --seed 721 --host 127.0.0.1 --port $VPORT --trust-remote-code --language-model-only --linear-backend cutlass \
      --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 "${PFX[@]}" >> "$d/server-nvfp4.log" 2>&1 ) &
  SP=$!
  ( while kill -0 "$SP" 2>/dev/null; do a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo); if [ "$a" -lt 6144 ]; then echo "$(date +%T) RAM FLOOR avail=${a} MiB, killing own vLLM group" >> "$d/watch.log"; kill -KILL -- "-$SP" 2>/dev/null; fi; sleep 5; done ) &
  WD=$!
  for i in $(seq 1 180); do
    kill -0 "$SP" 2>/dev/null || { nl_say "  vLLM died during start after $(( $(date +%s)-t1 )) s: $(grep -a -E 'Error|error' "$d/server-nvfp4.log" | tail -3 | cut -c1-200 | tr '\n' '|')"; kill -KILL -- "-$SP" 2>/dev/null; SP=""; return 1; }
    m=$(curl -s -m 3 http://127.0.0.1:$VPORT/v1/models 2>/dev/null || true)
    [[ "$m" == *"\"$NV_NAME\""* ]] && [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:$VPORT/health)" = 200 ] && { VSTART=$(( $(date +%s)-t1 )); nl_say "  vLLM up after ${VSTART} s (VRAM $(mem_used) MiB)"; return 0; }
    [ "$(date +%s)" -ge "$DEADLINE" ] && { nl_say "  window ended during vLLM start"; return 1; }
    sleep 5; done
  nl_say "  vLLM start timeout"; return 1; }
serve_llama() { # sets SP
  local d=$RUN t1=$(date +%s) i
  "$MAIN" -m "$Q4" --port $LPORT --n-gpu-layers 99 --host 127.0.0.1 --jinja --ctx-size $CTXCFG --parallel 1 --alias $Q4_NAME >> "$d/server-q4km.log" 2>&1 &
  SP=$!
  for i in $(seq 1 150); do
    curl -s -m 3 http://127.0.0.1:$LPORT/health | grep -qi ok && { LSTART=$(( $(date +%s)-t1 )); nl_say "  llama-server up after ${LSTART} s (VRAM $(mem_used) MiB)"; return 0; }
    kill -0 "$SP" 2>/dev/null || break; sleep 2; done
  return 1; }
chat_smoke() { # <port> <model>: a real tool-call request with thinking on: the parser must return tool_calls (and, on vLLM, a reasoning field), else Hermes would not work on it
  python3 - "$1" "$2" <<'PYEOF'
import json, sys, time, urllib.request
port, model = sys.argv[1], sys.argv[2]
body = {"model": model, "messages": [{"role": "user", "content": "What time is it in Paris? Use the tool."}], "max_tokens": 3000, "temperature": 1.0, "stream": False,
        "tools": [{"type": "function", "function": {"name": "get_time", "description": "current time in a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]}
t = time.time()
try:
    r = json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions" % port, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=240).read())
except Exception as e:
    print("SMOKE FAIL request: %s" % str(e)[:160]); sys.exit(1)
m = r["choices"][0]["message"]
tc = m.get("tool_calls") or []
rs = m.get("reasoning") or m.get("reasoning_content") or ""
ok = bool(tc) and tc[0]["function"]["name"] == "get_time" and "city" in json.loads(tc[0]["function"]["arguments"] or "{}")
n = (r.get("usage") or {}).get("completion_tokens")
print(("SMOKE OK" if ok else "SMOKE FAIL") + " tool_calls=%d reasoning_chars=%d completion_tokens=%s %.1fs %s" % (len(tc), len(rs), n, time.time() - t, json.dumps(tc[0]["function"]) if tc else json.dumps(m)[:200]))
sys.exit(0 if ok else 1)
PYEOF
}
DEADLINE=$(nl_deadline_epoch)
VSTART=0; LSTART=0
run_takes() { # <engine> <upstream> <alias>
  local eng=$1 up=$2 al=$3 n td left w rc
  for n in $(seq 1 "$TAKES"); do
    td=$RUN/takes/$eng-$n
    left=$(( DEADLINE - $(date +%s) ))
    [ "$STUB" = 1 ] && left=99999
    if [ "$left" -lt 540 ]; then nl_say "  take $eng-$n: only $((left/60)) min left (< 9), skipped"; continue; fi
    w=$(( WALLMIN * 60 )); [ "$((left - 360))" -lt "$w" ] && w=$(( left - 360 ))
    nl_say "=== take $eng-$n (wall cap $w s, $((left/60)) min left in the window) ==="
    UPSTREAM=$up ALIAS=$al TAG=$eng PROMPT=$PROMPT WALL_S=$w CTXCFG=$CTXCFG FR_PORT=$PPORT FR_PY=$RPY FR_CHECKOUT=$CHECKOUT FR_QUIET=$QUIET THINK=$THINK bash "$RA/r78z_take.sh" "$td" >> "$RUN/take.log" 2>&1; rc=$?
    nl_say "  take $eng-$n: rc=$rc $(tail -1 "$RUN/take.log" | cut -c1-200)"
    nl_say "  take $eng-$n: $(cat "$td/rec/rc" 2>/dev/null) $(cat "$td/rec/result.json" 2>/dev/null)"
    [ "$rc" = 0 ] || nl_say "  take $eng-$n: harness rc $rc (see $RUN/take.log)"
  done
}
if [ "$STUB" = 1 ]; then
  nl_say "STUB mode: no Donald drain, no server start (vllm=${SVL:-none} llama=${SLL:-none})"
else
  # rule 3, live: nothing but Donald may hold the card when we drain him
  FOREIGN=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ' | tr '\n' ' ')
  nl_say "  card owners before the drain: ${FOREIGN:-none}; tmux: $(tmux ls -F '#{session_name}' 2>/dev/null | tr '\n' ' ')"
  donald_stop
  for i in $(seq 1 60); do v=$(mem_used); [ "${v:-99999}" -lt 1024 ] && { nl_say "drain clear (${v} MiB)"; break; }; sleep 2; done
  [ "${v:-99999}" -lt 1024 ] || fail serve_nvfp4 "card memory still ${v:-?} MiB after the drain: another process owns the card, not loading"
fi

# ---- A: NVFP4 on vLLM ----
nl_mark serve_nvfp4 running
if [ -n "$SVL" ]; then UPV=$SVL
else
  serve_vllm || { stop_srv; fail serve_nvfp4 "vLLM did not come up (see $RUN/server-nvfp4.log)"; }
  UPV=http://127.0.0.1:$VPORT
  grep -a -i -E "Using .*(kernel|backend)|NvFp4|fp8 data type|GPU KV cache size|prefix caching|Maximum concurrency|tool.call.parser|reasoning" "$RUN/server-nvfp4.log" | sed 's/^.*\] //' | sort -u | head -14 > "$RUN/kernels-nvfp4.txt"
  echo "{\"engine\":\"vllm\",\"up\":true,\"start_s\":$VSTART,\"used_mib\":$(mem_used),\"ctx\":$CTXCFG}" >> "$RUN/fit.jsonl"
fi
SM=$(chat_smoke "${UPV##*:}" $NV_NAME); nl_say "  vLLM tool-call smoke: $SM"
case "$SM" in "SMOKE OK"*) ;; *) stop_srv; fail serve_nvfp4 "vLLM tool-call smoke failed: Hermes cannot drive this server ($SM)";; esac
nl_mark serve_nvfp4 done "up ${VSTART}s"
nl_mark takes_nvfp4 running
run_takes nvfp4 "$UPV" $NV_NAME
[ -n "$SVL" ] || { curl -s -m 5 "$UPV/metrics" > "$RUN/server-metrics-nvfp4.txt" 2>/dev/null; stop_srv; [ "$STUB" = 1 ] || wait_mem_free || fail takes_nvfp4 "card not clear after the vLLM stop"; }
nl_mark takes_nvfp4 done "$TAKES takes"

# ---- B: UD-Q4_K_M on llama.cpp ----
nl_mark serve_q4km running
if [ -n "$SLL" ]; then UPL=$SLL
else
  serve_llama || { stop_srv; fail serve_q4km "llama-server did not come up (see $RUN/server-q4km.log)"; }
  UPL=http://127.0.0.1:$LPORT
  echo "{\"engine\":\"llama\",\"up\":true,\"start_s\":$LSTART,\"used_mib\":$(mem_used),\"ctx\":$CTXCFG}" >> "$RUN/fit.jsonl"
fi
SM=$(chat_smoke "${UPL##*:}" $Q4_NAME); nl_say "  llama.cpp tool-call smoke: $SM"
case "$SM" in "SMOKE OK"*) ;; *) stop_srv; fail serve_q4km "llama.cpp tool-call smoke failed ($SM)";; esac
nl_mark serve_q4km done "up ${LSTART}s"
nl_mark takes_q4km running
run_takes q4km "$UPL" $Q4_NAME
[ -n "$SLL" ] || { curl -s -m 5 "$UPL/slots" > "$RUN/slots-end-q4km.json" 2>/dev/null; stop_srv; [ "$STUB" = 1 ] || wait_mem_free || fail takes_q4km "card not clear after the llama stop"; }
nl_mark takes_q4km done "$TAKES takes"
# the card is free from here: Donald back, then the CPU-only legs
if [ "$STUB" != 1 ]; then donald_restore; fi
nl_say "card released after $(( ($(date +%s)-t0)/60 )) min (drain to restore), the rest is CPU"

# ---- which take goes in the clip (rule fixed in the header: take 1 of each unless a recorder fault) ----
$RPY - "$RUN" "$TAKES" <<'PYEOF' > "$RUN/pair.json"
import json, os, sys
run, takes = sys.argv[1], int(sys.argv[2])
def ok(d):
    try:
        r = json.load(open(d + "/rec/result.json"))
        return r.get("end") in ("done", "wall") and os.path.getsize(d + "/rec/events-a.jsonl") > 1000 and os.path.getsize(d + "/hud/hud.jsonl") > 100
    except Exception:
        return False
pick, why = {}, {}
for e in ("nvfp4", "q4km"):
    for n in range(1, takes + 1):
        d = "%s/takes/%s-%d" % (run, e, n)
        if ok(d):
            pick[e] = d
            why[e] = "take %d%s" % (n, "" if n == 1 else " (take 1 had a recorder fault)")
            break
    else:
        pick[e], why[e] = None, "no usable take"
print(json.dumps({"left": pick["nvfp4"], "right": pick["q4km"], "why": why}))
PYEOF
LT=$($RPY -c "import json;print(json.load(open('$RUN/pair.json'))['left'] or '')"); RT=$($RPY -c "import json;print(json.load(open('$RUN/pair.json'))['right'] or '')")
nl_say "  clip pair: $(cat "$RUN/pair.json")"
[ -n "$LT" ] && [ -n "$RT" ] || fail gate "no usable take for both files (see $RUN/take.log)"

# ---- gates (CPU) ----
nl_mark gate running
for td in "$RUN"/takes/*; do
  mkdir -p "$td/gate"
  if [ -f "$td/work/index.html" ]; then
    setsid timeout 240 "$GPY" "$RA/r78z_gate.py" "$td/work/index.html" "$td/gate" --video-seconds 12 < /dev/null >> "$RUN/gate.log" 2>&1
  else echo '{"pass": false, "no_html": true}' > "$td/gate/gate.json"; nl_say "  $(basename "$td"): no index.html in its work dir"; fi
  nl_say "  gate $(basename "$td"): $($GPY -c "import json;d=json.load(open('$td/gate/gate.json'));print('pass' if d.get('pass') else 'FAIL', d.get('bytes'), 'bytes', 'diff', d.get('frame_diff'), [k for k,v in (d.get('checks') or {}).items() if not v])")"
done
nl_mark gate done

# ---- clip (CPU) ----
nl_mark render running
"$RPY" "$RA/r78z_render.py" race "$RUN/out/race.mp4" --left "$LT" --right "$RT" --ffmpeg "$FF" --stills "$RUN/out/stills" >> "$RUN/render.log" 2>&1; rr=$?
if [ "$rr" = 0 ]; then
  "$RPY" "$RA/r78z_facts.py" "$RES" "$RUN/out/race.mp4.summary.json" "$RUN/facts.json" >> "$RUN/render.log" 2>&1 && \
  "$RPY" "$RA/r78z_render.py" card "$RUN/out/card.mp4" --facts "$RUN/facts.json" --ffmpeg "$FF" >> "$RUN/render.log" 2>&1 && \
  "$RPY" "$RA/r78z_results.py" "$RUN/out/results.mp4" --left "$LT" --right "$RT" --ffmpeg "$FF" --summary "$RUN/out/race.mp4.summary.json" --race "$RUN/out/race.mp4" --concat "$RUN/out/card.mp4" \
      --final "$RUN/out/nvfp4-vs-q4km-final.mp4" >> "$RUN/render.log" 2>&1
  rr=$?
fi
"$RPY" - "$RUN/out" "$RUN/out/contact.png" <<'PYEOF' 2>>"$RUN/render.log"
import glob, os, sys
from PIL import Image
out = sys.argv[1]
fs = sorted(glob.glob(out + "/stills/*.png"))[:7]
fs += [p for p in (out + "/results.mp4.png", out + "/card.mp4.png") if os.path.exists(p)]
if fs:
    w, h = 640, 360
    sheet = Image.new("RGB", (w * 3, h * 3), "#000")
    for i, f in enumerate(fs[:9]):
        sheet.paste(Image.open(f).convert("RGB").resize((w, h)), ((i % 3) * w, (i // 3) * h))
    sheet.save(sys.argv[2])
PYEOF
nl_say "  render rc=$rr; final clip: $([ -f "$RUN/out/nvfp4-vs-q4km-final.mp4" ] && du -h "$RUN/out/nvfp4-vs-q4km-final.mp4" | cut -f1 || echo missing)"
python3 "$RA/r78z_summary.py" "$RUN" "$SLUG" "$REGIME" > "$RUN/summary.json" 2>>"$RUN/render.log"
nl_judge_artifact render "$rr" "$ART" "$t0" "$(( ($(date +%s)-t0)/60 )) min, render rc $rr" || { nl_say "FAIL: no fresh summary.json or render rc $rr"; exit 1; }
nl_say "  summary: $(python3 -c "import json;d=json.load(open('$ART'));print(json.dumps(d.get('headline')))")"
nl_say "DONE $SLUG"
exit 0
