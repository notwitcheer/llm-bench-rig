#!/bin/bash
# 9a1-ttf-ternary-27b.sh: idle-queue item (staged 2026-11-12, supersedes item-9a1.sh). ttf TQ1_P ternary Qwen3.8-27B (no ternary adapters, 6.74 GB)
#   vs stock Qwen3.8-27B Q6_K (22.88 GB), the same Hermes Agent task, one RTX 5090.
#   TERNARY = ttf-home 92d53f02d, engine llama.cpp 4afa91d03 (llama-server build 12193, CUDA sm_120, built from ttf-home --q38), file
#             ~/models/ttf-27b/ttf-Qwen3.8-27B-noadapters-TQ1_P.gguf (6,741,621,728 B, LFS de56c766, verified 2026-11-12)
#   STOCK   = llama.cpp cf34d3113 (~/llama.cpp-moecache, llama-server 0.9.0-dev build 6), file ~/models/qwen38-27b/Qwen3.8-27B-Q6_K.gguf (22,884,408,288 B)
# WHAT: both arms: -ngl 99, flash-attn on, --jinja, KV q8_0, ctx 65,536, one slot, 6 threads, --reasoning-budget 8192 with the series message,
#   no speculative decoding, no adapter. Hermes Agent pin 9bda188, own HERMES_HOME per take, toolsets terminal+file, approvals off, memory off,
#   max 60 turns, wall cap 10 min per take. Task: prompts/fix-shopcart.txt on the seed (4 planted bugs, 5 pristine tests).
#   Per arm, in order: tool-call smoke, probe (fixed prompt, 256 tokens, second of two runs, never gating), takes, GSM8K first 20 (greedy, thinking off).
#   Headline = 3 takes per arm (stock block then ttf block, one server load each). Extension legs = up to 8 takes per arm (resumable, exit 3 when the window cuts).
# MOJO: a 27b ternary gguf at 6.7 GB vs the 23 GB stock file, two Hermes Agent panes on one 5090, result card first with pass counts and file sizes | does the 6.7 GB file hold? | a 27b ternary gguf at 6.7 GB held the same hermes task: [ttf] of [N] passes against [stock] of [N] for the 23 GB file, [X] GB less on disk
# EV: private (projected post angle and reach bracket live in the ops vault status note)
#
# PRE-REGISTERED (9a1-ttf-ternary/PREREG.md, sha256 in FROZEN.sha256, frozen 2026-11-12 BEFORE any counted output; this item refuses to run when
#   FROZEN.sha256 differs from the pin below; mc_summary9a1.py --selftest proves the verdict branches).
#   verdict: BROKEN (ttf) | BROKEN (baseline) | INCOMPLETE (< 2 CLEAN takes in an arm: exit 3) | HOLDS (ttf passes >= stock passes - 1) | FALLS BEHIND (ttf passes <= stock passes - 2)
#   GSM8K (n=20) and decode tok/s are reported, never gating.
# CARD TIME (UNVERIFIED: no counted take has run; worst case is arithmetic from the wall cap, typical is a guess):
#   worst headline = 6 takes x 10 min cap + 2 loads + probes + gsm8k (40 requests at 512 max_tokens) ~ 75-85 min.
#   worst extension = 10 more takes x 10 min cap + 2 loads ~ 110 min, so the 8-per-arm total worst is ~190 min across blocks.
#   typical is UNKNOWN until a take runs.
# DISK: 0 GB new download here (the 6.74 GB file and the 22.88 GB Q6_K are already on disk, size-checked). floor 122 GB (brief). nothing deleted.
# WINDOW: real runs refuse (exit 3, nothing drained) before NOT_BEFORE (Fri 2026-11-13 14:10 CET, epoch 1794575400 (moved from Sat 00:10 on 2026-11-12); the weekend block is empty
#   and the 9a1 extension is what fills it). needs >= 60 min left in the block or exits 3 without draining Donald; a take is skipped when < 11 min remain.
# GPU lock: flock -w 1800 /home/witcheer/.gpu-smoke.lock for the card section.
#
# test knobs (empty in production): S9_SMOKE=1 (slug -smoke, 1 take per arm, 3 min wall, never counted), S9_STUB_STOCK / S9_STUB_TTF (CPU mock per arm, no Donald drain,
#   no server start, DRYRUN-safe), S9_TAKES, S9_EXT_TAKES, S9_WALL_MIN, S9_THINK=on|off, S9_QUIET, S9_NO_NOTBEFORE=1 (test only), MANIFEST_OVERRIDE, S9_NO_SELFTEST=1.
set -uo pipefail
NL_TAG=9a1
MANIFEST=/home/witcheer/9a1-ttf-ternary-27b-status.json
[ -n "${MANIFEST_OVERRIDE:-}" ] && MANIFEST=$MANIFEST_OVERRIDE
export MANIFEST
. /home/witcheer/night-lib.sh
H=/home/witcheer/voxel-stage/9a1-ttf-ternary
RUNROOT=/home/witcheer/9a1-runs
GPY=/home/witcheer/benchmark-rig/.venv/bin/python
RPY=/home/witcheer/scratch/race/venv/bin/python
FF=/home/witcheer/voxelbench/ffmpeg
CHECKOUT=/home/witcheer/.hermes/hermes-agent
HBIN=$CHECKOUT/.hermes/bin/hermes
HERMES_PIN=9bda188
TTF_SRV=/home/witcheer/ttf-home/engine-q38/build-cuda/bin/llama-server
STOCK_SRV=/home/witcheer/llama.cpp-moecache/build/bin/llama-server
TTF_M=/home/witcheer/models/ttf-27b/ttf-Qwen3.8-27B-noadapters-TQ1_P.gguf
STOCK_M=/home/witcheer/models/qwen38-27b/Qwen3.8-27B-Q6_K.gguf
TTF_BYTES=6741621728
STOCK_BYTES=22884408288
TTF_SHA=4afa91d03; TTF_TREE=92d53f02d; STOCK_SHA=cf34d3113
NOTBEFORE=${NOT_BEFORE:-1794575400}
LPORT=8294; PPORT=8295
CTXCFG=65536; THREADS=6; KVQ=q8_0
RB=8192; RB_MSG="Thinking budget reached. Act now with your commands."
PROMPT=$H/prompts/fix-shopcart.txt
SEEDDIR=$H/seed
H_FROZEN=dabba235b59852f343daf3aeff5076608a4d605f41df8eb45edb20252cdef3c9
SMOKE=${S9_SMOKE:-0}
NT=${S9_TAKES:-3}; EXT=${S9_EXT_TAKES:-8}; WALLMIN=${S9_WALL_MIN:-10}; THINK=${S9_THINK:-on}; QUIET=${S9_QUIET:-20}
[ "$SMOKE" = 1 ] && { NT=${S9_TAKES:-1}; WALLMIN=${S9_WALL_MIN:-3}; EXT=$NT; }
SLUG=9a1-ttf-ternary-27b; [ "$SMOKE" = 1 ] && SLUG=$SLUG-smoke
STUBS=${S9_STUB_STOCK:-}; STUBT=${S9_STUB_TTF:-}
RUN=$RUNROOT/$SLUG
ART=$RUN/summary.json
REGIME="Qwen3.8-27B: ttf TQ1_P ternary (noadapters, 6,741,621,728 B, engine ttf-home 92d53f02d / llama.cpp 4afa91d03) vs stock Q6_K (22,884,408,288 B, llama.cpp cf34d3113); -ngl 99, flash-attn on, --jinja, KV $KVQ, ctx $CTXCFG, one slot, $THREADS threads, thinking $THINK, --reasoning-budget $RB; Hermes 9bda188 terminal+file, approvals off, memory off; fix-shopcart (4 planted bugs, 5 pristine tests); $NT takes per arm headline, up to $EXT per arm"

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0} smoke=$SMOKE takes=$NT+$NT ext=$EXT wall=${WALLMIN}min stub=${STUBS:-none})"
nl_check_path "$TTF_SRV" "$STOCK_SRV" "$HBIN" "$GPY" "$RPY" "$FF" "$PROMPT" "$SEEDDIR/shopcart/cart.py" "$SEEDDIR/tests/test_cart.py" \
  "$H/mc_take.sh" "$H/rp_proxy.py" "$H/rp_count.py" "$H/rp_recount.py" "$H/rp_record.py" "$H/mc_mkhome.py" "$H/mc_render.py" \
  "$H/mc_summary9a1.py" "$H/facts9a1.py" "$H/arm9a1.py" "$H/rp_gate.py" "$H/PREREG.md" "$H/FROZEN.sha256" "$H/SEED.sha256" \
  /home/witcheer/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome
if [ -z "${S9_STUB_STOCK:-}" ]; then [ "$(stat -c %s "$STOCK_M" 2>/dev/null)" = "$STOCK_BYTES" ] && nl_ok "stock Q6_K $STOCK_BYTES B" || nl_fail "stock file size != $STOCK_BYTES"; fi
if [ -z "${S9_STUB_STOCK:-}" ]; then [ "$(stat -c %s "$TTF_M" 2>/dev/null)" = "$TTF_BYTES" ] && nl_ok "ttf file $TTF_BYTES B" || nl_fail "ttf file size != $TTF_BYTES"; fi
# the pre-registration is frozen: FROZEN.sha256 is pinned in this script; every frozen file is listed in it, and PREREG.md lists every frozen hash
FZ=$(sha256sum "$H/FROZEN.sha256" | cut -c1-64)
[ "$FZ" = "$H_FROZEN" ] && nl_ok "FROZEN.sha256 ${FZ:0:12}" || nl_fail "FROZEN.sha256 differs from the pin (${FZ:0:12})"
CK=$(cd "$H" && sha256sum -c FROZEN.sha256 2>&1) && nl_ok "frozen files match ($(echo "$CK" | grep -c ': OK') OK)" || nl_fail "frozen file differs: $(echo "$CK" | grep -v ': OK' | head -3 | tr '\n' ' ')"
SEEDH=$(cd "$SEEDDIR" && find . -type f ! -name '*.pyc' ! -path '*/__pycache__/*' | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-64)
H_SEED=$(cut -d' ' -f1 "$H/SEED.sha256" 2>/dev/null)
[ -n "$H_SEED" ] && [ "$SEEDH" = "$H_SEED" ] && nl_ok "frozen seed repo ${SEEDH:0:12}" || nl_fail "seed repo differs from the pre-registration"
PRG=$(cat "$H/PREREG.md")
for h in $(grep -v 'PREREG.md' "$H/FROZEN.sha256" | cut -d' ' -f1) "$H_SEED"; do [[ $PRG == *"$h"* ]] || nl_fail "PREREG.md does not list frozen hash ${h:0:12}"; done
"$GPY" -c "import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$H/mc_summary9a1.py" "$H/facts9a1.py" "$H/arm9a1.py" "$H/rp_gate.py" "$H/rp_proxy.py" "$H/rp_count.py" "$H/rp_recount.py" "$H/rp_record.py" "$H/mc_mkhome.py" || nl_fail "harness syntax"
"$GPY" -c "import pyarrow" 2>/dev/null && nl_ok "pyarrow in the benchmark-rig venv (gsm8k parquet read)" || nl_fail "pyarrow missing in $GPY"
(cd / && "$RPY" -c "import ast,sys,pyte,PIL,fontTools; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$H/mc_render.py") || nl_fail "renderer deps/syntax (pyte, PIL, fontTools in $RPY)"
bash -n "$H/mc_take.sh" || nl_fail "mc_take.sh syntax"
if [ "${S9_NO_SELFTEST:-0}" != 1 ]; then
  ST=$("$GPY" "$H/mc_summary9a1.py" --selftest 2>&1); case "$ST" in *"SELFTEST PASS"*) nl_ok "verdict self-test: PASS";; *) nl_fail "verdict self-test: $(echo "$ST" | tail -2 | tr '\n' ' ')";; esac
fi
# engine and build pins, checked at launch (rule: HEAD, not a ref)
TH=$(git -C /home/witcheer/ttf-home rev-parse --short=9 HEAD 2>/dev/null); [ "$TH" = "$TTF_TREE" ] && nl_ok "ttf-home @ $TTF_TREE" || nl_fail "ttf-home HEAD $TH want $TTF_TREE"
EH=$(git -C /home/witcheer/ttf-home/engine-q38 rev-parse --short=9 HEAD 2>/dev/null); [ "$EH" = "$TTF_SHA" ] && nl_ok "engine @ $TTF_SHA" || nl_fail "engine HEAD $EH want $TTF_SHA"
SH=$(git -C /home/witcheer/llama.cpp-moecache rev-parse --short=9 HEAD 2>/dev/null); [ "$SH" = "$STOCK_SHA" ] && nl_ok "stock llama.cpp @ $STOCK_SHA" || nl_fail "stock HEAD $SH want $STOCK_SHA"
HV=$(git -C "$CHECKOUT" rev-parse --short HEAD 2>/dev/null); case "$HV" in "$HERMES_PIN"*) nl_ok "Hermes checkout $HV (pin $HERMES_PIN)";; *) nl_fail "Hermes checkout $HV != pin $HERMES_PIN";; esac
for dd in installs tools; do [ -d "/home/witcheer/.hermes/$dd" ] || nl_fail "main Hermes root has no $dd/"; done
TVER=$(CUDA_VISIBLE_DEVICES= $TTF_SRV --version 2>&1 || true); [[ $TVER == *"$TTF_SHA"* || $TVER == *"build 12193"* ]] && nl_ok "ttf llama-server build 12193 ($TTF_SHA)" || nl_fail "ttf llama-server version changed"
SVER=$(CUDA_VISIBLE_DEVICES= $STOCK_SRV --version 2>&1 || true); [[ $SVER == *"$STOCK_SHA"* ]] && nl_ok "stock llama-server $STOCK_SHA" || nl_fail "stock llama-server version changed"
HELPL=$(CUDA_VISIBLE_DEVICES= $TTF_SRV --help 2>&1 || true)
for f in --jinja --parallel --ctx-size --alias --n-gpu-layers --flash-attn --cache-type-k --cache-type-v --metrics --threads --reasoning-budget --reasoning-budget-message; do
  [[ $HELPL == *"$f"* ]] || nl_fail "ttf llama-server flag $f not in this build"
done
HELPS=$(CUDA_VISIBLE_DEVICES= $STOCK_SRV --help 2>&1 || true)
for f in --jinja --parallel --ctx-size --alias --n-gpu-layers --flash-attn --cache-type-k --cache-type-v --metrics --threads --reasoning-budget --reasoning-budget-message; do
  [[ $HELPS == *"$f"* ]] || nl_fail "stock llama-server flag $f not in this build"
done
nl_check_disk 122
for p in $LPORT $PPORT; do LS=$(ss -ltnH "sport = :$p" 2>/dev/null || true); [ -z "$LS" ] || nl_fail "port $p is already in use"; done
for t in s9-stock s9-ttf; do tmux has-session -t "=$t" 2>/dev/null && nl_fail "tmux session $t already exists"; done
PT=$(cat "$PROMPT"); [[ $PT == */home/* || $PT == */Users/* || $PT == *witcheer* || $PT == *ternary9a1* || $PT == *9a1-ttf* ]] && nl_fail "prompt file names a home path, the account or the harness dir"
if nl_skip_exact "$ART"; then nl_say "nothing to run, item complete ($ART exists)"; exit 0; fi
DL=$(nl_deadline_epoch); LEFT=$(( (DL-$(date +%s))/60 ))
nl_say "  planned: $((NT*2)) headline takes + extension to $EXT per arm (wall cap ${WALLMIN} min each; hard stop $(date -d "@$DL" '+%a %H:%M') local, $LEFT min away)"
if [ "$SMOKE" != 1 ] && [ -z "$STUBS" ]; then
  [ "$LEFT" -ge 60 ] || { nl_say "  under 60 min left in the window: not starting (exit 3, resumable)"; exit 3; }
fi
if [ "${DRYRUN:-0}" = "1" ] || [ ! -f "$MANIFEST" ]; then nl_manifest_init serve_stock takes_stock serve_ttf takes_ttf gate render; fi
nl_gate

# NOT_BEFORE gate (real runs only, after the DRYRUN exit so the dry-run is green on Thu): exit 3 before any card use
if [ -z "$STUBS" ] && [ "$SMOKE" != 1 ] && [ "${S9_NO_NOTBEFORE:-0}" != 1 ]; then
  now=$(date +%s); [ "$now" -ge "$NOTBEFORE" ] || { nl_say "NOT_BEFORE not reached ($(date -d "@$NOTBEFORE" '+%a %F %H:%M %Z')), exit 3, nothing drained"; exit 3; }
fi

# ---- phase 2 ----
if [ -z "$STUBS" ]; then
  exec 9>/home/witcheer/.gpu-smoke.lock
  flock -w 1800 9 || { nl_say "gpu lock still held after 30 min: not starting, exit 3"; exit 3; }
  nl_say "gpu lock taken"
fi
SP=""; DONALD_STOPPED=0; DONALD_BACK=0
mem_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
stop_srv() { [ -n "$SP" ] || return 0; kill -TERM "$SP" 2>/dev/null; for i in $(seq 1 30); do kill -0 "$SP" 2>/dev/null || break; sleep 2; done; kill -KILL "$SP" 2>/dev/null; sleep 3; SP=""; }
kill_own_srv() { local pid; for pid in $(pgrep -x llama-server 2>/dev/null); do case "$(readlink -f /proc/$pid/exe 2>/dev/null)" in "$(readlink -f "$TTF_SRV")"|"$(readlink -f "$STOCK_SRV")") kill -9 "$pid" 2>/dev/null;; esac; done; return 0; }
wait_mem_free() { local i v; for i in $(seq 1 90); do v=$(mem_used); [ "${v:-99999}" -lt 1024 ] && { nl_say "  card clear (${v} MiB)"; return 0; }; sleep 2; done; nl_say "  card NOT clear after 180 s (${v} MiB)"; return 1; }
donald_stop() { nl_say "draining Donald (night-lib nl_donald_stop)"; nl_donald_stop; DONALD_STOPPED=1; }
donald_restore() {
  [ "$DONALD_STOPPED" = 1 ] && [ "$DONALD_BACK" = 0 ] || return 0
  DONALD_BACK=1; nl_say "restoring Donald (night-lib nl_donald_restore)"; nl_donald_restore
  local i h; for i in $(seq 1 60); do h=$(curl -s -m 5 localhost:8090/health || true); [[ ${h,,} == *ok* ]] && { nl_say "Donald healthy again after ~$((i*5)) s"; return 0; }; sleep 5; done
  nl_say "WARNING: Donald /health did not answer within 5 min after the restore"
}
cleanup() { for t in s9-stock s9-ttf; do tmux kill-session -t "=$t" 2>/dev/null; done; stop_srv
  if [ -z "$STUBS" ] && [ "$DONALD_STOPPED" = 1 ]; then kill_own_srv; sleep 2; wait_mem_free || true; fi
  donald_restore; }
trap cleanup EXIT
mkdir -p "$RUN/out" "$RUN/takes"
t0=$(date +%s)
fail() { nl_mark "$1" failed "$2"; nl_say "FAIL: $2"; exit 1; }
DEADLINE=$(nl_deadline_epoch)
SHORT=0
BASEARGS=(-ngl 99 -t $THREADS -fa on --jinja -c $CTXCFG --parallel 1 -ctk $KVQ -ctv $KVQ)
take_done() { [ -f "$RUN/takes/$1/rec/result.json" ] && grep -qE '"end": *"(done|wall)"' "$RUN/takes/$1/rec/result.json" 2>/dev/null; }
leg_pending() { local tag=$1 lim=$2 i; for i in $(seq 1 "$lim"); do take_done "$tag-$i" || return 0; done; return 1; }
wait_up() { local i h t1=$(date +%s); for i in $(seq 1 "$2"); do h=$(curl -s -m 3 "http://127.0.0.1:$1/health" || true); [[ ${h,,} == *ok* ]] && { LSTART=$(( $(date +%s)-t1 )); return 0; }; kill -0 "$SP" 2>/dev/null || return 1; sleep 2; done; return 1; }
serve_arm() { # <arm> <srv> <model>: sets SP
  local arm=$1 srv=$2 model=$3
  "$srv" -m "$model" "${BASEARGS[@]}" --host 127.0.0.1 --port $LPORT --alias qwen38 --metrics --reasoning-budget $RB --reasoning-budget-message "$RB_MSG" >> "$RUN/server-$arm.log" 2>&1 &
  SP=$!
  wait_up $LPORT 300 && { nl_say "  llama-server ($arm) up after ${LSTART} s (VRAM $(mem_used) MiB)"; return 0; }
  return 1
}
chat_smoke() {
  python3 - "$1" <<'PYEOF'
import json, sys, time, urllib.request
port = sys.argv[1]
body = {"model": "qwen38", "messages": [{"role": "user", "content": "What time is it in Paris? Use the tool."}], "max_tokens": 3000, "temperature": 0.6, "stream": False,
        "tools": [{"type": "function", "function": {"name": "get_time", "description": "current time in a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]}
t = time.time()
try:
    r = json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions" % port, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=300).read())
except Exception as e:
    print("SMOKE FAIL request: %s" % str(e)[:160]); sys.exit(1)
m = r["choices"][0]["message"]
tc = m.get("tool_calls") or []
ok = bool(tc) and tc[0]["function"]["name"] == "get_time" and "city" in json.loads(tc[0]["function"]["arguments"] or "{}")
tm = r.get("timings") or {}
print(("SMOKE OK" if ok else "SMOKE FAIL") + " tool_calls=%d completion_tokens=%s decode=%.1f %.1fs" % (len(tc), (r.get("usage") or {}).get("completion_tokens"), tm.get("predicted_per_second") or 0, time.time() - t))
sys.exit(0 if ok else 1)
PYEOF
}
probe() {
  python3 - "$2" "$RUN/probe-$1.json" <<'PYEOF'
import json, sys, urllib.request
port, out = sys.argv[1], sys.argv[2]
P = ("The lighthouse keeper had kept a logbook for thirty-one years, and in all that time he had never once written down a lie. So when the ship appeared on the horizon on the first calm morning of October, "
     "with no flag, no lights and no one on deck, he sat at his small wooden table for a long while before he picked up the pen. ")
res = []
for i in range(2):
    try:
        r = json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:%s/completion" % port, json.dumps({"prompt": P, "n_predict": 256, "temperature": 0.0, "ignore_eos": True, "cache_prompt": False}).encode(), {"Content-Type": "application/json"}), timeout=600).read())
        tm = r["timings"]
        res.append({"decode_tok_s": round(tm["predicted_per_second"], 2), "predicted_n": tm["predicted_n"]})
    except Exception as e:
        res.append({"error": str(e)[:120]})
ok = [x for x in res if "decode_tok_s" in x]
json.dump({"runs": res, "decode_tok_s": ok[-1]["decode_tok_s"] if ok else None}, open(out, "w"))
print("probe", json.dumps(res))
PYEOF
}
run_takes() { # <tag> <upstream> <from> <to>
  local tag=$1 up=$2 from=$3 to=$4 n td left w rc
  for n in $(seq "$from" "$to"); do
    td=$RUN/takes/$tag-$n
    if take_done "$tag-$n"; then nl_say "  take $tag-$n: already done (resume), kept"; continue; fi
    left=$(( DEADLINE - $(date +%s) ))
    [ -n "$STUBS" ] && left=99999
    if [ "$left" -lt 660 ]; then nl_say "  take $tag-$n: only $((left/60)) min left (< 11), skipped"; SHORT=1; continue; fi
    w=$(( WALLMIN * 60 )); [ "$((left - 360))" -lt "$w" ] && w=$(( left - 360 ))
    [ -d "$td" ] && mv "$td" "$td.partial-$(date +%s)"
    nl_say "=== take $tag-$n (wall cap $w s, $((left/60)) min left) ==="
    UPSTREAM=$up ALIAS=qwen38 TAG=$tag-$n PROMPT=$PROMPT SEEDDIR=$SEEDDIR WALL_S=$w CTXCFG=$CTXCFG FR_PORT=$PPORT FR_PY=$RPY FR_GPY=$GPY FR_CHECKOUT=$CHECKOUT FR_QUIET=$QUIET THINK=$THINK ARENA=/tmp/arena9a1 bash "$H/mc_take.sh" "$td" >> "$RUN/take.log" 2>&1; rc=$?
    nl_say "  take $tag-$n: rc=$rc $(cat "$td/rec/result.json" 2>/dev/null)"
    [ "$rc" = 0 ] || { nl_say "  take $tag-$n: harness rc $rc (see $RUN/take.log)"; SHORT=1; }
    mkdir -p "$td/gate"
    nl_say "  gate $tag-$n: $("$GPY" "$H/rp_gate.py" "$td/work" "$SEEDDIR/tests" "$td/gate" --pytest-python "$GPY" --seed-pkg "$SEEDDIR/shopcart" 2>&1 | head -1 | cut -c1-220)"
  done
}
leg_arm() { # <arm> <srv> <model> <from> <to>
  local arm=$1 srv=$2 model=$3 from=$4 to=$5 UP GS need=0 left
  GS=$RUN/gsm8k.$arm.json
  leg_pending "$arm" "$to" && need=1
  [ -f "$GS" ] || need=1
  if [ "$need" = 0 ]; then nl_say "leg $arm: takes $from-$to and gsm8k already done"; nl_mark "serve_$arm" done "resume: skipped"; return 0; fi
  left=$(( DEADLINE - $(date +%s) ))
  if [ -z "$STUBS" ] && [ "$left" -lt 660 ]; then nl_say "leg $arm: only $((left/60)) min left, not loading (resumable)"; SHORT=1; return 0; fi
  nl_mark "serve_$arm" running
  if [ -n "$STUBS" ]; then
    UP=$STUBS; [ "$arm" = ttf ] && UP=$STUBT
    echo "{\"loaded\":true,\"stub\":true,\"arm\":\"$arm\",\"bytes\":$(stat -c %s "$model"),\"file\":\"$(basename "$model")\"}" > "$RUN/fit-$arm.json"
    echo '{"runs":[],"decode_tok_s":null,"stub":true}' > "$RUN/probe-$arm.json"
  else
    LSTART=0
    serve_arm "$arm" "$srv" "$model" || { stop_srv; echo "{\"loaded\":false,\"why\":\"did not come up\"}" > "$RUN/fit-$arm.json"; nl_mark "serve_$arm" failed "no load"; nl_say "  llama-server did not come up for $arm: arm BROKEN, going on"; return 0; }
    UP=http://127.0.0.1:$LPORT
  fi
  SM=$(chat_smoke "${UP##*:}"); nl_say "  tool-call smoke ($arm): $SM"
  case "$SM" in
    "SMOKE OK"*) [ -z "$STUBS" ] && echo "{\"loaded\":true,\"start_s\":${LSTART:-0},\"vram_mib\":$(mem_used),\"bytes\":$(stat -c %s "$model"),\"file\":\"$(basename "$model")\",\"ctx\":$CTXCFG,\"kv\":\"$KVQ\"}" > "$RUN/fit-$arm.json";;
    *) echo "{\"loaded\":false,\"why\":\"tool-call smoke failed\"}" > "$RUN/fit-$arm.json"; stop_srv; nl_mark "serve_$arm" failed "smoke"; nl_say "  arm $arm BROKEN: tool-call smoke failed, going on"; return 0;;
  esac
  [ -z "$STUBS" ] && nl_say "  probe ($arm): $(probe "$arm" "${UP##*:}" 2>&1 | tail -1 | cut -c1-200)"
  nl_mark "serve_$arm" done "up"
  nl_mark "takes_$arm" running
  run_takes "$arm" "$UP" "$from" "$to"
  if [ ! -f "$GS" ]; then
    nl_say "  gsm8k ($arm): first 20, greedy, thinking off"
    "$GPY" "$H/arm9a1.py" gsm8k "$UP" qwen38 "$GS" "$arm" >> "$RUN/gsm8k.log" 2>&1 || nl_say "  gsm8k ($arm) rc != 0 (see gsm8k.log)"
  fi
  nl_mark "takes_$arm" done "takes $from-$to"
  [ -z "$STUBS" ] && { stop_srv; wait_mem_free || fail "takes_$arm" "card not clear after the $arm stop"; }
}

if [ -z "$STUBS" ]; then
  FOREIGN=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ' | tr '\n' ' ')
  nl_say "  card owners before the drain: ${FOREIGN:-none}; tmux: $(tmux ls -F '#{session_name}' 2>/dev/null | tr '\n' ' ')"
  donald_stop
  for i in $(seq 1 60); do v=$(mem_used); [ "${v:-99999}" -lt 1024 ] && { nl_say "drain clear (${v} MiB)"; break; }; sleep 2; done
  [ "${v:-99999}" -lt 1024 ] || fail serve_stock "card memory still ${v:-?} MiB after the drain: another process owns the card, not loading"
else nl_say "STUB mode: no Donald drain, no server start (stock $STUBS / ttf $STUBT)"; fi

# headline leg: stock block then ttf block (one server load each); extension legs (resumable) more takes per arm, same blocks
leg_arm stock "$STOCK_SRV" "$STOCK_M" 1 "$NT"
leg_arm ttf "$TTF_SRV" "$TTF_M" 1 "$NT"
if [ "$EXT" -gt "$NT" ]; then
  leg_arm stock "$STOCK_SRV" "$STOCK_M" 1 "$EXT"
  leg_arm ttf "$TTF_SRV" "$TTF_M" 1 "$EXT"
fi
if [ -z "$STUBS" ]; then donald_restore; fi
nl_say "card released after $(( ($(date +%s)-t0)/60 )) min, the rest is CPU"

for td in "$RUN"/takes/*; do
  case "$td" in *.partial-*) continue;; esac
  [ -s "$td/hud/hud.jsonl" ] || continue
  python3 "$H/rp_recount.py" "$td" --json "$RUN/out/recount-$(basename "$td").json" > /dev/null 2>&1
done
"$RPY" - "$RUN" "$NT" > "$RUN/pair.json" <<'PYEOF'
import json, os, sys
run, takes = sys.argv[1], int(sys.argv[2])
def ok(d):
    try:
        r = json.load(open(d + "/rec/result.json"))
        return r.get("end") in ("done", "wall") and os.path.getsize(d + "/rec/events-a.jsonl") > 1000 and os.path.getsize(d + "/hud/hud.jsonl") > 100
    except Exception:
        return False
pick, why = {}, {}
for e in ("stock", "ttf"):
    for n in range(1, takes + 1):
        d = "%s/takes/%s-%d" % (run, e, n)
        if ok(d):
            pick[e] = d; why[e] = "take %d" % n; break
    else:
        pick[e], why[e] = None, "no usable take"
print(json.dumps({"left": pick["stock"], "right": pick["ttf"], "why": why}))
PYEOF
LT=$("$RPY" -c "import json;print(json.load(open('$RUN/pair.json'))['left'] or '')"); RT=$("$RPY" -c "import json;print(json.load(open('$RUN/pair.json'))['right'] or '')")
nl_say "  clip pair: $(cat "$RUN/pair.json")"
nl_mark gate done "gates ran per take"
# GSM8K combine (n, ttf_correct, stock_correct): read by mc_summary9a1.py
"$RPY" - "$RUN" <<'PYEOF' 2>>"$RUN/render.log"
import json, os, sys
r = sys.argv[1]
def ld(a):
    p = "%s/gsm8k.%s.json" % (r, a)
    return json.load(open(p)) if os.path.exists(p) else {"n": 0, "correct": 0}
a, b = ld("ttf"), ld("stock")
json.dump({"n": min(a.get("n", 0), b.get("n", 0)) or max(a.get("n", 0), b.get("n", 0)), "ttf_correct": a.get("correct", 0), "stock_correct": b.get("correct", 0)}, open("%s/gsm8k.json" % r, "w"))
PYEOF
"$GPY" "$H/mc_summary9a1.py" "$RUN" "$SLUG" "$REGIME" > "$RUN/summary.pre.json" 2>>"$RUN/render.log"
VERD=$("$GPY" -c "import json;d=json.load(open('$RUN/summary.pre.json'));print(d['verdict'])" 2>/dev/null)
nl_say "  verdict (pre-registered rule): ${VERD:-unreadable}"
nl_mark render running
rr=1
if [ -n "$LT" ] && [ -n "$RT" ]; then
  "$RPY" "$H/mc_render.py" race "$RUN/out/race.mp4" --left "$LT" --right "$RT" --left-label "stock Q6_K" --right-label "ttf TQ1_P" --title "Qwen3.8-27B, two files, one 5090" --ffmpeg "$FF" --stills "$RUN/out/stills" --cut-after-first 8 --real-max-s 40 --target-main-s 30 --open-s 10 --close-s 12 --pre-s 3 >> "$RUN/render.log" 2>&1; rr=$?
  if [ "$rr" = 0 ]; then
    "$GPY" "$H/mc_summary9a1.py" "$RUN" "$SLUG" "$REGIME" > "$RUN/summary.card.json" 2>>"$RUN/render.log"
    "$RPY" "$H/facts9a1.py" "$RUN/summary.card.json" "$RUN/facts.json" >> "$RUN/render.log" 2>&1 && \
    "$RPY" "$H/mc_render.py" card "$RUN/out/card.mp4" --facts "$RUN/facts.json" --ffmpeg "$FF" --seconds 8 >> "$RUN/render.log" 2>&1 && \
    "$FF" -nostdin -hide_banner -loglevel error -y -i "$RUN/out/card.mp4" -i "$RUN/out/race.mp4" \
      -filter_complex "[0:v]setsar=1,fps=25[a];[1:v]setsar=1,fps=25[b];[a][b]concat=n=2:v=1:a=0[v]" -map "[v]" -c:v libx264 -crf 20 -pix_fmt yuv420p -movflags +faststart "$RUN/out/9a1-final.mp4" </dev/null >> "$RUN/render.log" 2>&1
    rr=$?
  fi
else nl_say "  no usable take on one of the arms: no race clip (the table and verdict still stand)"; fi
"$RPY" - "$RUN/out" "$RUN/out/contact.png" <<'PYEOF' 2>>"$RUN/render.log"
import glob, os, sys
from PIL import Image
out = sys.argv[1]
fs = sorted(glob.glob(out + "/stills/*.png"))[:8]
if fs:
    w, h = 640, 360
    sheet = Image.new("RGB", (w * 3, h * 3), "#000")
    for i, f in enumerate(fs[:9]):
        sheet.paste(Image.open(f).convert("RGB").resize((w, h)), ((i % 3) * w, (i // 3) * h))
    sheet.save(sys.argv[2])
PYEOF
nl_say "  render rc=$rr; final clip: $([ -f "$RUN/out/9a1-final.mp4" ] && du -h "$RUN/out/9a1-final.mp4" | cut -f1 || echo missing)"
if [ "$SHORT" = 1 ] || [ "$VERD" = INCOMPLETE ]; then
  nl_say "  a leg could not finish inside the window (verdict ${VERD:-?}): exit 3, resumable, summary.json NOT written (PROVISIONAL: $RUN/summary.pre.json)"
  nl_mark render failed "incomplete, resumable"
  exit 3
fi
"$GPY" "$H/mc_summary9a1.py" "$RUN" "$SLUG" "$REGIME" > "$ART" 2>>"$RUN/render.log"
nl_judge_artifact render "$rr" "$ART" "$t0" "$(( ($(date +%s)-t0)/60 )) min, render rc $rr" || { nl_say "FAIL: no fresh summary.json or render rc $rr"; exit 1; }
nl_say "  summary: verdict $VERD; line 1: $("$GPY" -c "import json;d=json.load(open('$ART'));print(d.get('line1'))")"
nl_say "DONE $SLUG"
exit 0
