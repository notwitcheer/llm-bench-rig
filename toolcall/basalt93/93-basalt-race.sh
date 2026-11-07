#!/bin/bash
# 93-basalt-race.sh: idle-queue item (STAGED in ~/voxel-stage, 2026-11-06, mercury subagent for the GPU thread, mojo-scan pick B). Basalt v0.3.0 (new Blackwell MoE engine, 2026-11-05) against mainline llama.cpp
#   on the IDENTICAL Qwen3.8-Flash-Next GSQ-RCO Q2_0 weights (ISTA-DASLab GGUF 66.4 GB vs the jesdga .basalt pack 70.4 GB, same quantiser), one RTX 5090 32 GB + 59 GB RAM.
# WHAT: a public claim to replay on our card: a community guide (r/LocalLLaMA 1xomy0i, RTX 5080 16 GB + 64 GB RAM, other quant) says Basalt decodes "about 3.5x llama.cpp without drafting and ~5x with it". Same weights, so the engine is the variable.
#   Rows (one harness, the engine's own timings): decode with NO drafting on both sides FIRST (R0 = basalt --spec 0 / llama.cpp at its best --n-cpu-moe), then each engine's own drafting (R1), prefill at 16K and 48K, the project's own
#   `basalt-bench --community` row, and the same Hermes Agent task (fix-shopcart, item-89 task) typed by each engine in separate recorded takes (no drafting), rendered side by side at real speed.
# MOJO: two engines racing on the identical Qwen3.8-Flash-Next file in the Hermes terminal you watch, a live tok/s counter each, the verdict card first | never-seen / intense (a two-day-old Blackwell-only engine against llama.cpp on the same weights, with our own 59.7 tok/s post as the before) | On the identical Qwen3.8-Flash-Next Q2_0 file, Basalt v0.3.0 decodes [B] tok/s without drafting against llama.cpp's [L] on one RTX 5090 with 59 GB of RAM: [R]x; the same Hermes Agent task took [m:ss] on Basalt and [m:ss] on llama.cpp
# EV: private (projected post angle and reach bracket live in the ops vault status note 99-personal/benchmark-rig/basalt93-status.md)
# serves: content (which-engine verdict on a brand-pull model with a picture) + hardening (second engine on the box, Hermes sessions off the main gateway).
#
# PRE-REGISTERED (h/PREREG.md, sha256-frozen in FROZEN.sha256 BEFORE any counted output; the PLAN phase refuses to run when a hash differs; `b93_summary.py --selftest` proves the verdict branches):
#   R0 = basalt_spec0 / llama_plain decode (token-weighted over 9 clean requests each). BASALT AHEAD R0 >= 1.25 | LEVEL 0.80..1.25 | LLAMA.CPP AHEAD R0 < 0.80 | BROKEN | INCOMPLETE (exit 3). A loss or a tie is a valid post, framed as one rig, one file.
#   Hermes Agent is never the subject of a failure: a failed take is a note about the engine or the model.
#
# DISK PLAN (operator brief 13:26: free >= 140 GB at every moment; no download starts unless free minus its size stays >= 140 GB; otherwise in-run behind the df guard, abort a leg under 125 GB). This item OWNS the two big downloads and moves them IN-RUN, one at a time:
#   leg A: download the .basalt pack (70.4 GB) -> Basalt rows + community row + 2 Hermes takes -> delete the pack;  leg B: download the GGUF pair (66.4 GB) -> ladder + llama rows + 2 Hermes takes -> delete the GGUF. Never both on disk.
#   staging leaves BOTH packs off disk (deleted 13:45 after the staging smokes; ~219-223 GB free then, 94's 26.4 GB models and builds already counted). Low point = free-70.4 = ~148-153 GB (above the 125 abort line, below the 140 comfort floor only if siblings eat the margin); the PLAN phase refuses unless (free - size) >= 125 GB, dl_verify.py re-checks its own floor (exit 2) and aborts the leg.
#   Sibling disk use (WebsiteBench Harbor images, ~3 GB each) is not controlled here: the guard reads df live. Downloads go through dl_verify.py (hf_xet, size-checked against the Hub API byte count). Item 94 (QSA) has its models staged (26.4 GB) and downloads nothing.
#   Files are deleted only after the leg's rows and takes are on disk; on exit 3 (window cut) the current pack stays so the resume does not re-download.
#
# POSTABLE ARTEFACT (planned here): runs/<slug>/out/basalt-final.mp4 (16:9 1920x1080, ~60-75 s: result card 8 s FIRST (title = the verdict in words, one column per engine with its GB, big numbers, grey how-measured line per row, footer with the claim tested and the limits),
#   then two Hermes panes (left llama.cpp, right Basalt, take 1 of each unless a recorder fault, start synced on each take's Enter, server-token tok/s counters from each engine's own stream, clock), cut 6 s after the faster pane is done) + out/contact.png + summary.json.
#   HUD LEAK READ: status-bar row blanked, account name / home path / tailnet address / hostname / e-mail shapes drawn as blocks (counts in race.mp4.summary.json). A timelapse pill ("Nx speed") is never a speed claim.
#
# CARD TIME (honest, ESTIMATE from the 2026-11-06 staging smokes, see the status note): Basalt load 16 s, llama.cpp load 3-9 s; community row ~3 min; decode rows 9 x ~5 s per arm; prefill 16K ~2 s / 48K ~6 s; ladder 4 x ~1.5 min; Hermes takes 2 x 2-10 min per engine
#   (Ripple item 89: 54-187 s; thinking-on Flash-Next at 80 tok/s may run longer, wall cap 10 min) => ~45-70 min of card; downloads add ~40-50 min of block time WITHOUT the card (Donald is restored during them). Worst case ~150 min.
# WINDOW RULE: needs >= 150 min left in the block or exits 3 without draining Donald (resumable). Weekend blocks never yield, so enqueue for a weekend block; a take is skipped when under 11 min remain; the item exits 3 when a leg could not finish.
#
# test knobs (empty in production): B93_SMOKE=1 (slug -smoke, 1 take per engine, wall 4 min, 1 rep, no window rule), B93_ONLY=basalt|llama (one leg), B93_STUB_BASALT / B93_STUB_LLAMA = http://127.0.0.1:PORT (CPU mocks: no Donald drain, no server start, no download, DRYRUN-safe),
#   B93_TAKES / B93_REPS / B93_SLUG (CPU proofs write to their own slug), B93_KEEP=1 (do not delete the downloads), B93_LADDER (default "10 11 12 14 16"), B93_WALL_MIN, B93_QUIET, MANIFEST_OVERRIDE (staging DRYRUN only), B93_NO_SELFTEST=1.
set -uo pipefail
NL_TAG=basalt93
MANIFEST=/home/witcheer/basalt93-status.json
[ -n "${MANIFEST_OVERRIDE:-}" ] && MANIFEST=$MANIFEST_OVERRIDE
export MANIFEST
. /home/witcheer/night-lib.sh
B=/home/witcheer/voxel-stage/basalt93
H=$B/h                                                      # the harness folder (frozen files live here)
SRC=$B/src                                                  # Basalt v0.3.0 checkout + build (bin/basalt, bin/basalt-bench)
VENV=$B/venv                                                # Basalt's own server deps (fresh venv, never shared)
LB=/home/witcheer/voxel-stage/llama-stock-18ab62a/build/bin # mainline llama.cpp 18ab62a (side build, CUDA 12.8 + gcc-14, arch 120)
GPY=/home/witcheer/benchmark-rig/.venv/bin/python           # pytest for the gate + the agent's test runner wrapper
RPY=/home/witcheer/scratch/race/venv/bin/python             # pyte + PIL + fontTools (recorder + renderer)
FF=/home/witcheer/voxelbench/ffmpeg
CHECKOUT=/home/witcheer/.hermes/hermes-agent
HBIN=$CHECKOUT/.hermes/bin/hermes
HERMES_PIN=9bda188
PACK_DIR=/home/witcheer/models/basalt93/pack; PACK=$PACK_DIR/Qwen3.8-FN-GSQ-RCO-Q2_0.basalt
GG_DIR=/home/witcheer/models/basalt93/gguf;   GG1=$GG_DIR/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf; GG2=$GG_DIR/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf
PACK_BYTES=70396904412; GG1_BYTES=37623740192; GG2_BYTES=28800138432
CORPUS=/home/witcheer/voxel-stage/docs-corpus/llms-full-2026-11-06.txt
BASALT_TAG=v0.3.0; BASALT_SHA=1bc0ba79; LL_SHA=18ab62a
BPORT=8274; LPORT=8275; PPORT=8292                          # own ports: never :8090 (Donald), 8261-8273 (staging smokes), 8196-8198, 8263/8291 (item 89)
CTXCFG=65536
PROMPT=$H/prompts/fix-shopcart.txt; SEEDDIR=$H/seed
SMOKE=${B93_SMOKE:-0}; ONLY=${B93_ONLY:-}; STUB_B=${B93_STUB_BASALT:-}; STUB_L=${B93_STUB_LLAMA:-}
LADDER=${B93_LADDER:-"10 11 12 14 16"}
WALLMIN=${B93_WALL_MIN:-10}; QUIET=${B93_QUIET:-20}; NTAKES=${B93_TAKES:-2}; REPS=${B93_REPS:-3}
[ "$SMOKE" = 1 ] && { NTAKES=${B93_TAKES:-1}; REPS=${B93_REPS:-1}; WALLMIN=${B93_WALL_MIN:-4}; }
SLUG=${B93_SLUG:-basalt93-basalt-race}; [ "$SMOKE" = 1 ] && SLUG=$SLUG-smoke
RUN=/home/witcheer/voxelbench/runs/$SLUG; ART=$RUN/summary.json
REGIME="Qwen3.8-Flash-Next GSQ-RCO Q2_0 (ISTA-DASLab GGUF 66.4 GB on llama.cpp $LL_SHA; jesdga .basalt pack 70.4 GB on Basalt $BASALT_TAG), one RTX 5090 32 GB + 59 GB RAM, ctx $CTXCFG, KV q8_0 (llama.cpp) / int8 (Basalt), one slot; decode rows: temp 0.6/0.6/0 top_p 0.95 top_k 20 min_p 0.05 thinking off, 3 prompts x $REPS reps, 700 max tokens; Hermes takes: terminal+file toolsets, approvals off, memory off, thinking on, temp 0.6 seed 721, no drafting; llama.cpp --n-cpu-moe tuned on a ladder, Basalt at engine defaults"
SP=""; DONALD_STOPPED=0; DONALD_BACK=0; SHORT=0
mem_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
want_leg() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0} smoke=$SMOKE only=${ONLY:-both} takes=$NTAKES+$NTAKES reps=$REPS wall=${WALLMIN}min stubB=${STUB_B:-none} stubL=${STUB_L:-none})"
nl_check_path "$GPY" "$RPY" "$HBIN" "$FF" "$PROMPT" "$SEEDDIR/shopcart/cart.py" "$SEEDDIR/tests/test_cart.py" "$CORPUS" "$H/PREREG.md" "$H/FROZEN.sha256" "$H/oai_bench.py" "$H/dl_verify.py" \
  "$H/b93_take.sh" "$H/b93_proxy.py" "$H/b93_count.py" "$H/b93_recount.py" "$H/b93_record.py" "$H/b93_mkhome.py" "$H/b93_render.py" "$H/b93_summary.py" "$H/b93_facts.py" "$H/b93_gate.py" \
  /home/witcheer/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome
if want_leg basalt && [ -z "$STUB_B" ]; then nl_check_path "$SRC/bin/basalt" "$SRC/bin/basalt-bench" "$VENV/bin/python"; fi
if want_leg llama && [ -z "$STUB_L" ]; then nl_check_path "$LB/llama-server" "$LB/llama-bench"; fi
# the pre-registration is frozen: every file it lists must hash to FROZEN.sha256 (paths relative to $H), PREREG.md itself included
if (cd "$H" && sha256sum -c FROZEN.sha256 >/dev/null 2>&1); then nl_ok "frozen files match FROZEN.sha256 ($(wc -l < "$H/FROZEN.sha256") entries)"; else nl_fail "a frozen file differs from FROZEN.sha256: $(cd "$H" && sha256sum -c FROZEN.sha256 2>&1 | grep -v ': OK' | head -3 | tr '\n' ' ')"; fi
grep -q "PREREG.md" "$H/FROZEN.sha256" || nl_fail "FROZEN.sha256 does not list PREREG.md"
grep -q "b93_summary.py" "$H/FROZEN.sha256" || nl_fail "FROZEN.sha256 does not list b93_summary.py"
"$GPY" -c "import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$H/b93_gate.py" "$H/b93_proxy.py" "$H/b93_count.py" "$H/b93_recount.py" "$H/b93_record.py" "$H/b93_mkhome.py" "$H/b93_summary.py" "$H/b93_facts.py" "$H/oai_bench.py" "$H/dl_verify.py" || nl_fail "harness syntax"
(cd / && "$RPY" -c "import ast,sys,pyte,PIL,fontTools; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$H/b93_render.py") || nl_fail "renderer deps/syntax (pyte, PIL, fontTools in $RPY)"
bash -n "$H/b93_take.sh" || nl_fail "b93_take.sh syntax"
if [ "${B93_NO_SELFTEST:-0}" != 1 ]; then
  ST=$("$GPY" "$H/b93_summary.py" --selftest 2>&1); case "$ST" in *BAD*) nl_fail "verdict self-test: $ST";; *) nl_ok "verdict rule self-test: $(echo "$ST" | grep -c OK) branches OK";; esac
  GT=$(mktemp -d /tmp/b93gate.XXXXXX); cp -a "$SEEDDIR/." "$GT/work"; mkdir -p "$GT/o"
  G0=$("$GPY" "$H/b93_gate.py" "$GT/work" "$SEEDDIR/tests" "$GT/o" --pytest-python "$GPY" --seed-pkg "$SEEDDIR/shopcart" 2>&1 | head -1)
  case "$G0" in "GATE FAIL: 1 passed, 4 failed"*) nl_ok "gate: the untouched seed fails ($G0)";; *) nl_fail "gate on the seed: $G0";; esac
  rm -rf "$GT"
fi
# engines: versions and the flags this item passes
if want_leg basalt && [ -z "$STUB_B" ]; then
  BV=$(cd "$SRC" && git describe --tags --always 2>/dev/null); case "$BV" in ${BASALT_TAG}*) nl_ok "Basalt checkout $BV (pin $BASALT_TAG)";; *) nl_fail "Basalt checkout $BV != $BASALT_TAG";; esac
  BH=$("$SRC/bin/basalt-bench" --help 2>&1 || true); for f in --community --submitter --community-file --model; do case "$BH" in *"$f"*) ;; *) nl_fail "basalt-bench flag $f missing";; esac; done
fi
if want_leg llama && [ -z "$STUB_L" ]; then
  export LD_LIBRARY_PATH=$LB
  case "$("$LB/llama-server" --version 2>&1)" in *"$LL_SHA"*) nl_ok "llama.cpp $LL_SHA";; *) nl_fail "llama.cpp build changed (expected $LL_SHA)";; esac
  HELPL=$("$LB/llama-server" --help 2>&1 || true); for f in --jinja --parallel --n-cpu-moe --spec-type --alias --flash-attn --cache-type-k; do case "$HELPL" in *"$f"*) ;; *) nl_fail "llama-server flag $f not in this build";; esac; done
  case "$HELPL" in *ngram-mod*) ;; *) nl_fail "llama-server has no ngram-mod spec type";; esac
fi
HV=$(cd "$CHECKOUT" && git rev-parse --short HEAD 2>/dev/null); case "$HV" in "$HERMES_PIN"*) nl_ok "Hermes checkout $HV (pin $HERMES_PIN)";; *) nl_fail "Hermes checkout $HV != pin $HERMES_PIN";; esac
for dd in installs tools; do [ -d "/home/witcheer/.hermes/$dd" ] || nl_fail "main Hermes root has no $dd/ to share with the take homes"; done
# the disk plan, checked as numbers: one pack at a time, never under 125 GB free, runner floor 100
FREE=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
nl_say "  disk plan: free ${FREE} GB now; leg A pulls 70.4 GB (low point $((FREE-71)) GB), leg B 66.4 GB after the pack is deleted (low point $((FREE-67)) GB); in-run guard 125 GB; staged files on disk: pack $([ -f "$PACK" ] && echo yes || echo no), gguf $([ -f "$GG1" ] && echo yes || echo no)"
if [ -z "$STUB_B$STUB_L" ]; then
  nl_check_disk 100
  # only the legs that will run, and only the files that are not on disk yet, count against the 125 GB guard
  if want_leg basalt && [ "$(stat -c %s "$PACK" 2>/dev/null)" != "$PACK_BYTES" ] && [ "$((FREE-71))" -lt 125 ]; then nl_fail "free ${FREE} GB: the 70.4 GB pack would take the disk under the 125 GB guard"; fi
  if want_leg llama && [ "$(stat -c %s "$GG1" 2>/dev/null)" != "$GG1_BYTES" ] && [ "$((FREE-67))" -lt 125 ]; then nl_fail "free ${FREE} GB: the 66.4 GB GGUF pair would take the disk under the 125 GB guard"; fi
fi
for p in $BPORT $LPORT $PPORT; do ss -ltnH "sport = :$p" 2>/dev/null | grep -q . && nl_fail "port $p is already in use"; done
for t in b93-llama b93-basalt; do tmux has-session -t "=$t" 2>/dev/null && nl_fail "tmux session $t already exists"; done
grep -qE '/home/|/Users/|witcheer|voxelbench|basalt93|b93' "$PROMPT" && nl_fail "prompt file names a home path, the account or the harness dir"
if nl_skip_exact "$ART"; then nl_say "nothing to run, item complete ($ART exists)"; exit 0; fi
DL=$(nl_deadline_epoch); LEFT=$(( (DL-$(date +%s))/60 ))
nl_say "  RUN leg A (basalt): pack $([ -f "$PACK" ] && echo present || echo download) -> community row, rows basalt_spec0 + pp16k/pp48k, $NTAKES takes (spec 0), rows basalt_default ; artefacts $RUN/rows/basalt_spec0/dec-*.json, $RUN/takes/basalt-*/rec/result.json"
nl_say "  RUN leg B (llama): gguf $([ -f "$GG1" ] && echo present || echo download) -> ladder [$LADDER], rows llama_plain + pp16k/pp48k, $NTAKES takes, rows llama_ngram ; artefacts $RUN/ladder.json, $RUN/rows/llama_plain/dec-*.json, $RUN/takes/llama-*/rec/result.json"
nl_say "  RUN clip -> $RUN/out/basalt-final.mp4 ; RUN artefact -> $ART ; hard stop $(date -d "@$DL" '+%a %H:%M') local, $LEFT min away"
if [ "$SMOKE" != 1 ] && [ -z "$STUB_B$STUB_L" ]; then
  [ "$LEFT" -ge 150 ] || { nl_say "  under 150 min left in the window: not starting (two downloads + drain + 2 legs would be cut short; exit 3, resumable)"; exit 3; }
fi
if [ "${DRYRUN:-0}" = "1" ] || [ ! -f "$MANIFEST" ]; then nl_manifest_init dl_basalt community basalt_rows basalt_takes dl_llama ladder llama_rows llama_takes gate render; fi
nl_gate

# ---- phase 2 ----
wait_mem_free() { local i v; for i in $(seq 1 90); do v=$(mem_used); [ "${v:-99999}" -lt 1024 ] && { nl_say "  card clear (${v} MiB)"; return 0; }; sleep 2; done; nl_say "  card NOT clear after 180 s (${v} MiB)"; return 1; }
stop_srv() { # own server only: the pid this item started
  [ -n "$SP" ] || return 0
  kill -TERM "$SP" 2>/dev/null
  for i in $(seq 1 40); do kill -0 "$SP" 2>/dev/null || break; sleep 2; done
  kill -KILL "$SP" 2>/dev/null; [ -n "$STUB_B$STUB_L" ] || pkill -KILL -x basalt 2>/dev/null; sleep 3; SP=""; }
donald_stop() { [ "$DONALD_STOPPED" = 1 ] && [ "$DONALD_BACK" = 0 ] && return 0; nl_say "draining Donald (sudo -n systemctl stop llama-server.service)"; sudo -n systemctl stop llama-server.service; DONALD_STOPPED=1; DONALD_BACK=0; sleep 5
  local v i; for i in $(seq 1 60); do v=$(mem_used); [ "${v:-99999}" -lt 1024 ] && { nl_say "drain clear (${v} MiB)"; return 0; }; sleep 2; done
  nl_say "card memory still ${v:-?} MiB after the drain: another process owns the card"; return 1; }
donald_restore() {
  [ "$DONALD_STOPPED" = 1 ] && [ "$DONALD_BACK" = 0 ] || return 0
  DONALD_BACK=1; nl_say "restoring Donald (sudo -n systemctl start llama-server.service)"
  sudo -n systemctl start llama-server.service
  for i in $(seq 1 60); do v=$(curl -s -m 5 localhost:8090/health); [[ "${v,,}" == *ok* ]] && { nl_say "Donald healthy again after ~$((i*5)) s"; return 0; }; sleep 5; done
  nl_say "WARNING: Donald /health did not answer within 5 min after the restore"; }
cleanup() { for t in b93-llama b93-basalt; do tmux kill-session -t "=$t" 2>/dev/null; done; stop_srv; [ -n "$STUB_B$STUB_L" ] || wait_mem_free || true; donald_restore; }
trap cleanup EXIT
mkdir -p "$RUN/out" "$RUN/takes" "$RUN/rows"
t0=$(date +%s)
fail() { nl_mark "$1" failed "$2"; nl_say "FAIL: $2"; exit 1; }
DEADLINE=$(nl_deadline_epoch)
take_done() { [ -f "$RUN/takes/$1/rec/result.json" ] && grep -qE '"end": *"(done|wall)"' "$RUN/takes/$1/rec/result.json" 2>/dev/null; }
rows_done() { local n; n=$(ls "$RUN/rows/$1"/dec-*.json 2>/dev/null | wc -l); [ "$n" -ge $((3*REPS)) ] && { [ "$2" = nopp ] || { [ -f "$RUN/rows/$1/pp16k.json" ] && [ -f "$RUN/rows/$1/pp48k.json" ]; }; }; }
dl() { # <repo> <dir> <files...>: size-checked HF pull, guard 125 GB free after
  local repo=$1 dir=$2; shift 2; python3 "$H/dl_verify.py" "$repo" "$dir" 125 "$@" 2>&1 | tr '\r' '\n' | grep -E "^(need|OK|BAD|REFUSED|FAIL)" | sed 's/^/    dl: /'; return "${PIPESTATUS[0]}"; }
oai() { # <arm> <name> <port> <alias> <extra args...>: one counted request, resumable
  local arm=$1 name=$2 port=$3 alias=$4; shift 4; local out=$RUN/rows/$arm/$name.json
  [ -f "$out" ] && grep -q '"timings": {' "$out" 2>/dev/null && return 0
  python3 "$H/oai_bench.py" --port "$port" --model "$alias" --corpus "$CORPUS" --calib "$RUN/calib.json" --out "$out" "$@" 2>&1 | sed "s/^/    [$arm $name] /" | cut -c1-230; }
rows_arm() { # <arm> <port> <alias> <withpp: pp|nopp>
  local arm=$1 port=$2 alias=$3 pp=$4 rep t tmp
  mkdir -p "$RUN/rows/$arm"
  [ -f "$RUN/rows/$arm/warmup.json" ] || python3 "$H/oai_bench.py" --port "$port" --model "$alias" --corpus "$CORPUS" --calib "$RUN/calib.json" --task conv --max-tokens 200 --temp 0.6 --seed 1 --think off --out "$RUN/rows/$arm/warmup.json" >/dev/null 2>&1
  for rep in $(seq 1 $REPS); do for t in conv essay lru; do
    tmp=0.6; [ $t = lru ] && tmp=0
    oai "$arm" "dec-$t-$rep" "$port" "$alias" --task $t --max-tokens 700 --temp $tmp --seed $((100+rep)) --think off
  done; done
  if [ "$pp" = pp ]; then
    oai "$arm" pp16k "$port" "$alias" --chars 67000 --task oneline --max-tokens 16 --temp 0 --cold --think off
    oai "$arm" pp48k "$port" "$alias" --chars 197000 --task oneline --max-tokens 16 --temp 0 --cold --think off
  fi; }
chat_smoke() { # <port> <alias>: a real tool-call request with thinking on: the chat template + tool parser must return tool_calls, else Hermes would not work on it
  python3 - "$1" "$2" <<'PYEOF'
import json, sys, time, urllib.request
port, alias = sys.argv[1], sys.argv[2]
body = {"model": alias, "messages": [{"role": "user", "content": "What time is it in Paris? Use the tool."}], "max_tokens": 3000, "temperature": 0.6, "stream": False,
        "tools": [{"type": "function", "function": {"name": "get_time", "description": "current time in a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]}
t = time.time()
try:
    r = json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions" % port, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=240).read())
except Exception as e:
    print("SMOKE FAIL request: %s" % str(e)[:160]); sys.exit(1)
m = r["choices"][0]["message"]
tc = m.get("tool_calls") or []
ok = bool(tc) and tc[0]["function"]["name"] == "get_time" and "city" in json.loads(tc[0]["function"]["arguments"] or "{}")
tm = r.get("timings") or {}
print(("SMOKE OK" if ok else "SMOKE FAIL") + " tool_calls=%d completion_tokens=%s decode=%.1f %.1fs %s" % (len(tc), (r.get("usage") or {}).get("completion_tokens"), tm.get("predicted_per_second") or 0, time.time() - t, json.dumps(tc[0]["function"]) if tc else json.dumps(m)[:200]))
sys.exit(0 if ok else 1)
PYEOF
}
serve_basalt() { # <arm: spec0|default>: sets SP
  local arm=$1 cfg=$RUN/basalt-$1.json spec="" t1=$(date +%s) i v
  [ "$arm" = spec0 ] && spec=', "spec": 0'
  printf '{"version": 2, "models": [{"alias": "fn", "model": "%s", "gpu": "0", "kv": "int8", "max-context": %s%s}], "engine": {"resident-experts": true, "vram-reserve-mib": 700}, "server": {"host": "127.0.0.1", "port": %s}}\n' "$PACK" "$CTXCFG" "$spec" "$BPORT" > "$cfg"
  ( cd "$SRC" && exec bin/basalt serve --config "$cfg" --python "$VENV/bin/python" ) >> "$RUN/server-basalt-$arm.log" 2>&1 &
  SP=$!
  for i in $(seq 1 180); do
    v=$(curl -s -m 3 http://127.0.0.1:$BPORT/health)
    [[ "$v" =~ \"loaded\":\ *true ]] && { nl_say "  basalt ($arm) up after $(( $(date +%s)-t1 )) s (VRAM $(mem_used) MiB); $(grep -E 'loaded +[0-9.]+GiB|resident=' "$RUN/server-basalt-$arm.log" | tail -2 | cut -c1-120 | tr '\n' ' ')"; return 0; }
    kill -0 "$SP" 2>/dev/null || break; sleep 2; done
  return 1; }
serve_llama() { # <n_cpu_moe> <tag> [extra flags...]: sets SP
  local n=$1 tag=$2 t1=$(date +%s) i; shift 2
  LD_LIBRARY_PATH=$LB "$LB/llama-server" -m "$GG1" --port $LPORT --host 127.0.0.1 -ngl 99 --n-cpu-moe "$n" -fa on --jinja -c $CTXCFG -ctk q8_0 -ctv q8_0 --parallel 1 --alias fn "$@" >> "$RUN/server-llama-$tag.log" 2>&1 &
  SP=$!
  for i in $(seq 1 180); do
    curl -s -m 3 http://127.0.0.1:$LPORT/health | grep -qi ok && { nl_say "  llama-server ($tag, n-cpu-moe $n) up after $(( $(date +%s)-t1 )) s (VRAM $(mem_used) MiB)"; return 0; }
    kill -0 "$SP" 2>/dev/null || break; sleep 2; done
  return 1; }
run_takes() { # <tag> <n> <upstream>
  local tag=$1 nt=$2 up=$3 n td left w rc
  for n in $(seq 1 "$nt"); do
    td=$RUN/takes/$tag-$n
    if take_done "$tag-$n"; then nl_say "  take $tag-$n: already done (resume), kept"; continue; fi
    left=$(( DEADLINE - $(date +%s) )); [ -n "$STUB_B$STUB_L" ] && left=99999
    if [ "$left" -lt 660 ]; then nl_say "  take $tag-$n: only $((left/60)) min left (< 11), skipped"; SHORT=1; continue; fi
    w=$(( WALLMIN * 60 )); [ "$((left - 360))" -lt "$w" ] && w=$(( left - 360 ))
    [ -d "$td" ] && mv "$td" "$td.partial-$(date +%s)"
    nl_say "=== take $tag-$n (wall cap $w s, $((left/60)) min left in the window) ==="
    UPSTREAM=$up ALIAS=fn TAG=$tag-$n PROMPT=$PROMPT SEEDDIR=$SEEDDIR WALL_S=$w CTXCFG=$CTXCFG FR_PORT=$PPORT FR_PY=$RPY FR_GPY=$GPY FR_CHECKOUT=$CHECKOUT FR_QUIET=$QUIET THINK=on ARENA=/tmp/arena93 bash "$H/b93_take.sh" "$td" >> "$RUN/take.log" 2>&1; rc=$?
    nl_say "  take $tag-$n: rc=$rc $(tail -1 "$RUN/take.log" | cut -c1-200)"
    nl_say "  take $tag-$n: $(cat "$td/rec/rc" 2>/dev/null) $(cat "$td/rec/result.json" 2>/dev/null)"
    [ "$rc" = 0 ] || { nl_say "  take $tag-$n: harness rc $rc (see $RUN/take.log)"; SHORT=1; }
    mkdir -p "$td/gate"
    nl_say "  gate $tag-$n: $("$GPY" "$H/b93_gate.py" "$td/work" "$SEEDDIR/tests" "$td/gate" --pytest-python "$GPY" --seed-pkg "$SEEDDIR/shopcart" 2>&1 | head -1 | cut -c1-220)"
  done; }
takes_pending() { local tag=$1 n=$2 i; for i in $(seq 1 "$n"); do take_done "$tag-$i" || return 0; done; return 1; }

if [ -z "$STUB_B$STUB_L" ]; then
  FOREIGN=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ' | tr '\n' ' ')
  nl_say "  card owners at start: ${FOREIGN:-none}; tmux: $(tmux ls -F '#{session_name}' 2>/dev/null | tr '\n' ' ')"
  [ -z "$FOREIGN" ] || fail dl_basalt "a process other than Donald holds the card: $FOREIGN"
fi

# =============================== leg A: Basalt ===============================
if want_leg basalt; then
  if rows_done basalt_spec0 pp && rows_done basalt_default nopp && ! takes_pending basalt "$NTAKES" && [ -f "$RUN/community.row.json" ]; then
    nl_say "leg A: everything already done (resume), pack not needed"; for ph in dl_basalt community basalt_rows basalt_takes; do nl_mark $ph done "resume: skipped"; done
  else
    nl_mark dl_basalt running
    if [ -n "$STUB_B" ]; then nl_mark dl_basalt done "stub"
    elif [ "$(stat -c %s "$PACK" 2>/dev/null)" = "$PACK_BYTES" ]; then nl_say "  pack present, $PACK_BYTES bytes"; nl_mark dl_basalt done "present"
    else
      nl_say "  downloading the .basalt pack (70.4 GB, guard: free after >= 125 GB)"
      dl jesdga/Qwen3.8-Flash-Next-Basalt "$PACK_DIR" Qwen3.8-FN-GSQ-RCO-Q2_0.basalt && [ "$(stat -c %s "$PACK" 2>/dev/null)" = "$PACK_BYTES" ] || { nl_say "  pack download failed or refused"; fail dl_basalt "pack download"; }
      nl_mark dl_basalt done "downloaded"
    fi
    if [ -z "$STUB_B" ]; then donald_stop || fail community "card not free"; fi
    # --- the project's own community row (cold engines, 3 starts), else the plain row
    nl_mark community running
    if [ -f "$RUN/community.row.json" ]; then nl_say "  community row present"
    elif [ -n "$STUB_B" ]; then echo '{"stub": true}' > "$RUN/community.row.json"
    else
      ( cd "$SRC" && timeout 900 bin/basalt-bench --community --model "$PACK" --submitter b93-run --community-file "$RUN/community-benchmarks.json" ) > "$RUN/community.log" 2>&1; crc=$?
      if [ "$crc" = 0 ]; then
        "$GPY" - "$RUN/community.log" "$RUN/community.row.json" <<'PYEOF'
import json, sys
log, out = sys.argv[1:3]
txt = open(log).read()
row = None
i = txt.find("COMMUNITY ROW:")
if i >= 0:
    j = txt.find("{", i)
    try:
        row, _ = json.JSONDecoder().raw_decode(txt[j:])      # the row the bench printed (the store file is not canonical in a fresh run dir, so the bench cannot append to it)
    except Exception:
        row = None
json.dump({"source": "community", "row": row}, open(out, "w"), indent=1)
print("community row:", None if row is None else {k: row.get(k) for k in ("prefill_16k_tps", "prefill_64k_tps", "decode_tps", "quant", "ram_gib")})
PYEOF
        nl_say "  community row: $(cut -c1-200 "$RUN/community.row.json" | tr '\n' ' ' | cut -c1-260)"
      else
        nl_say "  community preset rc=$crc (refused or failed; $(grep -iE 'spill|refus|free RAM|SSD' "$RUN/community.log" | head -1 | cut -c1-160)); fallback: plain basalt-bench -t conv,struct,prose -p 16384"
        ( cd "$SRC" && timeout 900 bin/basalt-bench --model "$PACK" -t conv,struct,prose -p 16384 -o json -- --kv int8 ) > "$RUN/community-fallback.json" 2>> "$RUN/community.log"
        echo "{\"source\": \"plain-bench-fallback\", \"community_rc\": $crc, \"file\": \"community-fallback.json\"}" > "$RUN/community.row.json"
      fi
    fi
    nl_mark community done
    # --- no-draft server: decode rows, prefill rows, the Hermes takes
    nl_mark basalt_rows running
    if rows_done basalt_spec0 pp && ! takes_pending basalt "$NTAKES"; then nl_say "  basalt spec0: rows and takes already done"
    else
      if [ -n "$STUB_B" ]; then UPB=$STUB_B; else serve_basalt spec0 || { stop_srv; fail basalt_rows "Basalt did not come up (see $RUN/server-basalt-spec0.log)"; }; UPB=http://127.0.0.1:$BPORT; fi
      SM=$(chat_smoke "${UPB##*:}" fn); nl_say "  tool-call smoke (basalt spec0): $SM"
      case "$SM" in "SMOKE OK"*) ;; *) stop_srv; fail basalt_rows "Basalt tool-call smoke failed: Hermes cannot run on it";; esac
      curl -s -m 5 "$UPB/slots" > "$RUN/basalt-slots-idle.json" 2>/dev/null
      rows_arm basalt_spec0 "${UPB##*:}" fn pp
      nl_mark basalt_rows done "spec0 rows"
      nl_mark basalt_takes running
      run_takes basalt "$NTAKES" "$UPB"
      curl -s -m 5 "$UPB/metrics" > "$RUN/basalt-metrics-end.json" 2>/dev/null
      stop_srv; [ -n "$STUB_B" ] || wait_mem_free || fail basalt_takes "card not clear after the Basalt stop"
      nl_mark basalt_takes done "$NTAKES takes"
    fi
    # --- default (MTP) server: drafting rows
    if rows_done basalt_default nopp; then nl_say "  basalt default: rows already done"
    else
      if [ -n "$STUB_B" ]; then UPB=$STUB_B; else serve_basalt default || { stop_srv; fail basalt_rows "Basalt (default) did not come up"; }; UPB=http://127.0.0.1:$BPORT; fi
      rows_arm basalt_default "${UPB##*:}" fn nopp
      stop_srv; [ -n "$STUB_B" ] || wait_mem_free || fail basalt_rows "card not clear after the Basalt (default) stop"
    fi
    if [ -z "$STUB_B" ] && [ "$SHORT" = 0 ] && [ -f "$PACK" ] && [ -z "${B93_KEEP:-}" ]; then
      if rows_done basalt_spec0 pp && rows_done basalt_default nopp && ! takes_pending basalt "$NTAKES"; then
        [ "$ONLY" = basalt ] || { donald_restore; nl_say "  leg A complete: deleting the Basalt pack (70.4 GB) before the GGUF pull"; rm -f "$PACK"; nl_say "  disk after the delete: $(df --output=avail -BG / | tail -1 | tr -d ' ') free"; }
      fi
    fi
  fi
fi

# =============================== leg B: llama.cpp ===============================
if want_leg llama; then
  if rows_done llama_plain pp && rows_done llama_ngram nopp && ! takes_pending llama "$NTAKES" && [ -f "$RUN/ladder.json" ]; then
    nl_say "leg B: everything already done (resume), GGUF not needed"; for ph in dl_llama ladder llama_rows llama_takes; do nl_mark $ph done "resume: skipped"; done
  else
    export LD_LIBRARY_PATH=$LB
    nl_mark dl_llama running
    if [ -n "$STUB_L" ]; then nl_mark dl_llama done "stub"
    elif [ "$(stat -c %s "$GG1" 2>/dev/null)" = "$GG1_BYTES" ] && [ "$(stat -c %s "$GG2" 2>/dev/null)" = "$GG2_BYTES" ]; then nl_say "  GGUF pair present"; nl_mark dl_llama done "present"
    else
      nl_say "  downloading the GGUF pair (66.4 GB, guard: free after >= 125 GB)"
      dl ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF "$GG_DIR" Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
        && [ "$(stat -c %s "$GG1" 2>/dev/null)" = "$GG1_BYTES" ] && [ "$(stat -c %s "$GG2" 2>/dev/null)" = "$GG2_BYTES" ] || fail dl_llama "GGUF download"
      nl_mark dl_llama done "downloaded"
    fi
    if [ -z "$STUB_L" ]; then donald_stop || fail ladder "card not free"; fi
    # --- the n-cpu-moe ladder (pre-registered: best decode tok/s over conv/essay/lru, 300 tokens, seed 5; larger N wins ties)
    nl_mark ladder running
    if [ -f "$RUN/ladder.json" ]; then nl_say "  ladder present: best n-cpu-moe $("$GPY" -c "import json;print(json.load(open('$RUN/ladder.json')).get('best_n'))")"
    elif [ -n "$STUB_L" ]; then echo '{"best_n": 12, "stub": true, "take_config": "stub"}' > "$RUN/ladder.json"
    else
      mkdir -p "$RUN/ladder"
      for n in $LADDER; do
        serve_llama "$n" "ladder-$n" || { stop_srv; wait_mem_free || true; nl_say "  n-cpu-moe $n: did not load"; echo '{"loaded": false}' > "$RUN/ladder/n$n.json"; continue; }
        for t in conv essay lru; do tmp=0.6; [ $t = lru ] && tmp=0
          python3 "$H/oai_bench.py" --port $LPORT --model fn --corpus "$CORPUS" --calib "$RUN/calib.json" --task $t --max-tokens 300 --temp $tmp --seed 5 --think off --out "$RUN/ladder/n$n-$t.json" >/dev/null 2>&1; done
        nl_say "  ladder n=$n: $(for t in conv essay lru; do "$GPY" -c "import json;d=json.load(open('$RUN/ladder/n$n-$t.json'));print(round(d.get('decode_tok_s') or 0,1),end=' ')" 2>/dev/null; done) tok/s (conv essay lru), VRAM $(mem_used) MiB"
        echo '{"loaded": true}' > "$RUN/ladder/n$n.json"
        stop_srv; wait_mem_free || fail ladder "card not clear after the n=$n stop"
      done
      "$GPY" - "$RUN" "$LADDER" <<'PYEOF'
import json, os, sys
run, ladder = sys.argv[1], sys.argv[2].split()
rows = []
for n in ladder:
    ds = []
    for t in ("conv", "essay", "lru"):
        f = "%s/ladder/n%s-%s.json" % (run, n, t)
        try:
            d = json.load(open(f))
            tm = d.get("timings") or {}
            if tm.get("predicted_per_second") and tm.get("predicted_n", 0) >= 100:
                ds.append((tm["predicted_n"], tm["predicted_per_second"]))
        except Exception:
            pass
    rows.append({"n": int(n), "clean": len(ds), "decode_tok_s": round(sum(a for a, b in ds) / sum(a / b for a, b in ds), 1) if len(ds) == 3 else None})
ok = [r for r in rows if r["decode_tok_s"]]
best = sorted(ok, key=lambda r: (r["decode_tok_s"], r["n"]))[-1]["n"] if ok else None
json.dump({"best_n": best, "rows": rows, "take_config": "llama.cpp 18ab62a --n-cpu-moe %s -fa on -c 65536 q8_0 KV, no drafting" % best}, open(run + "/ladder.json", "w"), indent=1)
print("ladder best n-cpu-moe:", best, rows)
PYEOF
    fi
    BESTN=$("$GPY" -c "import json;print(json.load(open('$RUN/ladder.json')).get('best_n') or '')")
    nl_mark ladder done "best n-cpu-moe ${BESTN:-none}"
    [ -n "$BESTN" ] || { donald_restore; fail llama_rows "no ladder point loaded and answered"; }
    # --- plain llama.cpp: decode rows, prefill rows, the Hermes takes
    nl_mark llama_rows running
    if rows_done llama_plain pp && ! takes_pending llama "$NTAKES"; then nl_say "  llama plain: rows and takes already done"
    else
      if [ -n "$STUB_L" ]; then UPL=$STUB_L; else serve_llama "$BESTN" plain || { stop_srv; fail llama_rows "llama-server did not come up at n-cpu-moe $BESTN"; }; UPL=http://127.0.0.1:$LPORT; fi
      SM=$(chat_smoke "${UPL##*:}" fn); nl_say "  tool-call smoke (llama plain): $SM"
      case "$SM" in "SMOKE OK"*) ;; *) stop_srv; fail llama_rows "llama.cpp tool-call smoke failed: Hermes cannot run on it";; esac
      rows_arm llama_plain "${UPL##*:}" fn pp
      nl_mark llama_rows done "plain rows"
      nl_mark llama_takes running
      run_takes llama "$NTAKES" "$UPL"
      curl -s -m 5 "$UPL/slots" > "$RUN/llama-slots-end.json" 2>/dev/null
      stop_srv; [ -n "$STUB_L" ] || wait_mem_free || fail llama_takes "card not clear after the llama stop"
      nl_mark llama_takes done "$NTAKES takes"
    fi
    # --- llama.cpp n-gram drafting rows
    if rows_done llama_ngram nopp; then nl_say "  llama ngram: rows already done"
    else
      if [ -n "$STUB_L" ]; then UPL=$STUB_L; else serve_llama "$BESTN" ngram --spec-type ngram-mod || { stop_srv; fail llama_rows "llama-server (ngram-mod) did not come up"; }; UPL=http://127.0.0.1:$LPORT; fi
      rows_arm llama_ngram "${UPL##*:}" fn nopp
      stop_srv; [ -n "$STUB_L" ] || wait_mem_free || fail llama_rows "card not clear after the llama ngram stop"
    fi
    if [ -z "$STUB_L" ] && [ "$SHORT" = 0 ] && [ -z "${B93_KEEP:-}" ] && rows_done llama_plain pp && rows_done llama_ngram nopp && ! takes_pending llama "$NTAKES"; then
      nl_say "  leg B complete: deleting the GGUF pair (66.4 GB)"; rm -rf "$GG_DIR"; nl_say "  disk after the delete: $(df --output=avail -BG / | tail -1 | tr -d ' ') free"
    fi
  fi
fi

# the card is free from here: Donald back, then the CPU-only legs
if [ -z "$STUB_B$STUB_L" ]; then donald_restore; fi
nl_say "card released after $(( ($(date +%s)-t0)/60 )) min (start to restore), the rest is CPU"

# ---- offline re-count of every take's HUD series against the engine's own numbers (CPU, seconds) ----
for td in "$RUN"/takes/*; do
  case "$td" in *.partial-*) continue;; esac
  [ -s "$td/hud/hud.jsonl" ] || continue
  python3 "$H/b93_recount.py" "$td" --json "$RUN/out/recount-$(basename "$td").json" > /dev/null 2>&1
  nl_say "  recount $(basename "$td"): $(python3 -c "import json;d=json.load(open('$RUN/out/recount-$(basename "$td").json'));print('engine',d['engine'],'recount',d['new_series_final'],'engine usage',d['exact_sum_predicted_n'],'hud as recorded',d['recorded_hud_final'],'(%s%%)'%d['hud_vs_recount_pct'],'HUD_OK' if d['hud_ok'] else 'HUD_OFF','recount_vs_engine','OK' if d['pass'] else 'OFF')" 2>&1 | cut -c1-200)"
done
# ---- which take goes in the clip (rule fixed in the header: take 1 of each engine unless a recorder fault) ----
"$RPY" - "$RUN" "$NTAKES" <<'PYEOF' > "$RUN/pair.json"
import json, os, sys
run, takes = sys.argv[1], int(sys.argv[2])
def ok(d):
    try:
        r = json.load(open(d + "/rec/result.json"))
        return r.get("end") in ("done", "wall") and os.path.getsize(d + "/rec/events-a.jsonl") > 1000 and os.path.getsize(d + "/hud/hud.jsonl") > 100
    except Exception:
        return False
pick, why = {}, {}
for e in ("llama", "basalt"):
    for n in range(1, takes + 1):
        d = "%s/takes/%s-%d" % (run, e, n)
        if ok(d):
            pick[e] = d
            why[e] = "take %d%s" % (n, "" if n == 1 else " (take 1 had a recorder fault)")
            break
    else:
        pick[e], why[e] = None, "no usable take"
print(json.dumps({"left": pick["llama"], "right": pick["basalt"], "why": why}))
PYEOF
LT=$($RPY -c "import json;print(json.load(open('$RUN/pair.json'))['left'] or '')"); RT=$($RPY -c "import json;print(json.load(open('$RUN/pair.json'))['right'] or '')")
nl_say "  clip pair: $(cat "$RUN/pair.json")"
nl_mark gate done "gates ran per take"
# ---- summary (pre-registered rule) then clip (CPU) ----
"$GPY" "$H/b93_summary.py" "$RUN" "$SLUG" "$REGIME" > "$RUN/summary.pre.json" 2>>"$RUN/render.log"
VERD=$("$GPY" -c "import json;d=json.load(open('$RUN/summary.pre.json'));print(d['verdict'], '(provisional)' if d['provisional'] else '')" 2>/dev/null)
nl_say "  verdict (pre-registered rule): ${VERD:-unreadable}"
nl_mark render running
rr=1
if [ -n "$LT" ] && [ -n "$RT" ]; then
  SPK=$("$GPY" -c "import json;d=json.load(open('$RUN/summary.pre.json'));a=d['arms'];m=max([(a.get(k) or {}).get('decode_tok_s') or 0 for k in ('llama_plain','basalt_spec0')]+[50]);print(int(-(-m*1.3//50)*50))" 2>/dev/null); SPK=${SPK:-200}
  "$RPY" "$H/b93_render.py" race "$RUN/out/race.mp4" --left "$LT" --right "$RT" --ffmpeg "$FF" --stills "$RUN/out/stills" --spark-max "$SPK" \
     --real-max-s 40 --open-s 10 --close-s 10 --target-main-s 30 --cut-after-first 6 --pre-s 2 --hold-s 1.5 >> "$RUN/render.log" 2>&1; rr=$?
  if [ "$rr" = 0 ]; then
    "$GPY" "$H/b93_summary.py" "$RUN" "$SLUG" "$REGIME" > "$RUN/summary.json.tmp" 2>>"$RUN/render.log" && mv "$RUN/summary.json.tmp" "$RUN/summary.card.json"
    "$RPY" "$H/b93_facts.py" "$RUN/summary.card.json" "$RUN/facts.json" >> "$RUN/render.log" 2>&1 && \
    "$RPY" "$H/b93_render.py" card "$RUN/out/card.mp4" --facts "$RUN/facts.json" --ffmpeg "$FF" --seconds 8 >> "$RUN/render.log" 2>&1 && \
    "$FF" -nostdin -hide_banner -loglevel error -y -i "$RUN/out/card.mp4" -i "$RUN/out/race.mp4" \
      -filter_complex "[0:v]setsar=1,fps=25[a];[1:v]setsar=1,fps=25[b];[a][b]concat=n=2:v=1:a=0[v]" -map "[v]" -c:v libx264 -crf 20 -pix_fmt yuv420p -movflags +faststart "$RUN/out/basalt-final.mp4" </dev/null >> "$RUN/render.log" 2>&1
    rr=$?
  fi
else nl_say "  no usable take on one of the engines: no race clip (the table and verdict still stand)"; fi
"$RPY" - "$RUN/out" "$RUN/out/contact.png" <<'PYEOF' 2>>"$RUN/render.log"
import glob, os, sys
from PIL import Image
out = sys.argv[1]
fs = sorted(glob.glob(out + "/stills/*.png"))[:8]
fs += [p for p in (out + "/card.mp4.png",) if os.path.exists(p)]
if fs:
    w, h = 640, 360
    sheet = Image.new("RGB", (w * 3, h * 3), "#000")
    for i, f in enumerate(fs[:9]):
        sheet.paste(Image.open(f).convert("RGB").resize((w, h)), ((i % 3) * w, (i // 3) * h))
    sheet.save(sys.argv[2])
PYEOF
nl_say "  render rc=$rr; final clip: $([ -f "$RUN/out/basalt-final.mp4" ] && du -h "$RUN/out/basalt-final.mp4" | cut -f1 || echo missing)"
if [ "$SHORT" = 1 ] || [[ "$VERD" == *provisional* ]] || [ -n "$ONLY" ]; then
  nl_say "  a leg could not finish inside the window or only one leg ran (verdict ${VERD:-?}): exit 3, resumable, summary.json NOT written (PROVISIONAL: $RUN/summary.pre.json)"
  nl_mark render failed "incomplete, resumable"
  exit 3
fi
"$GPY" "$H/b93_summary.py" "$RUN" "$SLUG" "$REGIME" > "$ART" 2>>"$RUN/render.log"
nl_judge_artifact render "$rr" "$ART" "$t0" "$(( ($(date +%s)-t0)/60 )) min, render rc $rr" || { nl_say "FAIL: no fresh summary.json or render rc $rr"; exit 1; }
nl_say "  summary: verdict ${VERD}; line 1: $("$GPY" -c "import json;d=json.load(open('$ART'));print(d.get('line1'))")"
nl_say "DONE $SLUG"
exit 0
