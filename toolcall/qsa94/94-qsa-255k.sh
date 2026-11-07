#!/bin/bash
# 94-qsa-255k.sh: idle-queue item (STAGED in ~/voxel-stage, 2026-11-06, mercury subagent for the GPU thread, mojo-scan pick C). Qwen Sparse Attention (QSA) on Qwen3.8-27B with 255K tokens of context on one RTX 5090:
#   the pentaCoxian fork of llama.cpp + the QSA IQ3_S file + the DFlash2 drafter, against STOCK llama.cpp on the non-QSA ISTA IQ3_S file of the same quantiser, same drafter, same flags.
# WHAT: the fork author's README (fetched 2026-11-06) gives, for an RTX 5090, greedy 141.8 tok/s and sampled 142.4 tok/s at 255K input, cold prefill 2,756 tok/s (93 s), peak 24.0 GB; and for an RTX 4090 stock llama.cpp at 40.6 tok/s against 114.7.
#   The stock figure on a 5090 is not published: this item fills that gap. Rows: cold prompts at 0 / 64K / 128K / 255K tokens x 2 seeds (code task) on both engines, a 20-request early-stop probe (the README reports ~7% of replies stopping within 16 tokens, cause unknown),
#   then one real Hermes Agent session per engine that answers 24 questions with known answers from the whole Hermes docs text (about 240K tokens carried as ephemeral prefill messages), answer keys outside the agent's reach (it has the todo toolset only).
# MOJO: two stacked live counters chewing the same 255K-token prompt (QSA holding ~140 tok/s while stock slows), then Hermes Agent answering questions about the whole docs set locally with the context bar nearly full | ahead + intense (the Qwen4 attention design grafted on today's 27B, tokens-deep decode that does not fall off a cliff) | On one RTX 5090, Qwen3.8-27B with Qwen Sparse Attention decodes [Q] tok/s with 255K tokens of context against [S] tok/s for the dense-attention build on stock llama.cpp: [R]x; Hermes Agent answered [k] of 24 questions about the whole docs set from one prompt
# EV: private (projected post angle and reach bracket live in the ops vault status note 99-personal/benchmark-rig/qsa94-status.md)
# serves: content (release-adjacent long-context verdict with a picture; also a rehearsal of how sparse attention behaves on our card before the Qwen 4 day-0) + hardening (long-context agent session off the main gateway).
#
# PRE-REGISTERED (h/PREREG.md, sha256-frozen in FROZEN.sha256 BEFORE any counted output; the PLAN phase refuses to run when a hash differs and re-runs qa_build.py to compare the question set; `c94_summary.py --selftest` proves the branches):
#   R255 = qsa decode / stock decode at 255K (code task, 2 seeds). QSA AHEAD >= 1.5 | LEVEL 0.8..1.5 | STOCK AHEAD < 0.8 | BROKEN | INCOMPLETE (exit 3). Claim check vs the author's 141.8: HELD (>= 127.6) | ABOVE (> 156.0) | BELOW.
#   A tie or a loss is a valid post ("one rig"). The 7% early-stop rate and the two-hop recall (11/16 vs 12/16) are quoted as the AUTHOR's. Hermes Agent is never the subject of a failure.
#
# DISK PLAN (operator brief: free >= 140 GB after staging, runs never under 125 GB): this item downloads NOTHING. Its models (QSA IQ3_S 12.3 GB + drafter 1.0 GB + projector 0.9 GB + stock IQ3_S 12.1 GB = 26.4 GB) and the two builds
#   (~0.7 GB) are already on disk from staging (~/models/qsa94/{qsa,stock}); the run writes < 2 GB (rows, takes, clip). It never moves the free space below what staging left. The big downloads (the 70.4 GB Basalt pack and 66.4 GB GGUF pair of item 93)
#   are in-run and sequential in item 93, so no two big files are ever on disk together. If both items are queued, 93 sorts first: do not start 94 while 93 has a pack on disk unless free >= 140 GB (the PLAN phase checks 125 GB).
#
# POSTABLE ARTEFACT (planned here): runs/<slug>/out/qsa-final.mp4 (16:9 1920x1080, ~65-80 s): result card 8 s FIRST (title = the verdict in words, one column per engine, depth-ladder numbers, grey how-measured lines, footer with the claim tested and the limits),
#   then the two engines chewing the same 255K prompt (stacked counters: prefill progress + prefill tok/s, then decode tok/s; a visible speed pill because the 230 s stock prefill is shown faster than real time; the clock stays real seconds),
#   then the two Hermes sessions side by side at real speed with the context bar and server-token counters, cut 6 s after the faster session has answered its last question. HUD LEAK READ: account name / home path / tailnet address / hostname / e-mail drawn as blocks.
#   a timelapse pill is never a speed claim.
#
# CARD TIME (honest, ESTIMATE from the 2026-11-06 staging smokes, see the status note): QSA load 12-31 s, stock 3-9 s. QSA cold rows: 22 s (64K) + 43 s (128K) + 90 s (255K) per seed; stock 27 s + 74 s + 232 s per seed (smokes 1+2).
#   per arm: QSA ~7 min + probe ~1 min + session ~3 min; stock ~15 min + probe ~1 min + session ~5 min => ~32-40 min of card typical. Worst case ~110 min (each session at its 40-min wall cap).
#   The README's "~40 min Hermes session" is NOT what this does: 24 cached-prefix questions take minutes, not 40 (cap 40 min); say so in the post. CPU after the card is released: recounts + render ~6-10 min.
# WINDOW RULE: needs >= 100 min left in the block or exits 3 without draining Donald (resumable); a session is skipped when under 15 min remain. Weekend blocks never yield; 14:10-17:10 weekday duck blocks are not for this item.
#
# test knobs (empty in production): C94_SMOKE=1 (slug -smoke, depths "0 64000", 1 seed, 3 probe requests, 6 questions, wall 6 min, no window rule), C94_STUB_QSA / C94_STUB_STOCK = http://127.0.0.1:PORT (CPU mocks: no Donald drain, no server start),
#   C94_SLUG (CPU proofs write to their own slug), C94_DEPTHS, C94_SEEDS, C94_PROBE_N, C94_NQ, C94_WALL_MIN, C94_QUIET, MANIFEST_OVERRIDE (staging DRYRUN only), C94_NO_SELFTEST=1, C94_ONLY=qsa|stock.
set -uo pipefail
NL_TAG=qsa94
MANIFEST=/home/witcheer/qsa94-status.json
[ -n "${MANIFEST_OVERRIDE:-}" ] && MANIFEST=$MANIFEST_OVERRIDE
export MANIFEST
. /home/witcheer/night-lib.sh
Q=/home/witcheer/voxel-stage/qsa94
H=$Q/h
QB=$Q/src/build/bin                                          # the fork, fe08b16, arch 120
SB=/home/witcheer/voxel-stage/llama-stock-18ab62a/build/bin   # mainline 18ab62a (shared with item 93; read-only here)
MQ=/home/witcheer/models/qsa94/qsa; MS=/home/witcheer/models/qsa94/stock
QSA_GGUF=$MQ/Qwen3.8-27B-QSA-GSQ-RCO-IQ3_S-mtp.gguf; DRAFT=$MQ/Qwen3.8-27B-DFlash2-IQ4_XS.gguf; STOCK_GGUF=$MS/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
QSA_BYTES=12329762336; DRAFT_BYTES=1038313056; STOCK_BYTES=12120016960
CORPUS=/home/witcheer/voxel-stage/docs-corpus/llms-full-2026-11-06.txt
GPY=/home/witcheer/benchmark-rig/.venv/bin/python
RPY=/home/witcheer/scratch/race/venv/bin/python
FF=/home/witcheer/voxelbench/ffmpeg
CHECKOUT=/home/witcheer/.hermes/hermes-agent; HBIN=$CHECKOUT/.hermes/bin/hermes; HERMES_PIN=9bda188
QSA_SHA=fe08b16; STOCK_SHA=18ab62a
SPORT=8276; PPORT=8293                                       # own ports: never :8090 (Donald), 8261-8275 (staging smokes, item 93), 8291 (item 89)
CTXCFG=262144
SMOKE=${C94_SMOKE:-0}; ONLY=${C94_ONLY:-}; STUB_Q=${C94_STUB_QSA:-}; STUB_S=${C94_STUB_STOCK:-}
DEPTHS=${C94_DEPTHS:-"0 64000 128000 255000"}; SEEDS=${C94_SEEDS:-"11 12"}; PROBE_N=${C94_PROBE_N:-20}; NQ=${C94_NQ:-24}; WALLMIN=${C94_WALL_MIN:-40}; QUIET=${C94_QUIET:-20}
[ "$SMOKE" = 1 ] && { DEPTHS=${C94_DEPTHS:-"0 64000"}; SEEDS=${C94_SEEDS:-"11"}; PROBE_N=${C94_PROBE_N:-3}; NQ=${C94_NQ:-6}; WALLMIN=${C94_WALL_MIN:-6}; }
SLUG=${C94_SLUG:-qsa94-qsa-255k}; [ "$SMOKE" = 1 ] && SLUG=$SLUG-smoke
RUN=/home/witcheer/voxelbench/runs/$SLUG; ART=$RUN/summary.json
REGIME="Qwen3.8-27B GSQ-RCO IQ3_S-mtp: QSA build (fork $QSA_SHA, pentacoxian-dev QSA file) vs dense build (ISTA-DASLab file, mainline llama.cpp $STOCK_SHA), same DFlash2 IQ4_XS drafter (4 drafts, probabilistic draft sampling), one RTX 5090 32 GB, ctx $CTXCFG, q8_0 KV, one slot, no vision, --cache-ram 0; speed rows: cold prompts, code task, temp 1.0 top-p 0.95 top-k 20, thinking off, seeds ${SEEDS// /,}; Hermes: todo toolset only, compression off, docs as prefill, $NQ questions, temp 0.7 top-p 0.95 top-k 20, thinking off"
SP=""; DONALD_STOPPED=0; DONALD_BACK=0; SHORT=0
mem_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
want_arm() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0} smoke=$SMOKE only=${ONLY:-both} depths=[$DEPTHS] seeds=[$SEEDS] probe=$PROBE_N questions=$NQ wall=${WALLMIN}min stubQ=${STUB_Q:-none} stubS=${STUB_S:-none})"
nl_check_path "$GPY" "$RPY" "$HBIN" "$FF" "$CORPUS" "$H/PREREG.md" "$H/FROZEN.sha256" "$H/oai_bench.py" "$H/qa_build.py" "$H/qa/questions.json" "$H/qa/prefill.json" "$H/qa/keys/keys.json" \
  "$H/c94_take.sh" "$H/c94_proxy.py" "$H/c94_count.py" "$H/c94_recount.py" "$H/c94_record.py" "$H/c94_mkhome.py" "$H/c94_render.py" "$H/c94_summary.py" "$H/c94_facts.py"
if [ -z "$STUB_Q" ] && want_arm qsa; then nl_check_path "$QB/llama-server" "$QSA_GGUF" "$DRAFT"; fi
if [ -z "$STUB_S" ] && want_arm stock; then nl_check_path "$SB/llama-server" "$STOCK_GGUF" "$DRAFT"; fi
if [ -z "$STUB_Q$STUB_S" ]; then
  for f in "$QSA_GGUF:$QSA_BYTES" "$DRAFT:$DRAFT_BYTES" "$STOCK_GGUF:$STOCK_BYTES"; do [ "$(stat -c %s "${f%%:*}" 2>/dev/null)" = "${f##*:}" ] && nl_ok "size ${f##*:} $(basename "${f%%:*}")" || nl_fail "size of ${f%%:*} != ${f##*:}"; done
fi
# frozen: every file FROZEN.sha256 lists (paths relative to $H; the docs text included) must match
if (cd "$H" && sha256sum -c FROZEN.sha256 >/dev/null 2>&1); then nl_ok "frozen files match FROZEN.sha256 ($(wc -l < "$H/FROZEN.sha256") entries)"; else nl_fail "a frozen file differs from FROZEN.sha256: $(cd "$H" && sha256sum -c FROZEN.sha256 2>&1 | grep -v ': OK' | head -3 | tr '\n' ' ')"; fi
for f in PREREG.md c94_summary.py qa/keys/keys.json qa/questions.json llms-full-2026-11-06.txt; do grep -q "$f" "$H/FROZEN.sha256" || nl_fail "FROZEN.sha256 does not list $f"; done
QT=$(mktemp -d /tmp/c94qa.XXXXXX); "$GPY" "$H/qa_build.py" "$CORPUS" "$QT" >/dev/null 2>&1
for f in questions.json prefill.json keys/keys.json docs_slice.txt; do cmp -s "$QT/$f" "$H/qa/$f" 2>/dev/null || { [ "$f" = docs_slice.txt ] && [ ! -e "$H/qa/docs_slice.txt" ]; } || nl_fail "qa_build.py output $f differs from the frozen qa/$f"; done
rm -rf "$QT"
"$GPY" -c "import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$H/c94_proxy.py" "$H/c94_count.py" "$H/c94_recount.py" "$H/c94_record.py" "$H/c94_mkhome.py" "$H/c94_summary.py" "$H/c94_facts.py" "$H/oai_bench.py" "$H/qa_build.py" || nl_fail "harness syntax"
(cd / && "$RPY" -c "import ast,sys,pyte,PIL,fontTools; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$H/c94_render.py") || nl_fail "renderer deps/syntax"
bash -n "$H/c94_take.sh" || nl_fail "c94_take.sh syntax"
if [ "${C94_NO_SELFTEST:-0}" != 1 ]; then
  ST=$("$GPY" "$H/c94_summary.py" --selftest 2>&1); case "$ST" in *BAD*) nl_fail "verdict self-test: $ST";; *) nl_ok "verdict rule self-test: $(echo "$ST" | grep -c OK) branches OK";; esac
fi
# the question set must be answerable: every key value occurs in the prefill text exactly once as `key: value` or `key (default value` (checked on the frozen files)
"$GPY" - "$H" "$NQ" <<'PYEOF' || nl_fail "question set check"
import json, re, sys
H, nq = sys.argv[1], int(sys.argv[2])
qs = json.load(open(H + "/qa/questions.json")); keys = json.load(open(H + "/qa/keys/keys.json")); txt = json.load(open(H + "/qa/prefill.json"))[0]["content"]
assert len(qs) >= nq, "fewer questions than asked"
bad = [q["id"] for q in qs[:nq] if len(re.findall(r"(?<![\w])" + re.escape(keys[q["id"]]["key"]) + r"(?![\w])", txt)) != 1]
print("questions ok" if not bad else "ambiguous keys: %s" % bad); sys.exit(1 if bad else 0)
PYEOF
if want_arm qsa && [ -z "$STUB_Q" ]; then
  export LD_LIBRARY_PATH=$QB
  case "$("$QB/llama-server" --version 2>&1)" in *"$QSA_SHA"*) nl_ok "QSA fork build $QSA_SHA";; *) nl_fail "QSA fork build changed (expected $QSA_SHA)";; esac
  HQ=$("$QB/llama-server" --help 2>&1 || true); for f in --spec-type --spec-draft-sampling --spec-draft-n-max --ctx-checkpoints --checkpoint-min-step --cache-ram --metrics -ngld; do case "$HQ" in *"$f"*) ;; *) nl_fail "QSA llama-server flag $f missing";; esac; done
fi
if want_arm stock && [ -z "$STUB_S" ]; then
  export LD_LIBRARY_PATH=$SB
  case "$("$SB/llama-server" --version 2>&1)" in *"$STOCK_SHA"*) nl_ok "stock llama.cpp $STOCK_SHA";; *) nl_fail "stock build changed (expected $STOCK_SHA)";; esac
  HS=$("$SB/llama-server" --help 2>&1 || true); for f in --spec-type --spec-draft-sampling --spec-draft-n-max --ctx-checkpoints --checkpoint-min-step --cache-ram --metrics -ngld; do case "$HS" in *"$f"*) ;; *) nl_fail "stock llama-server flag $f missing";; esac; done
fi
HV=$(cd "$CHECKOUT" && git rev-parse --short HEAD 2>/dev/null); case "$HV" in "$HERMES_PIN"*) nl_ok "Hermes checkout $HV (pin $HERMES_PIN)";; *) nl_fail "Hermes checkout $HV != pin $HERMES_PIN";; esac
for dd in installs tools; do [ -d "/home/witcheer/.hermes/$dd" ] || nl_fail "main Hermes root has no $dd/ to share with the take homes"; done
FREE=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
nl_say "  disk plan: free ${FREE} GB now; this item downloads nothing (models 26.4 GB already on disk), writes < 2 GB; guard 125 GB"
[ -z "$STUB_Q$STUB_S" ] && { [ "$FREE" -ge 125 ] && nl_ok "disk ${FREE}G free >= 125G" || nl_fail "disk ${FREE}G < 125G guard"; }
for p in $SPORT $PPORT; do ss -ltnH "sport = :$p" 2>/dev/null | grep -q . && nl_fail "port $p is already in use"; done
for t in c94-qsa c94-stock; do tmux has-session -t "=$t" 2>/dev/null && nl_fail "tmux session $t already exists"; done
if nl_skip_exact "$ART"; then nl_say "nothing to run, item complete ($ART exists)"; exit 0; fi
DL=$(nl_deadline_epoch); LEFT=$(( (DL-$(date +%s))/60 ))
nl_say "  RUN arm qsa: server (fork) -> depth ladder [$DEPTHS] x seeds [$SEEDS] (artefact rows/qsa/dec-lru-d<depth>-s<seed>.json), prose 255K, early-stop probe x$PROBE_N, Hermes session (takes/qsa-1/rec/result.json)"
nl_say "  RUN arm stock: same on mainline llama.cpp + the dense file (rows/stock/..., takes/stock-1/...)"
nl_say "  RUN clip -> $RUN/out/qsa-final.mp4 ; RUN artefact -> $ART ; hard stop $(date -d "@$DL" '+%a %H:%M') local, $LEFT min away"
if [ "$SMOKE" != 1 ] && [ -z "$STUB_Q$STUB_S" ]; then
  [ "$LEFT" -ge 100 ] || { nl_say "  under 100 min left in the window: not starting (exit 3, resumable)"; exit 3; }
fi
if [ "${DRYRUN:-0}" = "1" ] || [ ! -f "$MANIFEST" ]; then nl_manifest_init qsa_rows qsa_session stock_rows stock_session render; fi
nl_gate

# ---- phase 2 ----
wait_mem_free() { local i v; for i in $(seq 1 90); do v=$(mem_used); [ "${v:-99999}" -lt 1024 ] && { nl_say "  card clear (${v} MiB)"; return 0; }; sleep 2; done; nl_say "  card NOT clear after 180 s (${v} MiB)"; return 1; }
stop_srv() { [ -n "$SP" ] || return 0; kill -TERM "$SP" 2>/dev/null; for i in $(seq 1 40); do kill -0 "$SP" 2>/dev/null || break; sleep 2; done; kill -KILL "$SP" 2>/dev/null; sleep 3; SP=""; }
donald_stop() { [ "$DONALD_STOPPED" = 1 ] && [ "$DONALD_BACK" = 0 ] && return 0; nl_say "draining Donald (sudo -n systemctl stop llama-server.service)"; sudo -n systemctl stop llama-server.service; DONALD_STOPPED=1; DONALD_BACK=0; sleep 5
  local v i; for i in $(seq 1 60); do v=$(mem_used); [ "${v:-99999}" -lt 1024 ] && { nl_say "drain clear (${v} MiB)"; return 0; }; sleep 2; done
  nl_say "card memory still ${v:-?} MiB after the drain: another process owns the card"; return 1; }
donald_restore() {
  [ "$DONALD_STOPPED" = 1 ] && [ "$DONALD_BACK" = 0 ] || return 0
  DONALD_BACK=1; nl_say "restoring Donald (sudo -n systemctl start llama-server.service)"
  sudo -n systemctl start llama-server.service
  for i in $(seq 1 60); do v=$(curl -s -m 5 localhost:8090/health); [[ "${v,,}" == *ok* ]] && { nl_say "Donald healthy again after ~$((i*5)) s"; return 0; }; sleep 5; done
  nl_say "WARNING: Donald /health did not answer within 5 min after the restore"; }
cleanup() { for t in c94-qsa c94-stock; do tmux kill-session -t "=$t" 2>/dev/null; done; stop_srv; [ -n "$STUB_Q$STUB_S" ] || wait_mem_free || true; donald_restore; }
trap cleanup EXIT
mkdir -p "$RUN/out" "$RUN/takes" "$RUN/rows"
t0=$(date +%s)
fail() { nl_mark "$1" failed "$2"; nl_say "FAIL: $2"; exit 1; }
DEADLINE=$(nl_deadline_epoch)
take_done() { [ -f "$RUN/takes/$1/rec/result.json" ] && grep -qE '"end": *"(done|wall)"' "$RUN/takes/$1/rec/result.json" 2>/dev/null; }
# questions: the first goes in as the prompt file, the rest as the recorder's follow-ups (both from the frozen questions.json)
"$GPY" - "$H" "$NQ" "$RUN" <<'PYEOF'
import json, sys
H, nq, run = sys.argv[1], int(sys.argv[2]), sys.argv[3]
qs = json.load(open(H + "/qa/questions.json"))[:nq]
open(run + "/first-question.txt", "w").write(qs[0]["q"] + "\n")
json.dump([q["q"] for q in qs[1:]], open(run + "/followups.json", "w"))
json.dump(qs, open(run + "/questions-used.json", "w"))
PYEOF

serve() { # <arm> : sets SP; the fork's serve.sh flags and env, QSA-only env only for the fork, plus --metrics (observability only)
  local arm=$1 bin gguf t1=$(date +%s) i UP
  if [ "$arm" = qsa ]; then bin=$QB; gguf=$QSA_GGUF; else bin=$SB; gguf=$STOCK_GGUF; fi
  (
  export LLAMA_QSA_RESERVE_BLK=1 LLAMA_QSA_BLK_REBUILD=1 LLAMA_QSA27B_MASKED_PREFILL_MAX=8192 GGML_CUDA_TOPK_DET=1 LLAMA_MTP_KV_ONLY=1 LLAMA_MTP_PAD=4 LLAMA_MTP_N_UBATCH=64 LLAMA_MTP_LAG=0 LLAMA_MTP_ASYNC=0
  export LLAMA_REUSE_OUT_ORDER=1 LLAMA_KV_PAD_GEO=2 LLAMA_CKPT_TAIL_UBATCH=1 GGML_FT_GRAPH_CACHE=0 GGML_FT_GRAPH_CACHE_MIN_NODES=32 CUDA_SCALE_LAUNCH_QUEUES=4x
  [ "$arm" = qsa ] && export LLAMA_QSA_RAW_RING=8192 MTMD_ON_DEMAND=1 GGML_CUDA_EVICT_MIB=2048
  export LLAMA_SAMPLE_TOPK=1 LLAMA_SPEC_PREFILL_TAIL=2048 LLAMA_GPU_ARGMAX=1 LD_LIBRARY_PATH=$bin
  exec "$bin/llama-server" -m "$gguf" --alias q27 -ngl 99 -fa on -np 1 --fit off -t 8 -c $CTXCFG -b 2048 -ub 256 -ctk q8_0 -ctv q8_0 -ctkd f16 -ctvd f16 --no-context-shift --jinja --cache-ram 0 --ctx-checkpoints 8 --checkpoint-min-step 1024 \
    -md "$DRAFT" -ngld 99 --spec-type draft-dflash --spec-draft-p-min 0 --spec-draft-n-max 4 --spec-draft-sampling probabilistic --metrics --host 127.0.0.1 --port $SPORT ) >> "$RUN/server-$arm.log" 2>&1 &
  SP=$!
  for i in $(seq 1 200); do
    curl -s -m 3 http://127.0.0.1:$SPORT/health | grep -qi ok && { nl_say "  $arm server up after $(( $(date +%s)-t1 )) s (VRAM $(mem_used) MiB)"; return 0; }
    kill -0 "$SP" 2>/dev/null || break; sleep 3; done
  return 1; }
oai() { # <arm> <name> <port> <extra...>: one counted request, resumable (skips when the result already holds timings)
  local arm=$1 name=$2 port=$3; shift 3; local out=$RUN/rows/$arm/$name.json
  [ -f "$out" ] && grep -q '"timings": {' "$out" 2>/dev/null && return 0
  python3 "$H/oai_bench.py" --port "$port" --model q27 --corpus "$CORPUS" --calib "$RUN/calib.json" --out "$out" "$@" 2>&1 | sed "s/^/    [$arm $name] /" | cut -c1-230; }
rows_arm() { # <arm> <port>
  local arm=$1 port=$2 d s i
  mkdir -p "$RUN/rows/$arm"
  [ -f "$RUN/rows/$arm/warmup.json" ] || python3 "$H/oai_bench.py" --port "$port" --model q27 --corpus "$CORPUS" --calib "$RUN/calib.json" --task conv --max-tokens 100 --temp 0.6 --seed 1 --think off --cold --out "$RUN/rows/$arm/warmup.json" >/dev/null 2>&1
  for d in $DEPTHS; do for s in $SEEDS; do
    if [ "$d" = 0 ]; then oai "$arm" "dec-lru-d$d-s$s" "$port" --task lru --max-tokens 256 --temp 1.0 --min-p 0 --seed $s --cold --think off --progress
    else
      ser=""; [ "$d" = 255000 ] && [ "$s" = "${SEEDS%% *}" ] && ser="--series $RUN/rows/$arm/series-d255000.series"
      oai "$arm" "dec-lru-d$d-s$s" "$port" --tokens $d --task lru --max-tokens 256 --temp 1.0 --min-p 0 --seed $s --cold --think off --progress $ser
    fi
  done; done
  case " $DEPTHS " in *" 255000 "*) oai "$arm" "dec-essay-d255000-s${SEEDS%% *}" "$port" --tokens 255000 --task essay --max-tokens 256 --temp 1.0 --min-p 0 --seed "${SEEDS%% *}" --cold --think off --progress;; esac
  # early-stop probe: same ~60K-token prefix, different seeds (first one reads it cold, the rest reuse it)
  for i in $(seq 1 "$PROBE_N"); do oai "$arm" "probe-$i" "$port" --tokens 60000 --task lru --max-tokens 64 --temp 1.0 --min-p 0 --seed $i --think off; done; }
rows_complete() { local arm=$1 d s n=0 want=0; for d in $DEPTHS; do for s in $SEEDS; do want=$((want+1)); [ -f "$RUN/rows/$arm/dec-lru-d$d-s$s.json" ] && grep -q '"timings": {' "$RUN/rows/$arm/dec-lru-d$d-s$s.json" && n=$((n+1)); done; done; [ "$n" -ge "$want" ] && [ "$(ls "$RUN/rows/$arm"/probe-*.json 2>/dev/null | wc -l)" -ge "$PROBE_N" ]; }
run_session() { # <arm> <upstream>
  local arm=$1 up=$2 td=$RUN/takes/$1-1 left w rc
  if take_done "$arm-1"; then nl_say "  session $arm: already done (resume), kept"; return 0; fi
  left=$(( DEADLINE - $(date +%s) )); [ -n "$STUB_Q$STUB_S" ] && left=99999
  if [ "$left" -lt 900 ]; then nl_say "  session $arm: only $((left/60)) min left (< 15), skipped"; SHORT=1; return 0; fi
  w=$(( WALLMIN * 60 )); [ "$((left - 420))" -lt "$w" ] && w=$(( left - 420 ))
  [ -d "$td" ] && mv "$td" "$td.partial-$(date +%s)"
  nl_say "=== Hermes session $arm (wall cap $w s, $((left/60)) min left in the window) ==="
  UPSTREAM=$up ALIAS=q27 TAG=$arm-1 PROMPT=$RUN/first-question.txt PROMPTS_FILE=$RUN/followups.json C94_PREFILL=$H/qa/prefill.json WALL_S=$w CTXCFG=$CTXCFG FR_PORT=$PPORT FR_PY=$RPY FR_GPY=$GPY FR_CHECKOUT=$CHECKOUT FR_QUIET=$QUIET THINK=off ARENA=/tmp/arena94 \
    bash "$H/c94_take.sh" "$td" >> "$RUN/take.log" 2>&1; rc=$?
  nl_say "  session $arm: rc=$rc $(tail -1 "$RUN/take.log" | cut -c1-200)"
  nl_say "  session $arm: $(cat "$td/rec/rc" 2>/dev/null) $(cat "$td/rec/result.json" 2>/dev/null)"
  [ "$rc" = 0 ] || { nl_say "  session $arm: harness rc $rc (see $RUN/take.log)"; SHORT=1; }; }

run_arm() { # <arm>
  local arm=$1 UP
  if rows_complete "$arm" && take_done "$arm-1"; then nl_say "arm $arm: rows and session already done (resume)"; nl_mark "${arm}_rows" done "resume: skipped"; nl_mark "${arm}_session" done "resume: skipped"; return 0; fi
  [ "$arm" = qsa ] && [ -n "$STUB_Q" ] && UP=$STUB_Q; [ "$arm" = stock ] && [ -n "$STUB_S" ] && UP=$STUB_S
  nl_mark "${arm}_rows" running
  if [ -z "${UP:-}" ]; then
    export LD_LIBRARY_PATH=$([ "$arm" = qsa ] && echo $QB || echo $SB)
    serve "$arm" || { stop_srv; fail "${arm}_rows" "$arm server did not come up (see $RUN/server-$arm.log)"; }
    UP=http://127.0.0.1:$SPORT
  fi
  rows_arm "$arm" "${UP##*:}"
  curl -s -m 5 "$UP/slots" > "$RUN/rows/$arm/slots-end.json" 2>/dev/null
  nl_mark "${arm}_rows" done "rows"
  nl_mark "${arm}_session" running
  run_session "$arm" "$UP"
  stop_srv; [ -n "${STUB_Q}${STUB_S}" ] || wait_mem_free || fail "${arm}_session" "card not clear after the $arm stop"
  nl_mark "${arm}_session" done "session"; }

if [ -z "$STUB_Q$STUB_S" ]; then
  FOREIGN=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ' | tr '\n' ' ')
  nl_say "  card owners at start: ${FOREIGN:-none}; tmux: $(tmux ls -F '#{session_name}' 2>/dev/null | tr '\n' ' ')"
  [ -z "$FOREIGN" ] || fail qsa_rows "a process other than Donald holds the card: $FOREIGN"
  donald_stop || fail qsa_rows "card not free"
fi
want_arm qsa && run_arm qsa
want_arm stock && run_arm stock
if [ -z "$STUB_Q$STUB_S" ]; then donald_restore; fi
nl_say "card released after $(( ($(date +%s)-t0)/60 )) min (start to restore), the rest is CPU"

# ---- recounts of both sessions' HUD series against the engine's own numbers (CPU) ----
for td in "$RUN"/takes/*; do
  case "$td" in *.partial-*) continue;; esac
  [ -s "$td/hud/hud.jsonl" ] || continue
  python3 "$H/c94_recount.py" "$td" --json "$RUN/out/recount-$(basename "$td").json" > /dev/null 2>&1
  nl_say "  recount $(basename "$td"): $(python3 -c "import json;d=json.load(open('$RUN/out/recount-$(basename "$td").json'));print('recount',d['new_series_final'],'engine usage',d['exact_sum_predicted_n'],'hud as recorded',d['recorded_hud_final'],'(%s%%)'%d['hud_vs_recount_pct'],'HUD_OK' if d['hud_ok'] else 'HUD_OFF','recount_vs_engine','OK' if d['pass'] else 'OFF')" 2>&1 | cut -c1-200)"
done
nl_mark render running
"$GPY" "$H/c94_summary.py" "$RUN" "$SLUG" "$REGIME" > "$RUN/summary.pre.json" 2>>"$RUN/render.log"
VERD=$("$GPY" -c "import json;d=json.load(open('$RUN/summary.pre.json'));print(d['verdict'], '(provisional)' if d['provisional'] else '')" 2>/dev/null)
nl_say "  verdict (pre-registered rule): ${VERD:-unreadable}"
"$GPY" -c "
import json;d=json.load(open('$RUN/summary.pre.json'))
for a in ('qsa','stock'):
    x=(d['arms'].get(a) or {}).get('depths',{})
    print('   ',a,{k:(v.get('decode_tok_s'),v.get('prefill_tok_s')) for k,v in x.items()})
print('    R255',d.get('r255'),'claim',d.get('claim_check'),'sessions',{k:(v.get('score') or {}).get('correct') for k,v in d.get('sessions',{}).items()})" 2>/dev/null | while read -r l; do nl_say "$l"; done
rr=1
LT=$RUN/takes/qsa-1; RT=$RUN/takes/stock-1
if [ -s "$LT/hud/hud.jsonl" ] && [ -s "$RT/hud/hud.jsonl" ] && [ -s "$RUN/rows/qsa/series-d255000.series" ] && [ -s "$RUN/rows/stock/series-d255000.series" ]; then
  TTL="QSA against the dense build at 255K tokens of context, one RTX 5090"
  SUB="Qwen3.8-27B, same quantiser, drafter and flags. left: QSA fork of llama.cpp. right: stock llama.cpp 18ab62a. runs were back to back, clocks synced."
  SUBC="Qwen3.8-27B, same quantiser (GSQ-RCO IQ3_S), same DFlash2 drafter, same flags. top: QSA fork of llama.cpp. bottom: stock llama.cpp 18ab62a."
  "$RPY" "$H/c94_render.py" chew "$RUN/out/chew.mp4" --qsa-series "$RUN/rows/qsa/series-d255000.series" --stock-series "$RUN/rows/stock/series-d255000.series" --ffmpeg "$FF" --chew-seconds 20 \
     --title "$TTL" --subtitle "$SUBC" --left-label "QSA fork  ·  IQ3_S + DFlash2" --right-label "stock llama.cpp  ·  dense IQ3_S + DFlash2" --footer "same 255K-token prompt (Hermes docs text), cold, code task. counters from the server's own timings." >> "$RUN/render.log" 2>&1
  SPK=$("$GPY" -c "import json;d=json.load(open('$RUN/summary.pre.json'));a=d['arms'];m=max([((a.get(k) or {}).get('depths',{}).get('0') or {}).get('decode_tok_s') or 0 for k in ('qsa','stock')]+[50]);print(int(-(-m*1.3//50)*50))" 2>/dev/null); SPK=${SPK:-250}
  "$RPY" "$H/c94_render.py" race "$RUN/out/race.mp4" --left "$LT" --right "$RT" --ffmpeg "$FF" --stills "$RUN/out/stills" --spark-max "$SPK" --real-max-s 40 --open-s 10 --close-s 10 --target-main-s 22 --cut-after-first 6 --pre-s 2 --hold-s 1.5 \
     --title "Hermes Agent, 240K tokens of docs in one prompt, 24 questions, fully local" --subtitle "$SUB" --left-label "QSA fork  ·  12.3 GB" --right-label "stock llama.cpp  ·  12.1 GB" --ctx-total $CTXCFG \
     --footer "24 'default of X' questions with known answers, docs as prefill, no file tools. counters from the server's own stream." >> "$RUN/render.log" 2>&1; rr=$?
  if [ "$rr" = 0 ]; then
    "$GPY" "$H/c94_summary.py" "$RUN" "$SLUG" "$REGIME" > "$RUN/summary.json.tmp" 2>>"$RUN/render.log" && mv "$RUN/summary.json.tmp" "$RUN/summary.card.json"
    "$RPY" "$H/c94_facts.py" "$RUN/summary.card.json" "$RUN/facts.json" >> "$RUN/render.log" 2>&1 && \
    "$RPY" "$H/c94_render.py" card "$RUN/out/card.mp4" --facts "$RUN/facts.json" --ffmpeg "$FF" --seconds 8 >> "$RUN/render.log" 2>&1 && \
    "$FF" -nostdin -hide_banner -loglevel error -y -i "$RUN/out/card.mp4" -i "$RUN/out/chew.mp4" -i "$RUN/out/race.mp4" \
      -filter_complex "[0:v]setsar=1,fps=25[a];[1:v]setsar=1,fps=25[b];[2:v]setsar=1,fps=25[c];[a][b][c]concat=n=3:v=1:a=0[v]" -map "[v]" -c:v libx264 -crf 20 -pix_fmt yuv420p -movflags +faststart "$RUN/out/qsa-final.mp4" </dev/null >> "$RUN/render.log" 2>&1
    rr=$?
  fi
else nl_say "  missing a session or a 255K series: no clip (the table and verdict still stand)"; fi
"$RPY" - "$RUN/out" "$RUN/out/contact.png" <<'PYEOF' 2>>"$RUN/render.log"
import glob, os, sys
from PIL import Image
out = sys.argv[1]
fs = sorted(glob.glob(out + "/stills/*.png"))[:7]
fs += [p for p in (out + "/card.mp4.png", out + "/chew.mp4.mid.png") if os.path.exists(p)]
if fs:
    w, h = 640, 360
    sheet = Image.new("RGB", (w * 3, h * 3), "#000")
    for i, f in enumerate(fs[:9]):
        sheet.paste(Image.open(f).convert("RGB").resize((w, h)), ((i % 3) * w, (i // 3) * h))
    sheet.save(sys.argv[2])
PYEOF
nl_say "  render rc=$rr; final clip: $([ -f "$RUN/out/qsa-final.mp4" ] && du -h "$RUN/out/qsa-final.mp4" | cut -f1 || echo missing)"
if [ "$SHORT" = 1 ] || [[ "$VERD" == *provisional* ]] || [ -n "$ONLY" ]; then
  nl_say "  a leg could not finish inside the window or only one arm ran (verdict ${VERD:-?}): exit 3, resumable, summary.json NOT written (PROVISIONAL: $RUN/summary.pre.json)"
  nl_mark render failed "incomplete, resumable"
  exit 3
fi
"$GPY" "$H/c94_summary.py" "$RUN" "$SLUG" "$REGIME" > "$ART" 2>>"$RUN/render.log"
nl_judge_artifact render "$rr" "$ART" "$t0" "$(( ($(date +%s)-t0)/60 )) min, render rc $rr" || { nl_say "FAIL: no fresh summary.json or render rc $rr"; exit 1; }
nl_say "  summary: verdict ${VERD}; line 1: $("$GPY" -c "import json;d=json.load(open('$ART'));print(d.get('line1'))")"
nl_say "DONE $SLUG"
exit 0
