#!/bin/bash
# 73-agenttest-v2-c1.sh: idle-queue item, STAGED 2026-10-28 by mercury (subagent, GPU thread). Lane L1: "Hermes Agent on 8 / 12 / 16 / 24 / 32 GB: the measured pick per VRAM tier", first pass = C1, on-disk models only. Sorts after the 71/72 voxel items and the 09z duck items.
# EV: private (projected post angle and reach bracket live in the ops vault status note)
# serves: content (AI-at-home piece #1: "which model to run Hermes Agent on, per VRAM tier", a per-tier card + a filmed side-by-side of two tiers on the same task) + hardening (agenttest v2: a task set that spreads models, checkers proven before any output)
# What it measures: real Hermes Agent sessions (pinned capsule install, see HERMES_PIN) against a solo-card llama-server, one row at a time, on 30 frozen offline tasks (agenttest/v2: files, code, refactor, recovery, shell, data, long)
#   each with a deterministic checker proven before any model output. One fresh session per task, default approvals, memory off, no sampling sent by Hermes so the row's explicit llama-server flags (maker card values, rows.json) apply.
#   Per row: pass/30 (+ A/B/C difficulty split), tool calls, tool errors and approval blocks counted SEPARATELY, malformed calls, wall time, first-turn wait (cold), decode tok/s, measured peak VRAM at -c 65536 (f16 KV default, the
#   hermes-settings verdict: no setting moved S5 by more than 3 of 100, so the board recipe stays), then the tier from that peak. Legs, in run order (most decision-relevant first; a partial weekend still yields a usable answer):
#     1 spark-x26-4b-q4km 17 | 2 ornith15-9b-q4km 22 | 3 qwen38-27b-ud-q4km 45 | 4 qwen38-27b-ud-iq3xxs 40 | 5 ornith15-35b-a3b-q4km 17 | 6 gsq-rco-27b-iq2xs 40 | 7 gemma4-31b-qat-q4_0 50 | 8 qwen38-27b-q6k 48 | 9 ornith15-9b-q6k 24
#     10 ornith15-122b-ud-iq3xxs 65 (expert offload, --n-cpu-moe ladder 24 28 32) | 11 fit-q8: VRAM of every row with q8_0 K+V at -c 65536 (one 14k-token request per row, no tasks) ~15 min.  est minutes = my estimate, UNVERIFIED until row 1 reports.
# What it does NOT measure: accuracy with q8_0 KV on these tasks (only the fit leg's VRAM; the settings item measured accuracy on the toolcall slice only), more than one run per row, any row off the disk list (C2 = downloads, later).
# resumable: per task (<tid>.result.json) and per row (<slug>/summary.json = done; <slug>/load-failed.json = NOFIT, done). exit 3 while rows remain. hard stop = NL_HARD_STOP_EPOCH (the runner exports the block end); a task is only started with budget+120 s left.
# test knobs (all empty in production): AT2_SMOKE=cpu (CPU tiny-model smoke: no Donald drain, CUDA_VISIBLE_DEVICES empty, ngl 0), AT2_ROWS_JSON, AT2_ROWS (slugs), AT2_TASKS (task subset), AT2_RES, AT2_WORK, AT2_PORT, AT2_MANIFEST, AT2_RUNLOG, AT2_BUDGET_SCALE, AT2_CODE.
set -uo pipefail
NL_TAG=at2c1
MANIFEST=/home/witcheer/agenttest-v2-c1-status.json
export MANIFEST
. /home/witcheer/night-lib.sh
[ "${DRYRUN:-0}" = "1" ] && MANIFEST=/tmp/agenttest-v2-c1-dryrun.json   # a DRYRUN must never wipe the real manifest (the drain DRYRUNs before every block)
[ -n "${AT2_MANIFEST:-}" ] && MANIFEST=$AT2_MANIFEST
SMOKE=${AT2_SMOKE:-}
CODE=${AT2_CODE:-/home/witcheer/agenttest-v2/code}
RES=${AT2_RES:-/home/witcheer/agenttest-v2/results/c1}
WORK=${AT2_WORK:-/home/witcheer/agenttest-v2/work/c1}
RUNLOG=${AT2_RUNLOG:-/home/witcheer/agenttest-v2-c1-run.log}
PORT=${AT2_PORT:-8173}
ROWS_JSON=${AT2_ROWS_JSON:-$CODE/rows.json}
SRV=/home/witcheer/llama.cpp-bolt/build/bin/llama-server
BOLT_PIN=cd0a5b6a1                       # llama.cpp b11668, the build the Spark v1 run used and the only one with spark2_5
HERMES=/home/witcheer/.local/bin/hermes
HERMES_PIN=9bda188                       # Hermes Agent v0.22.2+2756.g9bda188 (2026.9.24) on capsule, recorded 2026-10-28; override the pin only with a decision (L1_HERMES_PIN)
HERMES_PIN=${L1_HERMES_PIN:-$HERMES_PIN}
MODELS=/home/witcheer/models
CTX=65536
LOADWAIT=${AT2_LOADWAIT:-900}
mkdir -p "$RES" "$WORK"

ROWSARGS=(); [ -n "${AT2_ROWS:-}" ] && ROWSARGS=(--only "$AT2_ROWS") ; ROWSARGS+=(--rows "$ROWS_JSON")
mapfile -t ROWS < <(python3 "$CODE/row_info.py" "${ROWSARGS[@]}")
art_of() { echo "$RES/$1/summary.json"; }
nofit_of() { echo "$RES/$1/load-failed.json"; }
done_row() { [ -f "$(art_of "$1")" ] || [ -f "$(nofit_of "$1")" ]; }
fitq8_of() { echo "$RES/fit-q8/$1.json"; }

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0}, smoke=${SMOKE:-none}, rows file $ROWS_JSON)"
[ "${#ROWS[@]}" -gt 0 ] || { nl_say "ABORT: no rows from row_info.py"; exit 2; }
# ---- harness integrity (frozen task set, proven checkers, parse) ----
nl_check_path "$CODE/run_agent.py" "$CODE/tasks.json" "$CODE/FROZEN.sha256" "$CODE/fit_probe.py" "$CODE/aggregate.py" "$CODE/parse_server_log.py" "$ROWS_JSON"
( cd "$CODE" && sha256sum -c --quiet FROZEN.sha256 ) && nl_ok "frozen task set sha256 intact (FROZEN.sha256)" || nl_fail "frozen task set changed after the freeze"
FZ=$(cd "$CODE" && AT2_STRICT=1 python3 freeze.py --check 2>&1 || true); case "$FZ" in *"matches the registry"*) nl_ok "tasks.json = registry + seed hashes";; *) nl_fail "tasks.json does not match the registry/seeds: $FZ";; esac
mkdir -p /home/witcheer/agenttest-v2
PROOF=$(cd "$CODE" && python3 prove_checkers.py --work /home/witcheer/agenttest-v2/at2-proof --out /tmp/at2-proof-capsule.json 2>&1 | tail -1 || true)
case "$PROOF" in "ALL_OK 30/30"*) nl_ok "checker proof on this box: $PROOF";; *) nl_fail "checker proof failed: $PROOF";; esac
python3 -m py_compile "$CODE/run_agent.py" "$CODE/aggregate.py" "$CODE/fit_probe.py" "$CODE/row_info.py" && nl_ok "harness scripts compile" || nl_fail "harness scripts do not compile"
# ---- binaries ----
nl_check_path "$SRV" "$HERMES"
case "$($SRV --version 2>&1)" in *"$BOLT_PIN"*) nl_ok "llama-server build $BOLT_PIN (b11668)";; *) nl_fail "llama-server build changed (want $BOLT_PIN)";; esac
HELP=$($SRV --help 2>&1 || true)
for want in "--n-cpu-moe" "--min-p" "--top-k" "--top-p" "--temp" "--jinja" "--flash-attn" "--cache-type-k" "--log-timestamps"; do case "$HELP" in *"$want"*) :;; *) nl_fail "llama-server lacks $want";; esac; done
nl_ok "llama-server lists the flags this item passes"
case "$($HERMES --version 2>&1)" in *"$HERMES_PIN"*) nl_ok "Hermes Agent pin $HERMES_PIN ($($HERMES --version 2>&1 | head -1))";; *) nl_fail "Hermes version changed (want $HERMES_PIN): $($HERMES --version 2>&1 | head -1)";; esac
HHELP=$($HERMES chat --help 2>&1 || true)
for want in "--format" "--source" "-q"; do case "$HHELP" in *"$want"*) :;; *) nl_fail "hermes chat lacks $want";; esac; done
nl_ok "hermes chat lists -q / --format / --source (budget = agent.run_budget_seconds in the profile config)"
LIBL=$(dirname "$SRV")/libllama.so
nl_check_disk 20
LSOF=$(ss -ltn 2>/dev/null | awk '{print $4}' | tr '\n' ' '); case " $LSOF " in *":$PORT "*) nl_fail "port $PORT already listening";; *) nl_ok "port $PORT free";; esac
# ---- rows ----
RUNS=0; EST=0; PH=""
for L in "${ROWS[@]}"; do IFS='|' read -r slug path bytes arch est temp top_p top_k min_p moe prof label <<< "$L"
  PH="$PH $slug"
  [ -f "$MODELS/$path" ] && [ "$(stat -c %s "$MODELS/$path")" = "$bytes" ] && nl_ok "$slug file $bytes B" || nl_fail "$slug: file missing or size differs from the pin ($bytes): $MODELS/$path"
  n=$(strings "$LIBL" 2>/dev/null | grep -ci -- "$arch" || true); [ "${n:-0}" -gt 0 ] && nl_ok "$slug arch $arch: $n libllama hits" || nl_fail "$slug: libllama lacks arch $arch"
  if done_row "$slug"; then nl_say "  $slug: SKIP, artifact present ($( [ -f "$(art_of "$slug")" ] && art_of "$slug" || nofit_of "$slug"))"
  else nl_say "  $slug: RUN  (est $est min; sampling temp $temp top_p $top_p top_k $top_k min_p $min_p; moe ladder $moe; profile $prof; checks $(art_of "$slug"))"; RUNS=$((RUNS+1)); EST=$((EST+est)); fi
done
FITQ=0; for L in "${ROWS[@]}"; do IFS='|' read -r slug _ <<< "$L"; [ -f "$(fitq8_of "$slug")" ] || FITQ=$((FITQ+1)); done
DL=$(nl_deadline_epoch); nl_say "  planned RUN: $RUNS rows (est $EST card min) + fit-q8 for $FITQ rows (~$((FITQ*2)) min); hard stop $(date -d "@$DL" '+%a %H:%M') local, $(( (DL-$(date +%s))/60 )) min away"
[ -z "${EXPECT_RUN:-}" ] || [ "$RUNS" = "$EXPECT_RUN" ] || { nl_say "ABORT: planned RUN $RUNS != EXPECT_RUN $EXPECT_RUN"; exit 2; }
[ "$RUNS" -gt 0 ] || [ "$FITQ" -gt 0 ] || { nl_say "nothing to run, item complete"; exit 0; }
if [ "${DRYRUN:-0}" = "1" ] || [ ! -f "$MANIFEST" ]; then nl_manifest_init $PH fit-q8 aggregate; fi
nl_gate

# =============================== real run ===============================
SP=""; GP=""; DRAINED=0
vram() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
wait_vram_free() { local i v; for i in $(seq 1 60); do v=$(vram); [ "${v:-99999}" -lt 1500 ] && return 0; sleep 2; done; return 1; }
stop_server() { # own pid only, never pkill: a sibling's llama-server must survive
  [ -n "$SP" ] && kill "$SP" 2>/dev/null; sleep 3; [ -n "$SP" ] && kill -0 "$SP" 2>/dev/null && { kill -9 "$SP" 2>/dev/null; sleep 2; }
  SP=""; [ -n "$SMOKE" ] || { wait_vram_free || nl_say "  WARN: VRAM still $(vram) MiB after server stop"; }; }
stop_sampler() { [ -n "$GP" ] && kill "$GP" 2>/dev/null; GP=""; }
restore() { stop_sampler; stop_server
  if [ "$DRAINED" = 1 ]; then nl_say "EXIT trap: restoring Donald"; systemctl is-active --quiet llama-server.service || sudo -n systemctl start llama-server.service || true
    for i in $(seq 1 30); do curl -s -m 3 http://127.0.0.1:8090/health | grep -qi ok && { nl_say "Donald proxy health OK"; break; }; sleep 2; done; fi; }
trap restore EXIT
if [ -z "$SMOKE" ]; then
  nl_say "draining Donald"; sudo -n systemctl stop llama-server.service; DRAINED=1; sleep 3
  for i in $(seq 1 120); do v=$(vram); [ "${v:-99999}" -lt 4000 ] && { nl_say "drain clear (${v}MiB)"; break; }; sleep 2; [ "$i" -eq 120 ] && { nl_say "FATAL drain timeout (a sibling holds the card?)"; exit 1; }; done
fi
DEADLINE=$(nl_deadline_epoch)
left_s() { echo $(( DEADLINE - $(date +%s) )); }

row_fail() { # <slug> <reason>: a persistent failure (not an OOM, not the window) twice in a row becomes NOFIT so the queue does not retry it every block
  local d="$RES/$1" n; n=$(( $(cat "$d/fails" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$d/fails"
  nl_say "  $1: FAILED ($2), failure $n of 2"; nl_mark "$1" failed "$2"
  if [ "$n" -ge 2 ]; then printf '{"label":"%s","load_failed":true,"reason":"%s twice in a row"}\n' "$1" "$2" > "$(nofit_of "$1")"; nl_say "  $1: recorded as NOFIT after two failures"; return 0; fi
  return 3; }
start_server() { # <path> <temp> <top_p> <top_k> <min_p> <moe> <log> [kvflags...] -> 0 up, 1 failed, 4 OOM signature. sets MOE_USED
  local path=$1 temp=$2 tp=$3 tk=$4 mp=$5 moe=$6 log=$7; shift 7
  local m ladder; ladder=$moe; [ "$moe" = "-" ] && ladder="-"
  for m in $ladder; do
    local extra=(); [ "$m" != "-" ] && extra=(--n-cpu-moe "$m")
    local gpu=(-ngl 99); local envp=()
    [ -n "$SMOKE" ] && { gpu=(-ngl 0 --threads 8); envp=(env CUDA_VISIBLE_DEVICES=); }
    "${envp[@]}" "$SRV" -m "$MODELS/$path" "${gpu[@]}" -fa on -c $CTX --jinja --host 127.0.0.1 --port "$PORT" --log-timestamps \
       --temp "$temp" --top-p "$tp" --top-k "$tk" --min-p "$mp" "${extra[@]}" "$@" >> "$log" 2>&1 &
    SP=$!
    local i; for i in $(seq 1 $((LOADWAIT/2))); do curl -s -m 3 http://127.0.0.1:$PORT/health | grep -qi ok && break; kill -0 $SP 2>/dev/null || break; sleep 2; done
    if curl -s -m 3 http://127.0.0.1:$PORT/health | grep -qi ok; then MOE_USED=$m; return 0; fi
    local oom=1; tail -300 "$log" | grep -qiE "out of memory|failed to allocate|cudaMalloc failed|ggml_backend_cuda_buffer_type_alloc_buffer|unable to allocate" && oom=4
    stop_server
    [ "$oom" = 4 ] || return 1
    nl_say "  load failed with an out-of-memory signature at n-cpu-moe=$m"
  done
  return 4
}
effort_probe() { # echoes unset | high | fail
  local c
  c=$(curl -s -o /tmp/at2-probe.json -w '%{http_code}' -m 180 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"x","messages":[{"role":"user","content":"Say ok."}],"reasoning_effort":"medium","max_tokens":24}')
  if [ "$c" = 200 ]; then echo unset; return; fi
  c=$(curl -s -o /tmp/at2-probe.json -w '%{http_code}' -m 180 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"x","messages":[{"role":"user","content":"Say ok."}],"reasoning_effort":"high","max_tokens":24}')
  [ "$c" = 200 ] && echo high || echo fail
}

run_row() { # <line>
  IFS='|' read -r slug path bytes arch est temp top_p top_k min_p moe prof label <<< "$1"
  local d="$RES/$slug" t0 left rc attempt effort MOE_USED="-"
  left=$(left_s)
  [ "$left" -gt 900 ] || { nl_say "window exhausted before $slug ($left s left), resumable"; nl_mark "$slug" failed "window exhausted (resumable)"; return 3; }
  mkdir -p "$d" "$WORK/$slug"; t0=$(date +%s); nl_mark "$slug" running; nl_say "=== $slug ($label, est $est min, budget $((left/60)) min) ==="
  start_server "$path" "$temp" "$top_p" "$top_k" "$min_p" "$moe" "$d/server.log"; rc=$?
  if [ "$rc" = 4 ]; then
    printf '{"label":"%s","load_failed":true,"reason":"out-of-memory signature in the server log at every n-cpu-moe tried (%s), ctx %s f16 KV","vram_mib_now":%s}\n' "$slug" "$moe" "$CTX" "$([ -n "$SMOKE" ] && echo 0 || vram)" > "$(nofit_of "$slug")"
    nl_say "  $slug: DOES NOT FIT at ctx $CTX (out of memory at load), recorded as NOFIT"; nl_mark "$slug" skipped "NOFIT"; return 0
  elif [ "$rc" != 0 ]; then row_fail "$slug" "server did not come up (not an OOM signature)"; return $?; fi
  local vl=0; [ -n "$SMOKE" ] || vl=$(vram)
  nl_say "  server up (n-cpu-moe $MOE_USED), VRAM $vl MiB"
  curl -s -m 10 http://127.0.0.1:$PORT/props > "$d/props.json" 2>/dev/null
  effort=$(effort_probe); nl_say "  reasoning_effort probe: $effort (unset = Hermes' own default 'medium' is accepted; high = the template rejects medium, the one disclosed deviation)"
  if [ "$effort" = fail ]; then stop_server; row_fail "$slug" "both reasoning_effort medium and high rejected"; return $?; fi
  python3 - "$d/props.json" "$d/row-meta.json" "$slug" "$label" "$path" "$bytes" "$temp" "$top_p" "$top_k" "$min_p" "$MOE_USED" "$effort" "$vl" "$HERMES_PIN" "$($SRV --version 2>&1 | grep -i version | head -1)" "$($HERMES --version 2>&1 | head -1)" <<'PY'
import json, sys
props, out, slug, label, path, nbytes, temp, tp, tk, mp, moe, effort, vl, pin, build, hv = sys.argv[1:17]
p = json.load(open(props)) if props else {}
g = (p.get("default_generation_settings") or {}).get("params") or {}
applied = {k: g.get(k) for k in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repeat_penalty", "n_predict")}
want = {"temperature": float(temp), "top_p": float(tp), "top_k": int(tk), "min_p": float(mp)}
ok = all(applied.get(k) is not None and abs(float(applied[k]) - v) < 1e-5 for k, v in want.items() if not (k == "top_k" and int(tk) <= 0))
json.dump({"row": slug, "label": label, "path": path, "bytes": int(nbytes), "ctx": p.get("default_generation_settings", {}).get("n_ctx"), "total_slots": p.get("total_slots"),
           "sampling_flags": want, "sampling_applied": applied, "sampling_match": ok, "n_cpu_moe_used": None if moe == "-" else int(moe), "effort": effort, "vram_loaded_mib": int(vl),
           "llama_build": build, "hermes_version": hv, "hermes_pin": pin, "template_mentions_reasoning_effort": "reasoning_effort" in (p.get("chat_template") or "")}, open(out, "w"), indent=1)
print("row-meta: sampling flags vs /props applied:", "MATCH" if ok else "MISMATCH", applied)
sys.exit(0 if ok else 5)
PY
  rc=$?
  if [ "$rc" != 0 ]; then stop_server; row_fail "$slug" "sampling flags do not match what /props reports"; return $?; fi
  [ -n "$SMOKE" ] || { nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits -l 2 >> "$d/gpu-samples.csv" 2>/dev/null & GP=$!; }
  local tasksarg=(); [ -n "${AT2_TASKS:-}" ] && tasksarg=(--tasks "$AT2_TASKS")
  local scale=(); [ -n "${AT2_BUDGET_SCALE:-}" ] && scale=(--budget-scale "$AT2_BUDGET_SCALE")
  for attempt in 1 2 3; do
    python3 -u "$CODE/run_agent.py" --row "$slug" --profile "$prof" --model-id "$slug" --port "$PORT" --work-root "$WORK/$slug" --out "$d" --effort "$effort" --deadline-epoch "$DEADLINE" \
       --hermes "$HERMES" --ctx $CTX --meta "$d/row-meta.json" --gpu-samples "$d/gpu-samples.csv" "${tasksarg[@]}" "${scale[@]}" >> "$RUNLOG" 2>&1
    rc=$?
    [ "$rc" = 4 ] || break
    nl_say "  $slug: server unhealthy mid-row (attempt $attempt), restarting it"; stop_server
    start_server "$path" "$temp" "$top_p" "$top_k" "$min_p" "$MOE_USED" "$d/server.log"; [ $? = 0 ] || { nl_say "  restart failed"; break; }
  done
  stop_sampler
  SKIP_FIRST=1 python3 "$CODE/parse_server_log.py" "$d/server.log" > "$d/server-timings.json" 2>/dev/null || true
  stop_server
  if [ "$rc" = 3 ]; then nl_say "  $slug: partial, resumable ($(ls "$d"/t*.result.json 2>/dev/null | wc -l) tasks done)"; nl_mark "$slug" failed "partial (resumable)"; return 3; fi
  if [ -n "${AT2_TASKS:-}" ]; then nl_say "  $slug: smoke subset finished rc=$rc: $(grep -a 'SMOKE-SUBSET' "$RUNLOG" | tail -1)"; nl_mark "$slug" done "smoke subset"; return 0; fi
  if nl_judge_artifact "$slug" "$rc" "$(art_of "$slug")" "$t0" "$(( ($(date +%s)-t0)/60 )) min"; then
    nl_say "  $(grep -a 'ROW-SUMMARY' "$RUNLOG" | tail -1)"; return 0
  else nl_say "  $slug: FAILED rc=$rc"; return 3; fi
}

fit_q8() { # leg 11: VRAM with q8_0 K+V at the same ctx, one ~14k-token request per row, no tasks
  local L slug path bytes arch est temp top_p top_k min_p moe prof label d out rc MOE_USED=- n=0 todo=0
  mkdir -p "$RES/fit-q8"
  for L in "${ROWS[@]}"; do IFS='|' read -r slug path bytes arch est temp top_p top_k min_p moe prof label <<< "$L"
    out=$(fitq8_of "$slug"); [ -f "$out" ] && { n=$((n+1)); continue; }
    done_row "$slug" && [ -f "$(nofit_of "$slug")" ] && { n=$((n+1)); continue; }
    todo=1
    [ "$(left_s)" -gt 600 ] || { nl_say "  fit-q8: window exhausted before $slug, resumable"; return 3; }
    nl_say "  fit-q8 $slug"
    start_server "$path" "$temp" "$top_p" "$top_k" "$min_p" "$moe" "$RES/fit-q8/$slug.server.log" --cache-type-k q8_0 --cache-type-v q8_0; rc=$?
    if [ "$rc" = 4 ]; then printf '{"label":"%s","load_failed":true,"kv":"q8_0","ctx":%s}\n' "$slug" "$CTX" > "$out"; nl_say "  fit-q8 $slug: does not fit even with q8_0 KV"; n=$((n+1)); continue
    elif [ "$rc" != 0 ]; then nl_say "  fit-q8 $slug: server did not come up (resumable)"; continue; fi
    local nog=(); [ -n "$SMOKE" ] && nog=(--no-gpu)
    timeout 900 python3 "$CODE/fit_probe.py" --port "$PORT" --out "$out" --label "$slug-q8" "${nog[@]}" >> "$RUNLOG" 2>&1 || rm -f "$out"
    stop_server
    [ -f "$out" ] && { nl_say "  fit-q8 $slug: $(python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print('peak', r.get('vram_peak_mib'), 'MiB, prompt', r.get('prompt_tokens'), 'tokens, http', r.get('http'))" "$out")"; n=$((n+1)); }
  done
  [ "$n" = "${#ROWS[@]}" ] && return 0 || return 3
}

for L in "${ROWS[@]}"; do IFS='|' read -r slug _ <<< "$L"
  if done_row "$slug"; then nl_mark "$slug" skipped "artifact present"; continue; fi
  run_row "$L"; rrc=$?
  [ "$rrc" = 3 ] && { [ "$(left_s)" -gt 900 ] || break; }   # a failed row with time left moves on to the next; a closed window ends the item
done
if [ -z "${AT2_TASKS:-}" ]; then
  nl_mark fit-q8 running; fit_q8 && nl_mark fit-q8 done "$(ls "$RES"/fit-q8/*.json 2>/dev/null | wc -l) rows" || nl_mark fit-q8 failed "partial (resumable)"
fi
python3 "$CODE/aggregate.py" --res "$RES" --out "$RES" --rows "$ROWS_JSON" > "$RES/aggregate.log" 2>&1 && nl_mark aggregate done "$RES/l1-readout.md" || nl_mark aggregate failed "see $RES/aggregate.log"
nl_say "readout: $RES/l1-readout.md ($(grep -c '^| ' "$RES/l1-readout.md" 2>/dev/null) table lines)"
for L in "${ROWS[@]}"; do IFS='|' read -r slug _ <<< "$L"; done_row "$slug" || { nl_say "rows remain, resumable"; exit 3; }; done
[ -n "${AT2_TASKS:-}" ] && { nl_say "smoke subset done"; exit 0; }
[ -n "$(ls "$RES"/fit-q8/*.json 2>/dev/null | head -1)" ] && [ "$(ls "$RES"/fit-q8/*.json | wc -l)" -ge "${#ROWS[@]}" ] || { nl_say "fit-q8 remains, resumable"; exit 3; }
nl_say "all rows and the fit-q8 leg done"; exit 0
