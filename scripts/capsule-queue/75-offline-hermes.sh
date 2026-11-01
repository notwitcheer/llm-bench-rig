#!/bin/bash
# 75-offline-hermes.sh: idle-queue item (lane L3, offline Hermes Agent). Public copy: the EV line and the provenance quote stay in the private vault.
# EV: private (kept in the operator's vault).
# serves: content (AI-at-home piece, rule 4, raw-video lane candidate) + a reproducible recipe (one script) + hardening of what "fully local" means (agent isolated, model on the host over a unix socket).
# What it measures: real Hermes Agent (official image nousresearch/hermes-agent:v0.22.4, 2026.9.24, upstream 4d043038, pinned by image id) running INSIDE `docker run --network none`, model = Qwen3.8-27B UD-Q4_K_M on this card
#   (llama-server b11668 cd0a5b6a1 on its OWN port, never :8090) reached through a unix socket (host relay + in-container relay, stdlib python: capsule has no socat), 10 frozen agenttest v1 tasks (prompts verbatim, one fresh
#   Hermes session each, v1 checkers run in a separate --network none container). Proof legs recorded in the run dir isolation/: L1 docker inspect NetworkMode none, L2 failing curl/DNS/socket probes inside, L3 /sys/class/net counters
#   inside (zero non-loopback interfaces and packets), C0 control (same flags, no bridge: every counter zero), C1 positive control (same curl on the default network succeeds). The proof runs BEFORE the card is touched; a FAIL aborts (rc 5).
# Scoring rules fixed before any model output: t06 (fetch a web page) is NOT passed offline whatever the v1 checker says (a model can write "Hermes Agent" from memory, which the checker accepts); its transcript is classified
#   (honest_no_fetch / silent_guess_from_memory / claimed_a_fetch_that_did_not_happen / gave_up_silently, regex heuristic, read t06.jsonl before quoting). line 1 counts passes of 10; the readout also gives passes of the 9 offline-capable tasks.
# What it does NOT measure: more than one sampled run (n=10, our own tasks, v1 is saturated online by Spark-X2.6-4B and Muse Glimmer, both 10/10), speed (card shared with nothing but the model is solo; decode tok/s from the server log is a side number),
#   any other model, "offline machine" (the model server is on the host and the host has a network: only the AGENT is cut off).
# sizing (UNVERIFIED until the first real run): basis = the same model and flags in agenttest v2 (row qwen38-27b-ud-q4km, 2026-10-29): easy-group median 13 s per task (max 32.5 s), mid-group median 36 s (max 179 s), decode p50 75 tok/s, server up 4 s after the drain (file cached),
#   cold first turn 6.6 s. The 10 v1 tasks are easy/mid, so ~10 x 15-40 s = 3-7 min + ~1 min load + ~1 min proof/readout = ~7-12 card min, expect ~10, plus up to 15 s Hermes start-up per task. Hard cap: no task starts that cannot finish inside OFF_CARDCAP min (26) of the drain (a task needs budget 300 s + 90 s);
#   the never-expected worst case is bounded by that cap, the item then stops resumable (exit 3).
# tail-filler rule: refuses to start with under OFF_MINMIN (35) min left in the block (logs it, exit 3, stays queued): the drain's own floor is 30 min; this item fits block tails under the 90-min voxel floor.
# resumable: per task (<tid>.result.json in the run dir); exit 3 while tasks remain; exit 0 when 10 results + summary.json exist.
# test knobs (all empty in production): OFF_SMOKE=mock (scripted mock model, no GPU, no Donald drain) | cpu (tiny 0.5B model, -ngl 0, CUDA_VISIBLE_DEVICES empty), OFF_RUN, OFF_PORT, OFF_TASKS, OFF_BUDGET, OFF_MANIFEST, OFF_RUNLOG, OFF_CODE, OFF_MINMIN, OFF_CARDCAP, OFF_NOPOS=1 (skip the positive control).
set -uo pipefail
NL_TAG=off75
MANIFEST=/home/witcheer/offline-hermes-status.json
export MANIFEST
. /home/witcheer/night-lib.sh
[ "${DRYRUN:-0}" = "1" ] && MANIFEST=/tmp/offline-hermes-dryrun.json   # a DRYRUN must never wipe the real manifest (the drain DRYRUNs before every block)
[ -n "${OFF_MANIFEST:-}" ] && MANIFEST=$OFF_MANIFEST
SMOKE=${OFF_SMOKE:-}
CODE=${OFF_CODE:-/home/witcheer/offline-hermes/code}
RUN=${OFF_RUN:-/home/witcheer/offline-hermes/results/main}
RUNLOG=${OFF_RUNLOG:-/home/witcheer/offline-hermes-run.log}
PORT=${OFF_PORT:-8178}
SRV=/home/witcheer/llama.cpp-bolt/build/bin/llama-server
BOLT_PIN=cd0a5b6a1                       # llama.cpp b11668: the build the agenttest v2 L1 rows ran on
MODELS=/home/witcheer/models
MODEL_REL=qwen38-27b/Qwen3.8-27B-UD-Q4_K_M.gguf
MODEL_BYTES=16464440224
MODEL_ID=qwen38-27b-ud-q4km
TEMP=1.0; TOP_P=0.95; TOP_K=20; MIN_P=0.0   # Qwen3.8 maker card thinking-mode values (agenttest v2 rows.json, read 2026-10-28)
CTX=65536
IMAGE=nousresearch/hermes-agent:v0.22.4
IMAGE_ID=sha256:7a69201726bc64db2072a1371557137c23312d29705cda2b8ea2a0214a50a4cf
HERMES_WANT="Hermes Agent v0.22.4 (2026.9.24)"
MINMIN=${OFF_MINMIN:-35}
CARDCAP=${OFF_CARDCAP:-26}
BUDGET=${OFF_BUDGET:-300}
LOADWAIT=${OFF_LOADWAIT:-900}
OFFPY=$CODE/offline/offline_run.py
mkdir -p "$RUN"

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0}, smoke=${SMOKE:-none}, run dir $RUN)"
# ---- harness integrity ----
nl_check_path "$OFFPY" "$CODE/offline/isolation.py" "$CODE/offline/bridge_host.py" "$CODE/offline/bridge_ctr.py" "$CODE/offline/mock_offline.py" "$CODE/offline/seedlib.py" "$CODE/tasks.json" "$CODE/v2/run_agent.py" "$CODE/offline/FROZEN.sha256"
( cd "$CODE" && sha256sum -c --quiet offline/FROZEN.sha256 ) && nl_ok "frozen files intact (v1 tasks.json, checkers, seed_data, gen_seeds, offline/*.py: offline/FROZEN.sha256)" || nl_fail "a frozen file changed after the freeze"
python3 -m py_compile "$CODE"/offline/*.py && nl_ok "offline harness scripts compile" || nl_fail "offline harness scripts do not compile"
# ---- container image ----
command -v docker >/dev/null && nl_ok "docker present, user in the docker group: $(docker --version)" || nl_fail "docker missing"
[ "$(docker image inspect "$IMAGE" --format '{{.Id}}' 2>/dev/null)" = "$IMAGE_ID" ] && nl_ok "image $IMAGE id pinned ${IMAGE_ID:0:19}" || nl_fail "image $IMAGE missing or its id changed (want $IMAGE_ID)"
HV=$(docker run --rm --network none --entrypoint hermes "$IMAGE" --version 2>&1 | head -1)
case "$HV" in *"$HERMES_WANT"*) nl_ok "Hermes inside the image: $HV";; *) nl_fail "Hermes version in the image changed (want $HERMES_WANT): $HV";; esac
HH=$(docker run --rm --network none --entrypoint hermes "$IMAGE" chat --help 2>&1 || true)
for want in "--format" "--source" "-q"; do case "$HH" in *"$want"*) :;; *) nl_fail "hermes chat lacks $want";; esac; done
nl_ok "hermes chat lists -q / --format / --source (budget = agent.run_budget_seconds in the home config)"
# ---- model server (own port, never :8090) ----
if [ -z "$SMOKE" ] || [ "$SMOKE" = cpu ]; then
  nl_check_path "$SRV"
  case "$($SRV --version 2>&1)" in *"$BOLT_PIN"*) nl_ok "llama-server build $BOLT_PIN (b11668)";; *) nl_fail "llama-server build changed (want $BOLT_PIN)";; esac
  HELP=$($SRV --help 2>&1 || true)
  for want in "--min-p" "--top-k" "--top-p" "--temp" "--jinja" "--flash-attn" "--log-timestamps"; do case "$HELP" in *"$want"*) :;; *) nl_fail "llama-server lacks $want";; esac; done
  nl_ok "llama-server lists the flags this item passes"
fi
if [ -z "$SMOKE" ]; then
  [ -f "$MODELS/$MODEL_REL" ] && [ "$(stat -c %s "$MODELS/$MODEL_REL")" = "$MODEL_BYTES" ] && nl_ok "model file $MODEL_BYTES B" || nl_fail "model file missing or size differs from the pin ($MODEL_BYTES): $MODELS/$MODEL_REL"
  LIBL=$(dirname "$SRV")/libllama.so
  n=$(strings "$LIBL" 2>/dev/null | grep -ci -- "qwen35" || true); [ "${n:-0}" -gt 0 ] && nl_ok "libllama arch qwen35: $n hits" || nl_fail "libllama lacks arch qwen35"
fi
[ "$PORT" != 8090 ] || nl_fail "port 8090 is Donald, never use it"
LSOF=$(ss -ltn 2>/dev/null | awk '{print $4}' | tr '\n' ' '); case " $LSOF " in *":$PORT "*) nl_fail "port $PORT already listening";; *) nl_ok "port $PORT free";; esac
nl_check_disk 20
# ---- isolation proof as a preflight: containers only, no GPU, no model server needed ----
PPOS=(); { [ "${DRYRUN:-0}" = "1" ] || [ -n "${OFF_NOPOS:-}" ]; } && PPOS=(--no-positive-control)
PR=$(python3 "$OFFPY" proof --image "$IMAGE" --run "/tmp/off75-proof" --port "$PORT" "${PPOS[@]}" 2>&1 | grep -a "PROOF-RESULT\|ISOLATION PROOF" | tr '\n' ' ')
case "$PR" in *"PROOF-RESULT PASS"*) nl_ok "isolation proof (NetworkMode none, failing probes, zero counters, control): ${PR:0:200}";; *) nl_fail "isolation proof FAILED: $PR";; esac
EQ=$(python3 "$OFFPY" equivalence --image "$IMAGE" --run "/tmp/off75-eq" 2>&1 | tail -1)
case "$EQ" in *"SAME"*) nl_ok "seedlib.seed == gen_seeds.build (real build() in a --network none container)";; *) nl_fail "seed equivalence: $EQ";; esac
# ---- plan ----
TASKS_TODO=0; PH=""
for i in 01 02 03 04 05 06 07 08 09 10; do
  PH="$PH t$i"
  if nl_skip_exact "$RUN/t$i.result.json"; then nl_say "  t$i: SKIP, result present ($RUN/t$i.result.json)"; else nl_say "  t$i: RUN (checks $RUN/t$i.result.json)"; TASKS_TODO=$((TASKS_TODO+1)); fi
done
DL=$(nl_deadline_epoch); LEFTM=$(( (DL-$(date +%s))/60 ))
nl_say "  planned RUN: $TASKS_TODO tasks (est 7-12 card min incl. load, hard cap $CARDCAP card min); hard stop $(date -d "@$DL" '+%a %H:%M') local, $LEFTM min away; min-minutes rule: needs >= $MINMIN min left ($([ "$LEFTM" -ge "$MINMIN" ] && echo "would start now" || echo "would REFUSE now with exit 3, tail too short"))"
[ -z "${EXPECT_RUN:-}" ] || [ "$TASKS_TODO" = "$EXPECT_RUN" ] || { nl_say "ABORT: planned RUN $TASKS_TODO != EXPECT_RUN $EXPECT_RUN"; exit 2; }
[ "$TASKS_TODO" -gt 0 ] || { [ -f "$RUN/summary.json" ] && { nl_say "nothing to run, item complete"; exit 0; }; }
if [ "${DRYRUN:-0}" = "1" ] || [ ! -f "$MANIFEST" ]; then nl_manifest_init isolation-pre model $PH isolation-post readout; fi
nl_gate

# =============================== real run ===============================
[ "$LEFTM" -ge "$MINMIN" ] || { nl_say "under $MINMIN min left ($LEFTM min): refusing to start, resumable (exit 3)"; exit 3; }
SP=""; GP=""; GP2=""; DRAINED=0; UP=0
vram() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
wait_vram_free() { local i v; for i in $(seq 1 60); do v=$(vram); [ "${v:-99999}" -lt 1500 ] && return 0; sleep 2; done; return 1; }
stop_server() { # own pid only, never pkill: a sibling's llama-server must survive
  [ -n "$SP" ] && kill "$SP" 2>/dev/null; sleep 3; [ -n "$SP" ] && kill -0 "$SP" 2>/dev/null && { kill -9 "$SP" 2>/dev/null; sleep 2; }
  SP=""; [ -n "$SMOKE" ] || { wait_vram_free || nl_say "  WARN: VRAM still $(vram) MiB after server stop"; }; }
stop_sampler() { [ -n "$GP" ] && kill "$GP" 2>/dev/null; [ -n "$GP2" ] && kill "$GP2" 2>/dev/null; GP=""; GP2=""; }
restore() { stop_sampler; stop_server
  if [ "$UP" = 1 ]; then python3 "$OFFPY" down --run "$RUN" --image "$IMAGE" --port "$PORT" >> "$RUNLOG" 2>&1 || true; UP=0; fi   # post counters + readout + own container + own bridge
  if [ "$DRAINED" = 1 ]; then nl_say "EXIT trap: restoring Donald"; systemctl is-active --quiet llama-server.service || sudo -n systemctl start llama-server.service || true
    for i in $(seq 1 30); do curl -s -m 3 http://127.0.0.1:8090/health | grep -qi ok && { nl_say "Donald proxy health OK"; break; }; sleep 2; done; fi; }
trap restore EXIT

# 1 isolation proof FIRST (the container + the bridge come up, nothing needs the card yet)
nl_mark isolation-pre running
PCARGS=(); [ -n "${OFF_NOPOS:-}" ] && PCARGS=(--no-positive-control)
python3 -u "$OFFPY" up --run "$RUN" --image "$IMAGE" --port "$PORT" --model-id "$MODEL_ID" --ctx $CTX --budget "$BUDGET" "${PCARGS[@]}" >> "$RUNLOG" 2>&1; rc=$?; UP=1
if [ "$rc" != 0 ]; then nl_say "FAIL: isolation proof or container start failed (rc=$rc), no task runs, the card was not touched"; nl_mark isolation-pre failed "rc=$rc"; exit 5; fi
nl_mark isolation-pre done "$(grep -a -c 'FAILED as required' "$RUN/isolation/proof-pre.txt") probes failed as required"
nl_say "isolation proof PASS: $(sed -n 3,4p "$RUN/isolation/proof-pre.txt" | tr '\n' ' ' | cut -c1-200)"

# 2 the model server on its own port
CARD0=$(date +%s)
if [ -z "$SMOKE" ]; then
  nl_say "draining Donald"; sudo -n systemctl stop llama-server.service; DRAINED=1; sleep 3
  for i in $(seq 1 120); do v=$(vram); [ "${v:-99999}" -lt 4000 ] && { nl_say "drain clear (${v}MiB)"; break; }; sleep 2; [ "$i" -eq 120 ] && { nl_say "FATAL drain timeout (a sibling holds the card?)"; exit 1; }; done
fi
start_server() { # -> 0 up, 1 failed
  case "$SMOKE" in
    mock) python3 "$CODE/offline/mock_offline.py" "$PORT" >> "$RUN/server.log" 2>&1 & SP=$!;;
    cpu)  env CUDA_VISIBLE_DEVICES= "$SRV" -m "$MODELS/tiny-probe/qwen2.5-0.5b-instruct-q4_k_m.gguf" -ngl 0 --threads 8 -fa on -c $CTX --jinja --host 127.0.0.1 --port "$PORT" --log-timestamps --temp "$TEMP" --top-p "$TOP_P" --top-k "$TOP_K" --min-p "$MIN_P" >> "$RUN/server.log" 2>&1 & SP=$!;;
    *)    "$SRV" -m "$MODELS/$MODEL_REL" -ngl 99 -fa on -c $CTX --jinja --host 127.0.0.1 --port "$PORT" --log-timestamps --temp "$TEMP" --top-p "$TOP_P" --top-k "$TOP_K" --min-p "$MIN_P" >> "$RUN/server.log" 2>&1 & SP=$!;;
  esac
  local i; for i in $(seq 1 $((LOADWAIT/2))); do curl -s -m 3 http://127.0.0.1:$PORT/health | grep -qi ok && return 0; kill -0 $SP 2>/dev/null || break; sleep 2; done
  curl -s -m 3 http://127.0.0.1:$PORT/health | grep -qi ok && return 0
  stop_server; return 1
}
nl_mark model running
start_server || { nl_say "FAIL: model server did not come up (see $RUN/server.log)"; nl_mark model failed "server did not come up"; exit 1; }
VL=0; [ -n "$SMOKE" ] || VL=$(vram)
nl_say "  server up on 127.0.0.1:$PORT (loopback only), VRAM $VL MiB"; nl_mark model done "vram ${VL} MiB"
curl -s -m 10 http://127.0.0.1:$PORT/props > "$RUN/props.json" 2>/dev/null
EFFORT=unset
if [ -z "$SMOKE" ] || [ "$SMOKE" = cpu ]; then  # the reasoning-effort probe (Hermes sends medium when unset); a 400 sets high and is disclosed
  c=$(curl -s -o /tmp/off75-probe.json -w '%{http_code}' -m 180 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"x","messages":[{"role":"user","content":"Say ok."}],"reasoning_effort":"medium","max_tokens":24}')
  if [ "$c" != 200 ]; then c=$(curl -s -o /tmp/off75-probe.json -w '%{http_code}' -m 180 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"x","messages":[{"role":"user","content":"Say ok."}],"reasoning_effort":"high","max_tokens":24}'); [ "$c" = 200 ] && EFFORT=high || EFFORT=fail; fi
  nl_say "  reasoning_effort probe: $EFFORT (unset = Hermes' default 'medium' accepted; high = template rejects medium, the one disclosed deviation)"
  [ "$EFFORT" != fail ] || { nl_say "FAIL: both reasoning_effort medium and high rejected"; exit 1; }
  python3 - "$RUN/props.json" "$RUN/row-meta.json" "$TEMP" "$TOP_P" "$TOP_K" "$MIN_P" "$EFFORT" "$VL" "$IMAGE_ID" "$HV" "$($SRV --version 2>&1 | grep -i version | head -1)" <<'PY'
import json, sys
props, out, temp, tp, tk, mp, effort, vl, img, hv, build = sys.argv[1:12]
p = json.load(open(props)) if props else {}
g = (p.get("default_generation_settings") or {}).get("params") or {}
applied = {k: g.get(k) for k in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repeat_penalty")}
want = {"temperature": float(temp), "top_p": float(tp), "top_k": int(tk), "min_p": float(mp)}
ok = all(applied.get(k) is not None and abs(float(applied[k]) - v) < 1e-5 for k, v in want.items())
json.dump({"sampling_flags": want, "sampling_applied": applied, "sampling_match": ok, "effort": effort, "vram_loaded_mib": int(vl), "image_id": img, "hermes_version": hv, "llama_build": build,
           "template_mentions_reasoning_effort": "reasoning_effort" in (p.get("chat_template") or "")}, open(out, "w"), indent=1)
print("row-meta: sampling flags vs /props applied:", "MATCH" if ok else "MISMATCH", applied)
sys.exit(0 if ok else 5)
PY
  [ $? = 0 ] || { nl_say "FAIL: sampling flags do not match what /props reports"; exit 1; }
fi
[ -n "$SMOKE" ] || { nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits -l 2 >> "$RUN/gpu-samples.csv" 2>/dev/null & GP=$!; nvidia-smi --query-compute-apps=timestamp,pid,process_name,used_memory --format=csv,noheader,nounits -l 2 >> "$RUN/gpu-apps.csv" 2>/dev/null & GP2=$!; echo "$SP" > "$RUN/server.pid"; }

# 3 the tasks, inside the container with the cable out
nl_mark tasks running
TARGS=(); [ -n "${OFF_TASKS:-}" ] && TARGS=(--tasks "$OFF_TASKS")
for attempt in 1 2 3; do
  python3 -u "$OFFPY" tasks --run "$RUN" --image "$IMAGE" --port "$PORT" --model-id "$MODEL_ID" --ctx $CTX --effort "$EFFORT" --budget "$BUDGET" --deadline-epoch "$DL" --card-start-epoch "$CARD0" --card-cap-min "$CARDCAP" "${TARGS[@]}" >> "$RUNLOG" 2>&1
  rc=$?; [ "$rc" = 4 ] || break
  nl_say "  server unhealthy mid-run (attempt $attempt), restarting it"; stop_server; start_server || { nl_say "  restart failed"; break; }
done
for i in 01 02 03 04 05 06 07 08 09 10; do if [ -f "$RUN/t$i.result.json" ]; then nl_mark "t$i" done "$(python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print('PASS' if r['pass'] else 'FAIL', r['wall_s'], 's')" "$RUN/t$i.result.json")"; fi; done
stop_sampler
SKIP_FIRST=1 python3 "$CODE/v2/parse_server_log.py" "$RUN/server.log" > "$RUN/server-timings.json" 2>/dev/null || true
stop_server
[ "$rc" = 3 ] && nl_mark tasks failed "stopped at the hard stop or the card cap (resumable)" || nl_mark tasks done "rc=$rc"

# 4 post counters + readout (the EXIT trap would do it too; do it now so the verdict lands in the log)
python3 "$OFFPY" down --run "$RUN" --image "$IMAGE" --port "$PORT" >> "$RUNLOG" 2>&1; UP=0
nl_mark isolation-post done "$(grep -a -m1 'non-loopback interfaces' "$RUN/isolation/proof-post.txt" 2>/dev/null | cut -c1-160)"
nl_say "  $(grep -a -m1 'ISOLATION COUNTERS' "$RUN/isolation/proof-post.txt")"
[ -f "$RUN/readout.md" ] && nl_mark readout done "$RUN/readout.md" || nl_mark readout failed "no readout"
nl_say "line 1: $(grep -a -m1 '^line 1' "$RUN/readout.md" 2>/dev/null | cut -c1-260)"
nl_say "card time this run: $(( ($(date +%s)-CARD0)/60 )) min"
[ -n "${OFF_TASKS:-}" ] && { nl_say "smoke subset finished rc=$rc"; exit 0; }
[ "$rc" = 3 ] && { nl_say "tasks remain, resumable"; exit 3; }
[ "$rc" = 0 ] || { nl_say "FAILED rc=$rc"; exit "$rc"; }
for i in 01 02 03 04 05 06 07 08 09 10; do [ -f "$RUN/t$i.result.json" ] || { nl_say "t$i has no result, resumable"; exit 3; }; done
nl_say "all 10 tasks done, readout $RUN/readout.md"; exit 0
