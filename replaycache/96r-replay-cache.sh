#!/bin/bash
# 96r-replay-cache.sh: idle-queue item (2026-11-09).
# WHAT: a reproduction, on one RTX 5090 + the CPU, of NVIDIA's CPU-ARC paper (arXiv 2611.09891v1, submitted 2026-11-06): a cache of compiled reasoning (a Python function + parameter schema
#   per question family) replayed on a CPU by a small model that only pulls the numbers out of a new question. The paper reports cache accuracy >= its source model on 7 of 8 benchmarks. Its entries are built from each
#   test problem's own transcript and judged against the golden label, so that is accuracy on the problems the cache was built from; it never reports a miss or a false-hit rate and runs no small-model-alone control.
#   We test the axis it did not: apple/GSM-Symbolic (HF, CC-BY-NC-ND 4.0, eval only), 20 templates (random.Random(20261109)), ONE entry per template built from instance 0 of main WITHOUT the answer key in the loop,
#   then 10 held-out main instances (new names/numbers, Apple's key) and 10 near-miss p1 instances (the same template plus one extra clause) per template = 200 + 200 queries.
#   arms: A = the source (Qwen3.8-27B Q6_K, reasoning on, from scratch, GPU) | B = cache replay (CPU only: embedder + best entry + stock Llama-3.2-1B Q8_0 pulling the numbers + solve()) | C = the 1B alone (CPU, no cache).
# MOJO: a 27B model reasoning from scratch next to a CPU replaying its cached answer in about a second, with near-miss questions flashing red when the cache is confidently wrong | esoteric (a 6 Nov NVIDIA paper reproduced on one card, on the axis it never measured: the near-miss) | NVIDIA says a CPU can replay a big model's reasoning from a cache. I rebuilt it on my 5090: [on new questions it scores B% against A% for the 27B reasoning from scratch, but on near-miss questions it was confidently wrong X% of the time]
#
# PRE-REGISTERED (PREREG.md in the harness dir, sha256-frozen BEFORE any counted output in FROZEN.sha256, checked in the PLAN phase; rc_summary.py applies the rule, nothing is worded by hand):
#   CACHE HOLDS if B gated >= A - 3 pts on held-out main AND B gated >= C + 20 pts (B gated = the paper's gated serve: cache first, the 27B's answer on a miss; cache-only reported beside it; C = the stronger of two 1B controls).
#   NEAR-MISS TRAP if, at the threshold that keeps >= 90% of held-out main as cache hits (set on held-out main only), >= 20% of the 200 p1 queries are cache hits that return a wrong answer. both can fire.
#   verdicts: BROKEN (< 10 validated entries after the build, or A < 50% on held-out main) | INCOMPLETE (a query missing from A, B or C: exit 3, resumes) | HOLDS AND TRAPS | HOLDS, NO TRAP | DOES NOT HOLD, TRAPS | DOES NOT HOLD, NO TRAP.
#   a null or a loss is a valid result and is worded as such. the subject of any failure is the cache method or the model, never the harness. no speed multiplier is claimed.
# POSTABLE ARTEFACT (planned here, CPU only on capsule after the card is released): run/replaycache96/out/replay-cache.mp4 (16:9 1920x1080, ~65 s: an 8 s result card first (verdict title, big numbers, a grey how-it-was-measured
#   line under each row, claim + limits footer), then the split screen: left the 27B thinking with a token counter, right the cache replaying the same question on CPU (embed, best entry, the 1B pulls the numbers, solve()),
#   a right/wrong tally per side and a red CONFIDENTLY WRONG flash on every near-miss the cache gets wrong) + out/contact-1.png (0.5 s) + out/contact-2.png (30 s) + out/summary.txt + summary.json. the questions shown are the first
#   6 held-out + 6 near-miss of the fixed run order, not picked. the 27B side is played faster than it ran and says so; measured seconds are on the card.
# HONEST LIMITS: 20 templates of one benchmark family, one run; entries built by the same model that is arm A; stock 1B (not the paper's fine-tuned one), our embedder (EmbeddingGemma 300M Q8_0), compiled form only, no semantic-function
#   network; Q6_K, server-default house sampling; the 27B's seconds are measured four-up on one slot pool, the CPU seconds on six threads. Apple's labels score the arms; the source's own answer is what each entry is checked against.
#
# CARD TIME (honest). staging smoke 2026-11-09 17:27-17:35 CET on the real selection, card free, Donald drained and restored through night-lib (drain 9 s, load 4 s, restore 10 s, Donald /health ok after ~10 s):
#   20 real source solves at 4 slots: 4.0 min wall, mean 24.4 s and 1,332 tokens per solve (max 11,902 tokens = 198 s), 52 tok/s per slot, 19/20 right; 3 real builder loops (4 parallel with the solves): source 6-10 s + builder 45-220 s
#   (mean 148 s, 1,296-7,880 tokens), all three validated at attempt 0; 26.7 GB VRAM, 584 W peak. ESTIMATE (UNVERIFIED until the real leg): 400 solves x 24.4 s / 4 slots = 41 min (x1.3 for a heavier tail = 53 min),
#   20 builds in 5 waves of 4 x ~(8 + 150..230 s) = 14-19 min, with a rewrite on a third of them ~25 min, + drain/load/restore ~1 min => ~60 min expected, ~100 min worst. budget in the 01:10 block: 270 min, so held-out/near-miss stay at 10 each.
#   CPU legs (card released first): B 400 x 1.6 s = ~11 min, C 400 x 4.2 s = ~28 min, summary + render ~1 min => ~40 min (nice 19, 6 threads, CUDA_VISIBLE_DEVICES empty, -ngl 0, ports 8316/8317).
# WINDOW RULE: NOT_BEFORE Tue 2026-11-10 01:00 CET (exit 3 before it, nothing touched); HARD STOP 06:00 CET Tue 2026-11-10 (min with the block end); the GPU legs need >= 60 min left at start or exit 3 without draining Donald;
#   no new request starts in the last 22 min of the block's GPU legs (8 min reserve + 14 min tail); resumable PER query (results/A/<key>.json) and PER template (entries/<tid>.json); CPU legs run inline when >= 60 min remain after the card is released, else they are
#   detached in tmux 96r-cpu (CPU only, no card) and the item exits 3, the next run finds summary.json and finishes. DISK: downloads were 1.7 GB (done at staging); run output < 100 MB; delete nothing.
#
# test knobs (empty in production): RC_SMOKE=1 (slug replaycache96-smoke, 2 templates + 8 queries (RC_SMOKE_TPL / RC_SMOKE_ITEMS), no NOT_BEFORE, no window rule, no summary/render, never counted; the caller wraps it in the smoke flock and checks rule 3),
#   RC_STUB_UPSTREAM=http://127.0.0.1:PORT (CPU mock of the house model: no Donald drain, no server start; with RC_SEL_OVERRIDE=<sel.json> and RC_SLUG_SUFFIX=-mock; never counted),
#   RC_NO_WINDOW=1 (skip NOT_BEFORE and the 60-minute rule, stub only), RC_NOT_BEFORE=<epoch> (red-path test), RC_NO_SELFTEST=1, RC_NO_RENDER=1, MANIFEST_OVERRIDE (staging only).
set -uo pipefail
MANIFEST=/home/witcheer/replaycache96-status.json
NL_TAG=replay96
[ -n "${MANIFEST_OVERRIDE:-}" ] && MANIFEST=$MANIFEST_OVERRIDE
[ "${DRYRUN:-0}" = "1" ] && [ -z "${MANIFEST_OVERRIDE:-}" ] && MANIFEST=/tmp/replaycache96-dry-manifest.json   # a DRYRUN never touches the live manifest
export MANIFEST
. /home/witcheer/night-lib.sh
RC=/home/witcheer/esoteric/replay-cache
CODE=$RC/code
STUB=${RC_STUB_UPSTREAM:-}
SMOKE=${RC_SMOKE:-0}
SLUG=replaycache96${RC_SLUG_SUFFIX:-}; [ "$SMOKE" = 1 ] && SLUG=replaycache96-smoke
LIMG=""; [ "$SMOKE" = 1 ] && LIMG="--limit-templates ${RC_SMOKE_TPL:-2}"
LIMS="$LIMG"; [ "$SMOKE" = 1 ] && LIMS="$LIMG --limit-items ${RC_SMOKE_ITEMS:-8}"
[ "$SMOKE" = 1 ] && export RC_LIMITS="$LIMS"
RUN=$RC/run/$SLUG; ART=$RUN/summary.json; OUT=$RUN/out
[ -n "$STUB" ] && export RC_MOCK_RENDER=1                  # a stub run stamps MOCK on every frame
SEL=$CODE/sel.json; [ -n "$STUB" ] && [ -n "${RC_SEL_OVERRIDE:-}" ] && SEL=$RC_SEL_OVERRIDE
SRV=/home/witcheer/llama.cpp-donald/build/bin/llama-server; LL_VER="839a572a4"
MODEL=/home/witcheer/models/qwen38-27b/Qwen3.8-27B-Q6_K.gguf; MODEL_BYTES=22884408288
M1B=$RC/models/Llama-3.2-1B-Instruct-Q8_0.gguf; M1B_BYTES=1321083008; M1B_SHA=432f310a77f4650a88d0fd59ecdd7cebed8d684bafea53cbff0473542964f0c3
MEMB=$RC/models/embeddinggemma-300M-Q8_0.gguf; MEMB_BYTES=333590944; MEMB_SHA=b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63
FF=/home/witcheer/voxelbench/ffmpeg; RPY=$RC/.venv/bin/python
NOT_BEFORE=${RC_NOT_BEFORE:-1794268800}          # Tue 2026-11-10 01:00 CET
HARD_STOP=1794286800                              # Tue 2026-11-10 06:00 CET
. "$CODE/regime.env"                              # GPORT CTX NP RB RB_MSG MAXTOK SRV_FLAGS (frozen)
export RC_DATA=$RC/data

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0} slug=$SLUG stub=${STUB:-none})"
nl_check_path "$SRV" "$MODEL" "$M1B" "$MEMB" "$FF" "$RPY" "$RC/data/main/test.jsonl" "$RC/data/p1/test.jsonl" "$SEL" /home/witcheer/night-lib.sh \
  "$CODE/PREREG.md" "$CODE/FROZEN.sha256" "$CODE/sel.json" "$CODE/regime.env" "$CODE/rc_common.py" "$CODE/rc_gpu.py" "$CODE/rc_cpu.py" "$CODE/rc_summary.py" "$CODE/rc_render.py" "$CODE/rc_chain.sh" "$CODE/rc_selftest.py" "$CODE/rc_mockrun.py"
nl_check_disk 120
[ "$(stat -c %s "$MODEL" 2>/dev/null)" = "$MODEL_BYTES" ] && nl_ok "Qwen3.8-27B Q6_K $MODEL_BYTES bytes" || nl_fail "Q6_K size != $MODEL_BYTES"
[ "$(stat -c %s "$M1B" 2>/dev/null)" = "$M1B_BYTES" ] && nl_ok "1B Q8_0 $M1B_BYTES bytes" || nl_fail "1B size != $M1B_BYTES"
[ "$(stat -c %s "$MEMB" 2>/dev/null)" = "$MEMB_BYTES" ] && nl_ok "embedder Q8_0 $MEMB_BYTES bytes" || nl_fail "embedder size != $MEMB_BYTES"
# the pre-registration is frozen: prereg, code, regime and the selection must be byte-identical to what FROZEN.sha256 lists
FZ=$(cd "$CODE" && sha256sum -c FROZEN.sha256 2>&1 | grep -v ': OK$' || true)
[ -z "$FZ" ] && nl_ok "FROZEN.sha256: every frozen file matches" || nl_fail "frozen file changed: $FZ"
if [ "${DRYRUN:-0}" = "1" ]; then
  H1=$(sha256sum "$M1B" | cut -d' ' -f1); [ "$H1" = "$M1B_SHA" ] && nl_ok "1B sha256 ${H1:0:12}" || nl_fail "1B sha256 ${H1:0:12} != HF ${M1B_SHA:0:12}"
  H2=$(sha256sum "$MEMB" | cut -d' ' -f1); [ "$H2" = "$MEMB_SHA" ] && nl_ok "embedder sha256 ${H2:0:12}" || nl_fail "embedder sha256 ${H2:0:12} != HF ${MEMB_SHA:0:12}"
fi
# the data files must be the ones the selection was drawn from
DCHK=$(python3 - "$SEL" "$RC/data" <<'PYEOF'
import hashlib, json, sys
sel = json.load(open(sys.argv[1])); bad = []
for rel, want in sel["data_sha256"].items():
    h = hashlib.sha256(open(sys.argv[2] + "/" + rel, "rb").read()).hexdigest()
    if h != want: bad.append(rel)
print("BAD " + ",".join(bad) if bad else "OK %d templates %d held %d near" % (len(sel["templates"]), sum(len(t["held"]) for t in sel["templates"]), sum(len(t["near"]) for t in sel["templates"])))
PYEOF
)
case "$DCHK" in OK*) nl_ok "data sha256 matches the selection ($DCHK)";; *) nl_fail "data files differ from the selection: $DCHK";; esac
VER=$($SRV --version 2>&1)
case "$VER" in *"$LL_VER"*) nl_ok "llama.cpp build $LL_VER";; *) nl_fail "llama.cpp build changed (expected $LL_VER)";; esac
HELPL=$($SRV --help 2>&1 || true)
for f in --jinja --parallel --ctx-size --alias --n-gpu-layers --flash-attn --cache-type-k --cache-type-v --cache-ram --reasoning-format --reasoning-budget --reasoning-budget-message --temp --top-p --top-k --min-p --metrics --threads --embedding; do
  case "$HELPL" in *"$f"*) ;; *) nl_fail "llama-server flag $f not in this build";; esac
done
SS=$(ss -ltn 2>/dev/null || true)
for p in $GPORT 8316 8317 8318 8319; do case "$SS" in *":$p "*) nl_fail "port $p already in use";; esac; done
if [ "${DRYRUN:-0}" = "1" ]; then
  # leak scan: this item + prereg + code must not carry paths, tokens, tailnet addresses or e-mail shapes
  LK=$(grep -nE '/Users/|ghp_[A-Za-z0-9]{10}|hf_[A-Za-z0-9]{20}|100\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|\.ts\.net|@[a-z0-9-]+\.[a-z]{2,}' "$0" "$CODE"/*.py "$CODE/PREREG.md" "$CODE/rc_chain.sh" 2>/dev/null | grep -v 'users.noreply' | head -3 || true)
  [ -z "$LK" ] && nl_ok "leak scan clean" || nl_fail "leak scan: $LK"
  # CPU proofs, never counted: parser/statistics/sandbox/validator self-test, verdict branches, a mock run through the real scorer + renderer, the two CPU servers
  if [ -z "${RC_NO_SELFTEST:-}" ]; then
    ST=$(python3 "$CODE/rc_selftest.py" 2>&1 | tail -1); [ "$ST" = "selftest: ok" ] && nl_ok "rc_selftest: parser, statistics, sandbox, validator, selection" || nl_fail "rc_selftest: $ST"
    VB=$(python3 "$CODE/rc_summary.py" --selftest 2>&1 | tail -1); case "$VB" in "selftest: ok"*) nl_ok "rc_summary $VB";; *) nl_fail "rc_summary selftest: $VB";; esac
    MD=$(mktemp -d "$RC/run/dry.XXXXXX")
    python3 "$CODE/rc_mockrun.py" "$MD" "$CODE/sel.json" >/dev/null 2>&1
    MV=$(python3 "$CODE/rc_summary.py" "$MD" "$CODE/sel.json" --mock 2>&1 | grep -m1 '^VERDICT' | cut -c1-60)
    [ -n "$MV" ] && nl_ok "mock run scored: $MV" || nl_fail "mock run could not be scored"
    if [ -z "${RC_NO_RENDER:-}" ]; then
      mkdir -p "$MD/out"; "$RPY" "$CODE/rc_render.py" "$MD" "$MD/out/mock.mp4" --mock --ffmpeg "$FF" --card-s 3 --per-q 1.0 >/dev/null 2>&1 \
        && [ -s "$MD/out/mock.mp4" ] && nl_ok "mock render ok ($(stat -c %s "$MD/out/mock.mp4") bytes)" || nl_fail "mock render failed"
    fi
    # the two CPU servers start with CUDA_VISIBLE_DEVICES empty and -ngl 0, the embedder answers 768 dims, the 1B answers a schema-constrained call
    CUDA_VISIBLE_DEVICES= nice -n 19 "$SRV" -m "$MEMB" --host 127.0.0.1 --port 8319 -ngl 0 -t 2 -c 2048 --embedding --alias embed >"$MD/emb.log" 2>&1 & EP=$!
    CUDA_VISIBLE_DEVICES= nice -n 19 "$SRV" -m "$M1B" --host 127.0.0.1 --port 8318 -ngl 0 -t 4 -c 2048 --jinja --alias llama1b >"$MD/lm.log" 2>&1 & LP=$!
    for i in $(seq 1 60); do H1=$(curl -s -m 2 http://127.0.0.1:8318/health || true); H2=$(curl -s -m 2 http://127.0.0.1:8319/health || true); [[ ${H1,,} == *ok* && ${H2,,} == *ok* ]] && break; sleep 1; done
    PR=$(python3 - <<'PYEOF'
import json, urllib.request
def post(u, o): return json.loads(urllib.request.urlopen(urllib.request.Request(u, data=json.dumps(o).encode(), headers={"Content-Type": "application/json"}), timeout=60).read())
try:
    e = post("http://127.0.0.1:8319/v1/embeddings", {"model": "embed", "input": ["task: search result | query: hello"]})["data"][0]["embedding"]
    r = post("http://127.0.0.1:8318/v1/chat/completions", {"model": "llama1b", "temperature": 0, "max_tokens": 60, "messages": [{"role": "user", "content": "x is 5 and y is 7. Give x and y."}],
             "response_format": {"type": "json_schema", "json_schema": {"name": "p", "strict": True, "schema": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}}, "required": ["x", "y"], "additionalProperties": False}}}})
    j = json.loads(r["choices"][0]["message"]["content"]); print("OK dims=%d x=%s y=%s" % (len(e), j["x"], j["y"]))
except Exception as ex: print("FAIL", repr(ex)[:100])
PYEOF
)
    kill $EP $LP 2>/dev/null; wait $EP $LP 2>/dev/null
    case "$PR" in "OK dims=768 x=5"*) nl_ok "CPU servers (CUDA_VISIBLE_DEVICES empty, -ngl 0): $PR";; *) nl_fail "CPU server probe: $PR";; esac
    rm -rf "$MD"
  fi
fi
if nl_skip_exact "$ART"; then nl_say "nothing to run, item complete ($ART exists)"; exit 0; fi

NOW=$(date +%s)
DL=$(nl_deadline_epoch); [ "$DL" -gt "$HARD_STOP" ] && DL=$HARD_STOP
LEFT=$(( (DL-NOW)/60 ))
PS=$(python3 "$CODE/rc_gpu.py" source --base x --out "$RUN" --sel "$SEL" --deadline 0 --count-only $LIMS 2>&1 | tail -1); PB=$(python3 "$CODE/rc_gpu.py" build --base x --out "$RUN" --sel "$SEL" --deadline 0 --count-only $LIMG 2>&1 | tail -1)
nl_say "  RUN gpu_source -> $RUN/results/A/<key>.json   ($PS)"
nl_say "  RUN gpu_build  -> $RUN/entries/<tid>.json     ($PB)"
nl_say "  RUN cpu_B + cpu_C + summary -> $RUN/cpu/, $ART ; clip -> $OUT/replay-cache.mp4 ; hard stop $(date -d "@$DL" '+%a %H:%M') local, $LEFT min away; NOT_BEFORE $(date -d "@$NOT_BEFORE" '+%a %H:%M')"
if [ "${DRYRUN:-0}" != "1" ] && [ -z "$STUB" ] && [ "$SMOKE" != 1 ]; then
  [ "$NOW" -ge "$NOT_BEFORE" ] || { nl_say "  before NOT_BEFORE ($(date -d "@$NOT_BEFORE" '+%a %H:%M') local): not starting (exit 3, nothing touched)"; exit 3; }
  [ "$DL" -gt "$NOW" ] || { nl_say "  past the item's hard stop: exit 3"; exit 3; }
fi
GPU_PENDING=0; case "$PS$PB" in *"PENDING source 0 of"*"PENDING build 0 of"*) ;; *) GPU_PENDING=1;; esac
if [ "${DRYRUN:-0}" != "1" ] && [ "$GPU_PENDING" = 1 ] && [ -z "${RC_NO_WINDOW:-}" ] && [ "$SMOKE" != 1 ]; then
  [ "$LEFT" -ge 60 ] || { nl_say "  under 60 min left in the window: not starting (drain + load + a leg would be cut short; exit 3, resumable, Donald untouched)"; exit 3; }
fi
if [ "${DRYRUN:-0}" = "1" ] || [ ! -f "$MANIFEST" ]; then nl_manifest_init gpu_source gpu_build cpu_B cpu_C summary render; fi
nl_gate

# ---- phase 2 (real run) ----
mkdir -p "$RUN/out" "$RUN/cpu"
CHAIN_TMUX=96r-cpu
CHAIN_LIVE=$(tmux ls -F '#{session_name}' 2>/dev/null | tr '\n' ' ')
case " $CHAIN_LIVE " in *" $CHAIN_TMUX "*) nl_say "the detached CPU chain ($CHAIN_TMUX) is still running: nothing to do now (exit 3)"; exit 3;; esac
mem_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
wait_mem_free() { local i v; for i in $(seq 1 90); do v=$(mem_used); [ "${v:-99999}" -lt 1024 ] && { nl_say "  card clear (${v} MiB)"; return 0; }; sleep 2; done; nl_say "  card NOT clear after 180 s (${v} MiB)"; return 1; }
SP=""; GS=""; DONALD_STOPPED=0; DONALD_BACK=0
stop_srv() { [ -n "$GS" ] && { kill "$GS" 2>/dev/null; GS=""; }; [ -n "$SP" ] || return 0; kill -TERM "$SP" 2>/dev/null; for i in $(seq 1 30); do kill -0 "$SP" 2>/dev/null || break; sleep 2; done; kill -KILL "$SP" 2>/dev/null; sleep 3; SP=""; }
donald_restore() {
  [ "$DONALD_STOPPED" = 1 ] && [ "$DONALD_BACK" = 0 ] || return 0
  DONALD_BACK=1; nl_say "restoring Donald (night-lib nl_donald_restore)"
  nl_donald_restore
  local i H; for i in $(seq 1 60); do H=$(curl -s -m 5 localhost:8090/health); [[ "${H,,}" == *ok* ]] && { nl_say "Donald healthy again after ~$((i*5)) s"; return 0; }; sleep 5; done
  nl_say "WARNING: Donald /health did not answer within 5 min after the restore"
}
cleanup() { stop_srv; [ "$DONALD_STOPPED" = 1 ] && { wait_mem_free || true; }; donald_restore; }
trap cleanup EXIT
fail() { nl_mark "$1" failed "$2"; nl_say "FAIL: $2"; exit 1; }
REMAIN=0
GBASE=http://127.0.0.1:$GPORT
[ -n "$STUB" ] && GBASE=$STUB
GPU_DL=$(( DL - 8*60 ))

if [ "$GPU_PENDING" = 1 ]; then
  if [ -n "$STUB" ]; then nl_say "STUB UPSTREAM $STUB: no Donald drain, no server start"
  else
    FOREIGN=$(nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null | grep -v llama-server | tr -d ' ' | tr '\n' ' ' || true)
    nl_say "  rule 3, live: foreign compute apps: ${FOREIGN:-none}; tmux: $(tmux ls -F '#{session_name}' 2>/dev/null | tr '\n' ' ')"
    [ -z "$FOREIGN" ] || fail gpu_source "a foreign GPU process owns the card: $FOREIGN"
    nl_say "draining Donald (night-lib nl_donald_stop)"; DONALD_STOPPED=1; nl_donald_stop
    for i in $(seq 1 60); do v=$(mem_used); [ "${v:-99999}" -lt 1024 ] && { nl_say "drain clear (${v} MiB)"; break; }; sleep 2; done
    [ "${v:-99999}" -lt 1024 ] || fail gpu_source "card memory still ${v:-?} MiB after the drain: another process owns the card, not loading"
    T_CARD=$(date +%s)
    "$SRV" --model "$MODEL" --host 127.0.0.1 --port $GPORT $SRV_FLAGS --reasoning-budget $RB --reasoning-budget-message "$RB_MSG" > "$RUN/server.log" 2>&1 &
    SP=$!
    for i in $(seq 1 120); do H=$(curl -s -m 3 $GBASE/health || true); [[ ${H,,} == *ok* ]] && break; kill -0 "$SP" 2>/dev/null || break; sleep 2; done
    [[ ${H,,} == *ok* ]] || fail gpu_source "server did not come up"
    nl_say "  server up after $(( $(date +%s) - T_CARD )) s, VRAM $(mem_used) MiB"
    SM=$(python3 - "$GBASE" <<'PYEOF'
import json, sys, urllib.request
b = sys.argv[1]
try:
    r = json.loads(urllib.request.urlopen(urllib.request.Request(b + "/v1/chat/completions", data=json.dumps({"model": "qwen38-27b", "max_tokens": 1500, "messages": [{"role": "user", "content": "What is 17 * 23? Answer briefly."}], "chat_template_kwargs": {"enable_thinking": True}}).encode(), headers={"Content-Type": "application/json"}), timeout=300).read())
    m = r["choices"][0]["message"]; ok = bool(m.get("reasoning_content")) and "391" in (m.get("content") or "")
    print(("SMOKE OK" if ok else "SMOKE FAIL"), "think_chars", len(m.get("reasoning_content") or ""))
except Exception as e: print("SMOKE FAIL", repr(e)[:80])
PYEOF
)
    nl_say "  template smoke: $SM"; case "$SM" in "SMOKE OK"*) ;; *) fail gpu_source "template smoke failed ($SM)";; esac
    ( while :; do echo "$(date +%s),$(nvidia-smi --query-gpu=power.draw,memory.used,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')" >> "$RUN/gpu.csv"; sleep 2; done ) >/dev/null 2>&1 &
    GS=$!
  fi
  T_GPU=$(date +%s)
  nl_mark gpu_source running
  python3 "$CODE/rc_gpu.py" source --base $GBASE --out "$RUN" --sel "$SEL" --deadline $GPU_DL --slots $NP --max-tokens $MAXTOK $LIMS >> "$RUN/legs-source.log" 2>&1; src=$?
  nl_say "  gpu_source rc=$src: $(grep -E '\[rc_gpu\] leg source end' "$RUN/legs-source.log" | tail -1 | cut -c1-200)"
  if [ "$src" = 0 ]; then nl_mark gpu_source done; else REMAIN=1; nl_mark gpu_source skipped "rc=$src"; fi
  nl_mark gpu_build running
  python3 "$CODE/rc_gpu.py" build --base $GBASE --out "$RUN" --sel "$SEL" --deadline $GPU_DL --slots $NP --max-tokens $MAXTOK $LIMG >> "$RUN/legs-build.log" 2>&1; brc=$?
  nl_say "  gpu_build rc=$brc: $(grep -E '\[rc_gpu\] leg build end' "$RUN/legs-build.log" | tail -1 | cut -c1-200)"
  if [ "$brc" = 0 ]; then nl_mark gpu_build done; else REMAIN=1; nl_mark gpu_build skipped "rc=$brc"; fi
  curl -s -m 5 "$GBASE/metrics" 2>/dev/null | grep -E '^llamacpp:(tokens_predicted_total|tokens_predicted_seconds_total|prompt_tokens_total)' | tr '\n' ' ' > "$RUN/metrics-end-$(date +%H%M).txt" || true
  # the card is released here: server stopped, Donald back, THEN the CPU work
  stop_srv
  if [ -z "$STUB" ]; then
    wait_mem_free || true; donald_restore
    CARD_MIN=$(( ($(date +%s) - T_CARD + 59) / 60 ))
    python3 - "$RUN/card.json" "$CARD_MIN" "$RUN/gpu.csv" <<'PYEOF'
import json, os, sys
p, add, csvp = sys.argv[1], int(sys.argv[2]), sys.argv[3]
d = json.load(open(p)) if os.path.exists(p) else {"card_min": 0, "blocks": 0}
d["card_min"] += add; d["blocks"] += 1
try:
    rows = [l.strip().split(",") for l in open(csvp) if l.count(",") == 3]
    d["gpu_w_max"] = max(float(r[1]) for r in rows); d["vram_mib_max"] = max(int(float(r[2])) for r in rows)
except Exception: pass
json.dump(d, open(p, "w"), indent=1)
PYEOF
    nl_say "  card released after ~${CARD_MIN} min this block (total in card.json)"
  fi
fi
S2=$(python3 "$CODE/rc_gpu.py" source --base x --out "$RUN" --sel "$SEL" --deadline 0 --count-only $LIMS 2>&1 | tail -1); B2=$(python3 "$CODE/rc_gpu.py" build --base x --out "$RUN" --sel "$SEL" --deadline 0 --count-only $LIMG 2>&1 | tail -1)
case "$S2$B2" in *"PENDING source 0 of"*"PENDING build 0 of"*) ;; *) nl_say "  GPU legs not finished ($S2 | $B2): exit 3, resumable, summary.json NOT written"; exit 3;; esac

# ---- CPU legs (card released): inline when they fit, otherwise detached ----
NOW=$(date +%s); LEFT2=$(( (DL-NOW)/60 ))
if [ -z "$STUB" ] && [ "$LEFT2" -lt 60 ] && [ "${RC_FORCE_INLINE:-0}" != 1 ]; then
  nl_say "  only $LEFT2 min left after the card: detaching the CPU legs in tmux $CHAIN_TMUX (CPU only, no card) and exiting 3"
  tmux new-session -d -s "$CHAIN_TMUX" "RC_LIMITS='${RC_LIMITS:-}' RC_NO_RENDER=${RC_NO_RENDER:-} bash $CODE/rc_chain.sh --nogpu ALL $RUN $SEL $RUN/entries >> $RUN/cpu-chain.log 2>&1"
  sleep 3; tmux has-session -t "$CHAIN_TMUX" 2>/dev/null && nl_say "  chain started: tail -f $RUN/cpu-chain.log" || nl_say "  chain exited fast: $(tail -2 "$RUN/cpu-chain.log" | tr '\n' ' ')"
  exit 3
fi
nl_mark cpu_B running
RC_DEADLINE=$(( DL - 180 )) RC_NO_RENDER=${RC_NO_RENDER:-} bash "$CODE/rc_chain.sh" --nogpu ALL "$RUN" "$SEL" "$RUN/entries" >> "$RUN/cpu-chain.log" 2>&1; crc=$?
nl_say "  cpu chain rc=$crc: $(tail -3 "$RUN/cpu-chain.log" | tr '\n' ' ' | cut -c1-240)"
if [ "$SMOKE" = 1 ]; then nl_say "SMOKE done (rc=$crc): $(ls "$RUN/cpu/B" 2>/dev/null | wc -l) B + $(ls "$RUN/cpu/C" 2>/dev/null | wc -l) C results, $(ls "$RUN/entries"/*.json 2>/dev/null | wc -l) entries; no summary/render in smoke"; exit 0; fi
NB=$(ls "$RUN/cpu/B" 2>/dev/null | wc -l); NC=$(ls "$RUN/cpu/C" 2>/dev/null | wc -l)
[ "$NB" -ge 400 ] && nl_mark cpu_B done "$NB" ; [ "$NC" -ge 400 ] && nl_mark cpu_C done "$NC"
if [ "$crc" = 3 ]; then nl_say "  a CPU leg could not finish inside the window (B $NB/400, C $NC/400): exit 3, resumable (INCOMPLETE, summary.pre.json if written)"; exit 3; fi
[ "$crc" = 0 ] || { nl_mark summary failed "chain rc=$crc"; nl_say "FAIL: cpu chain rc=$crc"; exit 1; }
nl_mark summary done "$(grep -m1 '^VERDICT' "$OUT/summary.txt" | cut -c9-70)"
nl_judge_artifact render 0 "$OUT/replay-cache.mp4" "$(date -d '-3 hours' +%s)" "clip" || nl_say "  render missing (see $OUT/render.log); the verdict stands"
{ echo "# replay-cache readout ($SLUG)"; echo; sed -n 1,2p "$OUT/summary.txt"; echo; echo "regime: llama.cpp $LL_VER; A = Qwen3.8-27B Q6_K, 4 slots, ctx $CTX, KV q8_0, temp 1.0 top-p 0.95 top-k 20, reasoning budget $RB, max $MAXTOK tokens; B/C = Llama-3.2-1B Q8_0 + EmbeddingGemma-300M Q8_0 on CPU (6 threads, -ngl 0)"; } > "$OUT/readout.md"
L1=$(cat "$OUT/line1.txt" 2>/dev/null)
echo "# MOJO line 1 (rule-built): $L1" > "$OUT/mojo-line1.txt"
nl_say "ANGLE: $L1"
exit 0
