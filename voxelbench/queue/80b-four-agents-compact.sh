#!/bin/bash
# 80b-four-agents-compact.sh: idle-queue item (STAGED in ~/voxel-stage; built 2026-11-03 by a mercury subagent for the GPU thread). RERUN A of item 80.
# operator, GPU thread, 2026-11-03 ~16:00 CET: "rerun as A" = keep four agents, each compacts its context earlier so the four together never exceed the pooled KV; the yes covers
#   building and enqueueing this rerun. why: item 80 (ran 14:26-14:45 CET) filled the shared 163,840-token KV pool at 14:38-14:40 with all four agents at 39-41k tokens of context;
#   llama-server rejected all four in-flight requests, three agents stopped on that error, one finished (vault 99-personal/benchmark-rig/four-agents-80-status.md). cause: our pool sizing.
# MOJO: four Hermes Agent sessions on one RTX 5090 building the same thing at once, each compacting its own context so the shared KV never runs out | intense/never-seen | one RTX 5090 is serving four Hermes Agents at once: same prompt, same local Qwen3.8-27B, four panes racing, [N] tok/s combined on the card, and not one of them hit the memory wall
# EV: rule 4 (fully local, reproducible Hermes piece: one card, one GGUF, four private Hermes homes, scripts in voxelbench/four-agents) + mojo hook intense / never-seen (a race on screen with a live aggregate HUD;
#   checked 2026-11-03: a filmed four-session Hermes race on one card was not found on our account). comparables ("Hermes Agent on a local model" posts, views, gpu-ev-gate-2026-10-25.md section 1):
#   14,324 / 7,850 / 5,476 / 5,168 / 4,640. bracket 4-10k. ONLY post material if the pre-registered rule below says PASS: a FAIL (any agent on a server error) reads as the stack failing and stays internal.
#   HONEST LIMITS: one race = four samples of one model, no 'agent X is better'; per-agent tok/s is not the single-slot series speed; a page that fails the gate shows FAIL in the result tile; the card is
#   the same one-race-per-config sample as item 80, so item 80 vs 80b is a before/after of ONE setting (compaction budget + proxy guard + -n cap), not a controlled A/B.
# serves: content (the card as a team server, a Hermes-driven showcase) + hardening (a parallel-slot Hermes harness that never touches the main gateway; fixes our pool sizing, the item-80 failure)
#
# CHANGE vs ITEM 80 (everything else is the same: prompt lighthouse-v1.txt md5 cff719be0490, model Qwen3.8-27B Q6_K, server build 839a572 and flags, series sampling, RB 16384, MTP n-max 2,
#   isolation per agent, counting proxy, HUD, recorder, gates, clip + mosaic):
#   1. each agent's home config (four_mkhome.py v2-compact) adds `compression.threshold_tokens: 34000` and `compression.protect_first_n: 0`. threshold_tokens is the documented absolute cap on the compaction trigger
#      (Hermes compacts at the LOWER of its ratio trigger and this count; live docs https://hermes-agent.nousresearch.com/docs/developer-guide/context-compression-and-caching, parameter table, fetched 2026-11-03; same key
#      and min() clamp in the pinned checkout 9bda188: agent/agent_init.py ~1538, agent/context_compressor.py _effective_threshold_cap). context_length STAYS 65536: the pinned install refuses a window under 64,000
#      (MINIMUM_CONTEXT_LENGTH, agent/agent_init.py), so the per-agent budget cannot go through context_length. ITEM-80 LESSON: with no cap the trigger is the 0.75 small-window floor x 65,536 = 49,152, not the ~33k
#      this harness assumed; four agents at 49k each = 197k > 163,840.
#   2. WHY 34,000 AND NOT LOWER (found by a CPU run on 2026-11-03, the mock race: agents at ~71k tokens never compacted): a compaction needs more than head + 3 messages. After an agent's FIRST turn it holds 3 messages
#      (task, write_file call, result) at 24.5-29.8k tokens (item 80: 24.5 / 26.1 / 29.6 / 29.8k = 5.2k prompt + a 19-25k generation). A trigger at or under that makes Hermes try, find nothing to compact,
#      and wait 300 s before it tries again (agent/context_compressor.py _STRUCTURAL_NO_OP_BACKOFF_SECONDS, read from the pinned source): the agent would then not compact for 5 minutes,
#      the window in which item 80's pool filled (t=586-684 s). 34,000 clears the largest first turn seen by 14% (a page of ~33 KB). With the default protect_first_n 3 a compaction needs 8+ messages and the first
#      three rows (the 20k-token blob) are kept until the first compaction, so b/d (34.8-35.0k at messages 5-7) would hit the same wait: protect_first_n 0 makes compaction legal from the 5th message and lets it drop that blob.
#      REPLAY (item 80 agent b's real transcript through the pinned ContextCompressor, summariser stubbed, scripts/replay_compress.py): 5 messages 19.9k -> 3.8k rough tokens, 7 messages 21.0k -> 4.1k,
#      3 messages = refused + 300 s wait armed; default head at 7 messages = refused. a compacted session is then ~4.8k (system + tools) + <= 3.3k summary + a short tail = ~10-13k.
#   3. sizing against the pool: item 80's request log (hud/requests.jsonl): first turns end at 24.5-29.8k, then a full-page rewrite of 5-10k tokens (a 9,968, c 7,216, d 5,148) put a, c, d at 36-39k while b sat at 41k, all four
#      inside t=586-684 s; their sum 158k + prefill filled the 163,840 pool. 80b cannot lower that rewrite phase (it starts below the trigger); it compacts each agent as it crosses 34k, one after another (the finishes were
#      staggered by 10-70 s), which takes each from ~38k to ~12k within about a minute (summary request ~35-40k prompt prefill + <= 3.3k generation, ~100 s). so the pool is also raised to 180,224 tokens (below) and guarded.
#      worst case if all four sit at 34k and each starts a 10k generation = 176k: inside 180,224, outside 163,840; that is the margin the bigger pool buys. NOT proven on the real model: UNVERIFIED until it runs.
#   4. pool: LADDER 180224 first, 163840 and 131072 as fallbacks, a rung is kept only with >= FREE_FLOOR (1,200) MiB free. measured 44.0 KiB/token q8_0 (item 80: 163,840 = 30,408 MiB used / 2,199 free; 196,608 = 31,816 / 791):
#      180,224 is predicted at ~31,112 MiB used / ~1,495 free (more headroom than the 791 of the 196,608 rung item 80 refused). if it loads with less, the ladder drops to 163,840 by itself.
#   5. guard A (proxy, four_proxy.py --pool, 92% of the pool actually loaded): admission control. a request is held (<= 90 s, never when no other request is running) while the estimated pool use would exceed the limit;
#      the hold is logged (guard_wait_s) and counted; it turns a would-be 500 into a short wait. a Hermes summary call (no tools array) is reserved at 4k tokens, a normal request at 12k, a first request at 26.6k. NOT a Hermes feature.
#   6. guard B (server, llama-server --predict 32768): no single generation runs past 32,768 tokens (item 80 first turns: 19-25k), so a runaway loop cannot fill the pool alone.
#   7. the agent that stops on a server error now ends as 'server_error' (four_record.py), no longer 'done': item 80's pane-quiet detector counted three such stops as finishes.
# PRE-REGISTERED SUCCESS RULE (written before the run; four_summary.py computes it, summary.json -> success_rule):
#   PASS = ALL four agents (1) saw no failed request (HTTP != 200, a dropped connection, or an error event inside a stream), (2) ended 'done' (finished on their own) or 'wall' (40 min cap), never
#   'server_error' or an early Hermes exit, (3) the server log holds no context-exceeded / no-memory-slot line, and (4) every agent has at least one completed request with server timings (a run with no
#   data cannot pass). Else FAIL, naming the broken clause.
#   REPORTED NEXT TO IT, never folded in: gates per page (liveness only; a vision read of the result mosaic stays mandatory, the gate passes a page with no lighthouse), per-agent tok/s (end to end and decode),
#   aggregate tok/s (mean while 3+ agents generate, peak 5 s), compaction events per agent counted from each agent's own state.db (rows with _compressed_summary=1), peak context per agent and the
#   estimated pool peak, proxy guard holds. PASS with 0 compaction events in an agent whose context never reached 34,000 is fine; an agent that DID pass 34,000 without a compaction event is a broken mechanism, said plainly.
# ANGLE after PASS: line 1 above, [N] = the measured mean aggregate; rule 4; call X as a 'which card, how many agents' piece, never a speed brag; the sceptical point to expect is 'thinking-heavy: the first 5 minutes are all reasoning'.
#
# WHAT: Qwen3.8-27B Q6_K served once by llama-server with --parallel 4 (pooled KV, q8_0); four Hermes Agent sessions, each its OWN HERMES_HOME (not a profile of the main install,
#   so the harness stays isolated from the always-on gateway), each behind its own port on a counting proxy, get the SAME prompt (prompts/lighthouse-v1.txt: one animated
#   isometric voxel lighthouse island, single HTML file, plain canvas) typed into four tmux/pty panes and submitted at the same instant. Time-capped at WALL min. A HUD sampler logs
#   llama-server /metrics + /slots and per-agent stream counts every second. After the race: Donald restored, the four pages gated on CPU (no card memory), the clip rendered
#   (2x2 grid, HUD, timelapse middle, results mosaic) on capsule.
# CLIP PLAN (planned here, not rescued later): runs/<slug>/out/four-agents-final.mp4 = race clip (real speed first 20 s and last 30 s, timelapse between, per-pane clocks + tok/s, aggregate
#   tok/s + sparkline + 'one agent alone' baseline, result card) + a 12 s results mosaic of the four pages (same fixed 2x2 layout, each page at its own default view). fixed layout 1920x1080.
#   HUD LEAK READ: Hermes' status-bar row is blanked in every pane; any cell matching the account name / home path / tailnet address / hostname / e-mail shape is drawn as a block (counts in
#   out/race.mp4.summary.json 'leak_masked_states', leakscan.json); work dirs are /tmp/arena80/<agent> and HOME is a throwaway dir, so no home path appears by construction. contact.png = a 3x3 sheet
#   of stills for the sceptical read before anything is offered (grid fixed? any text clipped? any path/name/host? do the four pages differ and move?).
#
# EVIDENCE BEHIND THE FLAGS (vault note 99-personal/benchmark-rig/four-agents-80-status.md has the measured numbers and what is still UNVERIFIED):
#   server flags are the 72fa-voxel-trees-h4-qwen line (Donald's build 839a572, Qwen3.8-27B Q6_K, q8_0 KV, flash-attn, temp 1.0 / top-p 0.95 / top-k 20 / min-p 0.05 = the GGUF keys,
#   RB 16384 + message, MTP n-max 2) plus --parallel 4 and a pooled KV (--kv-unified, one pool of CTX tokens shared by the four slots). Hermes needs >= 64,000 per agent (MINIMUM_CONTEXT_LENGTH);
#   4 x 65536 split KV does not fit 32 GB with this file (ladder rung 0 in the smoke = the OOM number), so the pool is smaller than 4 x 64k and every agent's config says
#   context_length 65536 (item 80 assumed Hermes compacts near 33k: wrong, the small-window floor makes it 49,152; 80b caps it at 34,000, see CHANGE above).
#   agents: `<checkout>/.hermes/bin/hermes chat --cli --toolsets terminal,file` with HERMES_HOME=<own>, HOME=<throwaway>, HERMES_DISABLE_LAZY_INSTALLS=1 and the main root's installs/ + tools/ symlinked into
#   the home (without the symlinks a new data root builds its own runtime: python-olm cmake, minutes, then fails; the checkout's old venv lacks modules, every tool call died on snowballstemmer).
# REGIME (label every number and clip with it): parallel 4 on one server, pooled KV; per-agent speed is NOT the single-slot speed of the series; plain first-turn prompt, not /goal (a finish time
#   is then defined: the pane returns to its prompt and the proxy shows no request in flight for 20 s); toolsets terminal + file only (no delegation: a sub-agent would add requests the HUD
#   cannot attribute); thinking on, RB 16384; approvals off; memory off; temperature 1.0 so the four runs differ (that is the point of a race).
# resumable: needs >= 90 min left in the window (load + probe + WALL min race + gates + clip, about 60 min worst case with WALL 40) or exits 3 (stays queued).
# card time: about 50 min (load 2 + probe 3 + race <= WALL + 3 barrier/slack + restore 1); the gates and the render run after Donald is back (CPU only).
# artifact: runs/<slug>/summary.json (exact). bench homes are throwaway (/tmp/arena80); the recorder/proxy/server pids are this item's own, never killed by name.
# test knobs (empty in production): FOUR_COMPACT_TOKENS (default 34000; 0 = the item-80 v1 homes), FOUR_POOL_GUARD=0 (proxy guard off), FOUR_NO_PREDICT_CAP=1 (no --predict), FOUR_SMOKE=1 (slug -smoke, WALL 7 min, no 90-min rule; run it by hand in tmux with NL_HARD_STOP_EPOCH set), FOUR_WALL_MIN, FOUR_LADDER ("ctx:mode ..." mode = unified|split),
#   FOUR_NO_MTP=1, FOUR_MODEL_FILE (red-path test), FOUR_STUB_UPSTREAM=http://127.0.0.1:PORT (CPU stub: no Donald drain, no server start, the race runs against a mock; DRYRUN-safe).
set -uo pipefail
NL_TAG=voxfour80b
MANIFEST=/home/witcheer/voxel-voxfour80b-status.json
# throwaway manifest for a staging DRYRUN only (the drain greps the literal ^MANIFEST= line above)
[ -n "${MANIFEST_OVERRIDE:-}" ] && MANIFEST=$MANIFEST_OVERRIDE
export MANIFEST
. /home/witcheer/night-lib.sh
V=/home/witcheer/voxelbench
FA=$V/four-agents
BIN=/home/witcheer/llama.cpp-donald/build/bin/llama-server
GPY=/home/witcheer/benchmark-rig/.venv/bin/python     # playwright (the four_gate.py CPU gate)
RPY=/home/witcheer/scratch/race/venv/bin/python        # pyte + PIL + fontTools (recorder + renderer)
CHECKOUT=/home/witcheer/.hermes/hermes-agent
HBIN=$CHECKOUT/.hermes/bin/hermes
HERMES_PIN=9bda188
FF=$V/ffmpeg
MODEL_FILE=${FOUR_MODEL_FILE:-/home/witcheer/models/qwen38-27b/Qwen3.8-27B-Q6_K.gguf}
MODEL_BYTES=22884408288
ALIAS=Qwen3.8-27B-Q6_K.gguf
PORT=8190                                              # own port: never :8090 (llama-swap / Donald)
KV=q8_0
SAMPLING_FLAGS="--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05"
PARAMS_SRC="GGUF general.sampling keys of the maker file (temp 1.0, top_p 0.95, top_k 20), read 2026-10-22, voxelbench/model-params.md row Qwen3.8-27B Q6_K; no min_p key, so llama.cpp default 0.05; RB 16384 (series value)"
EXTRA_ARR=(--spec-type draft-mtp --spec-draft-n-max 2)
[ "${FOUR_NO_MTP:-0}" = 1 ] && EXTRA_ARR=()
PRED_ARR=(); [ "${FOUR_NO_PREDICT_CAP:-0}" = 1 ] || PRED_ARR=(--predict 32768)     # = PREDICT_CAP
RB=16384
RB_MSG="Thinking budget reached. Stop planning and write the files now with your tools; refine in later turns."
LADDER=${FOUR_LADDER:-"180224:unified 163840:unified 131072:unified"}    # measured 2026-11-03: 163840 loads at 30,408 MiB (2,199 free), race peak 30,508; 196608 at 31,816 (791 free); 4 x 65536 split fails to fit; 180224 is predicted ~1,495 free
FREE_FLOOR=${FOUR_FREE_FLOOR:-1200}                         # MiB the card must keep free with the server loaded (item 80 used 800)
WALL=${FOUR_WALL_MIN:-40}
COMPACT=${FOUR_COMPACT_TOKENS:-34000}                      # per-agent compaction trigger in tokens (compression.threshold_tokens); 0 = item-80 v1 homes
GUARD_PCT=${FOUR_POOL_GUARD:-92}                           # proxy admission guard: hold requests while the estimated pool use would exceed this % of the pool; 0 = off
PREDICT_CAP=32768                                          # server-side cap on one generation (item 80 first turns: 19-25k); FOUR_NO_PREDICT_CAP=1 drops it
SLUG=four80b-qwen38-q6k-par4-compact
SMOKE=${FOUR_SMOKE:-0}
SLUG=$SLUG${FOUR_SLUG_SUFFIX:-}                       # test knob: lets the stub tests run several configs side by side
if [ "$SMOKE" = 1 ]; then SLUG=$SLUG-smoke; WALL=${FOUR_WALL_MIN:-7}; fi
STUB=${FOUR_STUB_UPSTREAM:-}
RUN=$V/runs/$SLUG
ART=$RUN/summary.json
PROMPT=$FA/prompts/lighthouse-v1.txt

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0} smoke=$SMOKE stub=${STUB:-none})"
nl_check_path "$BIN" "$GPY" "$RPY" "$HBIN" "$FF" "$MODEL_FILE" "$PROMPT" "$FA/four_proxy.py" "$FA/four_record.py" "$FA/four_mkhome.py" "$FA/four_race.sh" "$FA/four_render.py" \
  "$FA/four_gate.py" "$FA/four_results.py" "$FA/four_probe.py" "$FA/four_summary.py" "$V/lib/libasound.so.2" /home/witcheer/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome
SZ=$(stat -c %s "$MODEL_FILE" 2>/dev/null || echo 0)
if [ -z "${FOUR_MODEL_FILE:-}" ]; then
  [ "$SZ" = "$MODEL_BYTES" ] && nl_say "  ok: model file $SZ bytes" || nl_fail "model file size $SZ != $MODEL_BYTES"
else nl_say "  model file override (test): $MODEL_FILE ($SZ bytes)"; fi
"$GPY" -c "import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$FA/four_gate.py" "$FA/four_probe.py" "$FA/four_summary.py" "$FA/four_proxy.py" "$FA/four_record.py" "$FA/four_mkhome.py" || nl_fail "harness syntax"
"$RPY" -c "import ast,sys,pyte,PIL,fontTools; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$FA/four_render.py" "$FA/four_results.py" || nl_fail "renderer deps/syntax (pyte, PIL, fontTools in $RPY)"
bash -n "$FA/four_race.sh" || nl_fail "four_race.sh syntax"
HELP=$($BIN --help 2>&1 || true)
for f in --parallel --kv-unified --flash-attn --cache-type-k --metrics --slots --reasoning-budget-message --spec-type --spec-draft-n-max --predict; do case "$HELP" in *"$f"*) ;; *) nl_fail "flag $f not in this build";; esac; done
nl_say "  ok: server flags exist in build (parallel, kv-unified, flash-attn, cache-type, metrics, slots, reasoning-budget-message, spec-type)"
for f in $SAMPLING_FLAGS; do case "$f" in --*) case "$HELP" in *"$f"*) ;; *) nl_fail "flag $f not in this build";; esac;; esac; done
$BIN --version 2>&1 | grep -q "839a572" && nl_say "  ok: llama.cpp 839a572" || nl_fail "llama.cpp build changed or missing (expected 839a572)"
HV=$(cd "$CHECKOUT" && git rev-parse --short HEAD 2>/dev/null)
case "$HV" in "$HERMES_PIN"*) nl_say "  ok: Hermes checkout $HV (pin $HERMES_PIN)";; *) nl_fail "Hermes checkout $HV != pin $HERMES_PIN";; esac
# the compaction key must exist in the tree the agents run (read the SOURCE, not a doc): the cap is applied as min(ratio trigger, cap) and the 64,000 floor is why context_length stays 65536
grep -q 'threshold_tokens = cfg.get("threshold_tokens"' "$CHECKOUT/agent/agent_init.py" && grep -q "_effective_threshold_cap" "$CHECKOUT/agent/context_compressor.py" \
  && grep -q "MINIMUM_CONTEXT_LENGTH = 64_000" "$CHECKOUT/agent/model_metadata.py" || nl_fail "compression.threshold_tokens / MINIMUM_CONTEXT_LENGTH not found in the Hermes checkout"
nl_say "  ok: compression.threshold_tokens present in the pinned checkout (cap = min(ratio trigger, cap)); MINIMUM_CONTEXT_LENGTH 64,000 => context_length stays 65536"
PRT=$(mktemp -d /tmp/f80b-plan.XXXXXX)
FR_COMPACT_TOKENS=$([ "$COMPACT" = 0 ] && echo "" || echo "$COMPACT") python3 "$FA/four_mkhome.py" "$PRT" a 8191 "$ALIAS" 65536 > /dev/null 2>&1 \
  && { if [ "$COMPACT" = 0 ]; then ! grep -q threshold_tokens "$PRT/home-a/config.yaml"; else grep -qx "  threshold_tokens: $COMPACT" "$PRT/home-a/config.yaml" && grep -qx "  protect_first_n: 0" "$PRT/home-a/config.yaml" && grep -q "v2-compact threshold_tokens=$COMPACT protect_first_n=0" "$PRT/home-a/home-version.txt"; fi; } \
  && grep -qx "  context_length: 65536" "$PRT/home-a/config.yaml" && nl_say "  ok: agent home config proof ($(cat "$PRT/home-a/home-version.txt"))" || nl_fail "four_mkhome.py did not write the expected home config (compact=$COMPACT)"
rm -f "$PRT"/home-a/config.yaml "$PRT"/home-a/home-version.txt "$PRT"/home-a/SOUL.md "$PRT"/home-a/installs "$PRT"/home-a/tools; rmdir "$PRT/home-a" "$PRT" 2>/dev/null   # symlinks removed as links, targets untouched
[ "$COMPACT" = 0 ] || [ "$COMPACT" -ge 16000 ] || nl_fail "FOUR_COMPACT_TOKENS $COMPACT below 16000"
[ "$COMPACT" = 0 ] || [ "$COMPACT" -ge 32000 ] || nl_say "  WARNING: compaction trigger $COMPACT is under the largest first-turn context seen in item 80 (29.8k): a first-turn attempt is refused and Hermes waits 300 s before retrying (see CHANGE 2)"
for d in installs tools; do [ -d "/home/witcheer/.hermes/$d" ] || nl_fail "main Hermes root has no $d/ to share with the agent homes"; done
nl_say "  params: src='$PARAMS_SRC' flags='$SAMPLING_FLAGS'"
nl_check_disk 20
for p in $PORT 8191 8192 8193 8194; do ss -ltnH "sport = :$p" 2>/dev/null | grep -q . && nl_fail "port $p is already in use"; done
for a in a b c d; do tmux has-session -t "=f80-$a" 2>/dev/null && nl_fail "tmux session f80-$a already exists"; done
# the leak scan must not find the hidden check names in other queued items (n/a here: no hidden criteria); the prompt must carry no path of the account
grep -qE '/home/|/Users/|witcheer' "$PROMPT" && nl_fail "prompt file names a home path or the account"
if nl_skip_exact "$ART"; then nl_say "nothing to run, item complete ($ART exists)"; exit 0; fi
DL=$(nl_deadline_epoch); LEFT=$(( (DL-$(date +%s))/60 ))
nl_say "  planned RUN: $SLUG (ladder: $LADDER, free floor $FREE_FLOOR MiB; wall $WALL min + ~8 min load/probe/slack; card ~$((WALL+10)) min; hard stop $(date -d "@$DL" '+%a %H:%M') local, $LEFT min away)"
if [ "$SMOKE" != 1 ]; then
  [ "$LEFT" -ge 90 ] || { nl_say "  under 90 min left in the window: not starting (load + probe + $WALL min race + gates + clip would be cut short; exit 3, resumable)"; exit 3; }
fi
if [ "${DRYRUN:-0}" = "1" ] || [ ! -f "$MANIFEST" ]; then nl_manifest_init serve probe race gate render; fi
nl_gate

SRV=""; SRVLOG=""; DONALD_STOPPED=0; DONALD_BACK=0
killsrv() { { [ -n "$SRV" ] && kill -9 "$SRV" && wait "$SRV"; } 2>/dev/null; SRV=""; }                     # only the server THIS item started
donald_stop() { nl_say "draining Donald (sudo -n systemctl stop llama-server.service)"; sudo -n systemctl stop llama-server.service; DONALD_STOPPED=1; sleep 5; }
donald_restore() {
  [ "$DONALD_STOPPED" = 1 ] && [ "$DONALD_BACK" = 0 ] || return 0
  DONALD_BACK=1; nl_say "restoring Donald (sudo -n systemctl start llama-server.service)"
  sudo -n systemctl start llama-server.service
  for i in $(seq 1 60); do curl -s -m 5 localhost:8090/health | grep -qi ok && { nl_say "Donald healthy again after ~$((i*5)) s"; return 0; }; sleep 5; done
  nl_say "WARNING: Donald /health did not answer within 5 min after the restore"
}
cleanup() { for a in a b c d; do tmux kill-session -t "=f80-$a" 2>/dev/null; done; killsrv; sleep 3; donald_restore; }
trap cleanup EXIT
mkdir -p "$V/runs"
[ -d "$RUN" ] && mv "$RUN" "$RUN.stale-$(date +%s)"
mkdir -p "$RUN/out"
t0=$(date +%s)
rungfail() { nl_mark "$1" failed "$2"; nl_say "FAIL: $2"; exit 1; }

if [ -n "$STUB" ]; then
  UPSTREAM=$STUB; CTX_USED=stub; RUNG=stub
  nl_mark serve done "stub upstream $STUB"
else
  nl_mark serve running
  donald_stop
  for i in $(seq 1 60); do v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${v:-99999}" -lt 1024 ] && { nl_say "drain clear (${v} MiB)"; break; }; sleep 2; done
  [ "${v:-99999}" -lt 1024 ] || rungfail serve "card memory still ${v:-?} MiB after the drain: another process owns the card, not loading"
  UP=0; CTX_USED=""; RUNG=""; FIT="$RUN/fit.jsonl"
  for r in $LADDER; do
    c=${r%%:*}; mode=${r##*:}
    UNI=(); [ "$mode" = unified ] && UNI=(--kv-unified)
    SRVLOG=$RUN/server-$c-$mode.log
    nl_say "serve rung $r (parallel 4, ctx pool $c, per-agent window $((c/4)) if split)"
    # shellcheck disable=SC2086
    $BIN --model "$MODEL_FILE" --host 127.0.0.1 --port $PORT --n-gpu-layers 99 --ctx-size $c --parallel 4 "${UNI[@]}" --flash-attn on \
      --cache-type-k $KV --cache-type-v $KV --alias "$ALIAS" --metrics --reasoning-budget $RB --reasoning-budget-message "$RB_MSG" $SAMPLING_FLAGS "${EXTRA_ARR[@]}" "${PRED_ARR[@]}" \
      > "$SRVLOG" 2>&1 &
    SRV=$!
    for i in $(seq 1 180); do curl -s -m 3 127.0.0.1:$PORT/health | grep -qi ok && { UP=1; break; }; kill -0 $SRV 2>/dev/null || break; sleep 2; done
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); tot=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if [ "$UP" = 1 ]; then
      free=$(( tot - used ))
      echo "{\"rung\":\"$r\",\"up\":true,\"used_mib\":$used,\"free_mib\":$free,\"mtp\":\"${EXTRA_ARR[*]:-off}\"}" >> "$FIT"
      if [ "$free" -ge "$FREE_FLOOR" ]; then CTX_USED=$c; RUNG=$r; nl_say "  server up at rung $r: ${used} MiB used, ${free} MiB free"; break; fi
      nl_say "  rung $r loads but leaves only ${free} MiB free (< $FREE_FLOOR): next rung"; UP=0
    else
      echo "{\"rung\":\"$r\",\"up\":false,\"used_mib\":${used:-0},\"why\":\"$(grep -m1 -iE 'out of memory|failed|error' "$SRVLOG" | head -c 160 | tr '"' "'")\"}" >> "$FIT"
      nl_say "  rung $r did not come up: $(grep -m1 -iE 'out of memory|failed|error' "$SRVLOG" | head -c 200)"
    fi
    killsrv; sleep 5
  done
  [ -n "$RUNG" ] || rungfail serve "no ladder rung came up with >= $FREE_FLOOR MiB free (see $RUN/fit.jsonl)"
  UPSTREAM=http://127.0.0.1:$PORT
  curl -s 127.0.0.1:$PORT/props > "$RUN/props.json"
  SCHK=$(python3 - "$RUN/props.json" <<'PYEOF'
import json, sys
j = json.load(open(sys.argv[1])); g = j.get("default_generation_settings", {}); p = g.get("params", g)
want = {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.05}
got = {k: p.get(k) for k in want}
ok = all(got[k] is not None and abs(float(got[k]) - v) < 1e-3 for k, v in want.items()) and j.get("total_slots") == 4
print(("OK " if ok else "BAD ") + json.dumps(got) + " slots=" + str(j.get("total_slots")) + " n_ctx=" + str(g.get("n_ctx")))
PYEOF
)
  nl_say "  server sampling defaults: $SCHK"
  case "$SCHK" in OK*) ;; *) rungfail serve "server sampling/slots wrong: $SCHK";; esac
  nl_mark serve done "rung $RUNG, $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
fi

# ---- speed probe: one stream alone, then four at once, same server (the HUD's 'one agent alone' baseline) ----
nl_mark probe running
python3 "$FA/four_probe.py" "$UPSTREAM" "$ALIAS" "$RUN/speed-probe.json" >> "$RUN/probe.log" 2>&1
nl_say "  speed probe: $(python3 -c "import json;d=json.load(open('$RUN/speed-probe.json'));print('solo %.1f tok/s, four at once %.1f each, %.1f aggregate' % (d['solo_tps'], d['four_each_tps'], d['four_agg_tps']))" 2>/dev/null || echo 'failed, see probe.log')"
SOLO=$(python3 -c "import json;print(json.load(open('$RUN/speed-probe.json'))['solo_tps'])" 2>/dev/null || echo "")
nl_mark probe done "solo ${SOLO:-?}"

# ---- the race ----
DEADLINE=$(nl_deadline_epoch)
WALL_S=$(( WALL * 60 ))
MAXW=$(( DEADLINE - $(date +%s) - 1500 )); [ "$MAXW" -lt "$WALL_S" ] && [ "$MAXW" -gt 300 ] && WALL_S=$MAXW
nl_say "race: four agents, wall $WALL_S s, upstream $UPSTREAM"
nl_mark race running
tr0=$(date +%s)
POOLTOK=0
if [ "$GUARD_PCT" != 0 ]; then POOLTOK=$(( ${CTX_USED/stub/163840} * GUARD_PCT / 100 )); fi
nl_say "  agents: compaction trigger ${COMPACT} tokens + protect_first_n 0 (0 = v1 homes), proxy pool guard ${POOLTOK} tokens (0 = off), server --predict cap ${PRED_ARR[*]:-off}"
UPSTREAM=$UPSTREAM ALIAS=$ALIAS PROMPT=$PROMPT WALL_S=$WALL_S CTXCFG=65536 FR_PY=$RPY FR_CHECKOUT=$CHECKOUT FR_COMPACT_TOKENS=$([ "$COMPACT" = 0 ] && echo "" || echo "$COMPACT") FR_POOL=$([ "$POOLTOK" = 0 ] && echo "" || echo "$POOLTOK") bash "$FA/four_race.sh" "$RUN" >> "$RUN/race.log" 2>&1; rrc=$?
tail -8 "$RUN/race.log" | while read -r l; do nl_say "  race: $l"; done
curl -s -m 5 $UPSTREAM/metrics > "$RUN/server-metrics.txt" 2>/dev/null
curl -s -m 5 $UPSTREAM/slots > "$RUN/slots-end.json" 2>/dev/null
[ -z "$STUB" ] && nvidia-smi --query-gpu=memory.used --format=csv,noheader > "$RUN/vram-end.txt"
[ "$rrc" = 0 ] || { nl_mark race failed "four_race.sh rc=$rrc"; killsrv; donald_restore; nl_say "FAIL: race rc=$rrc (see $RUN/race.log)"; exit 1; }
nl_mark race done "$(( ($(date +%s)-tr0)/60 )) min"
# the card is free from here: server down, Donald back, then the CPU-only legs
killsrv; sleep 3
if [ -z "$STUB" ]; then donald_restore; fi
nl_say "card released after $(( ($(date +%s)-t0)/60 )) min (server + race), the rest is CPU"

# ---- gates (CPU) ----
nl_mark gate running
for a in a b c d; do
  mkdir -p "$RUN/gate/$a"
  if [ -f "$RUN/work/$a/index.html" ]; then
    setsid timeout 240 "$GPY" "$FA/four_gate.py" "$RUN/work/$a/index.html" "$RUN/gate/$a" --video-seconds 12 < /dev/null >> "$RUN/gate.log" 2>&1
  else
    echo '{"pass": false, "no_html": true}' > "$RUN/gate/$a/gate.json"; nl_say "  agent $a: no index.html in its work dir"
  fi
  nl_say "  gate $a: $(python3 -c "import json;d=json.load(open('$RUN/gate/$a/gate.json'));print('pass' if d.get('pass') else 'FAIL', d.get('bytes'), 'bytes', 'diff', d.get('frame_diff'), [k for k,v in (d.get('checks') or {}).items() if not v])")"
done
nl_mark gate done

# ---- clip (CPU) ----
nl_mark render running
SOLOARG=(); [ -n "$SOLO" ] && SOLOARG=(--solo-tps "$SOLO")
"$RPY" "$FA/four_render.py" "$RUN" "$RUN/out/race.mp4" --ffmpeg "$FF" --stills "$RUN/out/stills" "${SOLOARG[@]}" >> "$RUN/render.log" 2>&1; rr=$?
if [ "$rr" = 0 ]; then
  "$RPY" "$FA/four_results.py" "$RUN" "$RUN/out/results.mp4" --ffmpeg "$FF" --summary "$RUN/out/race.mp4.summary.json" --concat "$RUN/out/race.mp4" --final "$RUN/out/four-agents-final.mp4" >> "$RUN/render.log" 2>&1
fi
"$RPY" - "$RUN/out/stills" "$RUN/out/contact.png" <<'PYEOF' 2>>"$RUN/render.log"
import glob, sys
from PIL import Image
fs = sorted(glob.glob(sys.argv[1] + "/*.png"))[:9]
if fs:
    w, h = 640, 360
    sheet = Image.new("RGB", (w * 3, h * 3), "#000")
    for i, f in enumerate(fs):
        sheet.paste(Image.open(f).convert("RGB").resize((w, h)), ((i % 3) * w, (i // 3) * h))
    sheet.save(sys.argv[2])
PYEOF
nl_say "  render rc=$rr; final clip: $([ -f "$RUN/out/four-agents-final.mp4" ] && du -h "$RUN/out/four-agents-final.mp4" | cut -f1 || echo missing)"
python3 "$FA/four_summary.py" "$RUN" "$SLUG" "$WALL_S" "${RUNG:-stub}" "${EXTRA_ARR[*]:-off}" "$HV" "$(md5sum "$PROMPT" | cut -c1-12)" > "$RUN/summary.json" 2>>"$RUN/render.log"
nl_judge_artifact render 0 "$ART" "$t0" "$(( ($(date +%s)-t0)/60 )) min, render rc $rr" || { nl_say "FAIL: no fresh summary.json"; exit 1; }
nl_say "  summary: $(python3 -c "import json;d=json.load(open('$ART'));print(json.dumps(d.get('headline')))")"
nl_say "  RULE: $(python3 -c "import json;d=json.load(open('$ART'));r=d['success_rule'];print('PASS' if r['pass'] else 'FAIL', r['broken'])")"
nl_say "  compaction events per agent (state.db): $(python3 -c "import json;d=json.load(open('$ART'));print(d['compaction']['per_agent'], 'homes', d['compaction']['homes'])")"
nl_say "  per agent: $(python3 -c "import json;d=json.load(open('$ART'));print({a:(v['end'],v['peak_ctx_request_start'],v['peak_ctx_request_end'],v['tok_s_end_to_end'],v['failed_requests']) for a,v in d['per_agent'].items()}, 'pool_peak_est', d['pool_peak_est_tokens'], 'guard', d['pool_guard'])")"
nl_say "DONE $SLUG"
exit 0
