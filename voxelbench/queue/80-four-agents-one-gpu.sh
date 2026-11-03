#!/bin/bash
# 80-four-agents-one-gpu.sh: idle-queue item (STAGED in ~/voxel-stage, not enqueued; 2026-11-03, mercury subagent for the GPU thread, mojo slate F4 pick 1 = decision B 1, 2026-11-02).
# operator, GPU thread, 2026-11-02 ~19:35 CET: decision B 1: "four Hermes agents, one GPU, the same task, live: llama.cpp parallel slots, four panes racing in real time,
#   aggregate tok/s on screen. hook: one card serving a whole team of agents." (vault 99-personal/benchmark-rig/mojo-slate-2026-11-02.md)
# MOJO: four Hermes Agent sessions on one RTX 5090 building the same thing at once | intense/never-seen | one RTX 5090 is serving four Hermes Agents at once: same prompt, same local Qwen3.8-27B, four panes racing, [N] tok/s combined on the card
# EV: line 1 above (the [N] is the measured mean aggregate); rules: gpu-ev-gate-2026-10-25 section 2 rule 4 (a reproducible, fully local Hermes piece: one card, one GGUF, four
#   private Hermes homes, scripts in voxelbench/four-agents) + mojo hook intense / never-seen (a race on screen, aggregate tok/s live; checked 2026-11-03: llama.cpp --parallel is
#   documented, but a filmed four-session Hermes race with a live aggregate HUD was not found on our account). comparables ("Hermes Agent on a local model" posts, views, from
#   99-personal/benchmark-rig/gpu-ev-gate-2026-10-25.md section 1): 14,324 / 7,850 / 5,476 / 5,168 / 4,640; voxel sequels 1.7-2.6k. bracket 4-10k. HONEST LIMITS: one race = four
#   samples of one model, so no 'agent X is better' claim; per-agent tok/s here is NOT comparable to the single-slot series numbers (parallel slots, pooled KV); a page that
#   fails the liveness gate is shown in the results tile as FAIL, not hidden. the picture is the hook: pair it with the which-card angle ("one card, a team of agents"), never a speed brag.
# serves: content (a Hermes-driven showcase: the card as a team server) + hardening (a parallel-slot Hermes harness that never touches the main hermes-gateway.service).
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
#   context_length 65536 (Hermes compacts at 50% of it, about 33k, so four agents sit near 4 x 35k).
#   agents: `<checkout>/.hermes/bin/hermes chat --cli --toolsets terminal,file` with HERMES_HOME=<own>, HOME=<throwaway>, HERMES_DISABLE_LAZY_INSTALLS=1 and the main root's installs/ + tools/ symlinked into
#   the home (without the symlinks a new data root builds its own runtime: python-olm cmake, minutes, then fails; the checkout's old venv lacks modules, every tool call died on snowballstemmer).
# REGIME (label every number and clip with it): parallel 4 on one server, pooled KV; per-agent speed is NOT the single-slot speed of the series; plain first-turn prompt, not /goal (a finish time
#   is then defined: the pane returns to its prompt and the proxy shows no request in flight for 20 s); toolsets terminal + file only (no delegation: a sub-agent would add requests the HUD
#   cannot attribute); thinking on, RB 16384; approvals off; memory off; temperature 1.0 so the four runs differ (that is the point of a race).
# resumable: needs >= 90 min left in the window (load + probe + WALL min race + gates + clip, about 60 min worst case with WALL 40) or exits 3 (stays queued).
# card time: about 50 min (load 2 + probe 3 + race <= WALL + 3 barrier/slack + restore 1); the gates and the render run after Donald is back (CPU only).
# artifact: runs/<slug>/summary.json (exact). bench homes are throwaway (/tmp/arena80); the recorder/proxy/server pids are this item's own, never killed by name.
# test knobs (empty in production): FOUR_SMOKE=1 (slug -smoke, WALL 7 min, no 90-min rule; run it by hand in tmux with NL_HARD_STOP_EPOCH set), FOUR_WALL_MIN, FOUR_LADDER ("ctx:mode ..." mode = unified|split),
#   FOUR_NO_MTP=1, FOUR_MODEL_FILE (red-path test), FOUR_STUB_UPSTREAM=http://127.0.0.1:PORT (CPU stub: no Donald drain, no server start, the race runs against a mock; DRYRUN-safe).
set -uo pipefail
NL_TAG=voxfour80
MANIFEST=/home/witcheer/voxel-voxfour80-status.json
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
RB=16384
RB_MSG="Thinking budget reached. Stop planning and write the files now with your tools; refine in later turns."
LADDER=${FOUR_LADDER:-"163840:unified 131072:unified"}    # smoke 2026-11-03: 163840 loads at 30,408 MiB (2,199 free), race peak 30,508; 196608 loads at 31,816 (791 free: too tight); 4 x 65536 split fails to fit
WALL=${FOUR_WALL_MIN:-40}
SLUG=four80-qwen38-q6k-par4
SMOKE=${FOUR_SMOKE:-0}
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
for f in --parallel --kv-unified --flash-attn --cache-type-k --metrics --slots --reasoning-budget-message --spec-type --spec-draft-n-max; do case "$HELP" in *"$f"*) ;; *) nl_fail "flag $f not in this build";; esac; done
nl_say "  ok: server flags exist in build (parallel, kv-unified, flash-attn, cache-type, metrics, slots, reasoning-budget-message, spec-type)"
for f in $SAMPLING_FLAGS; do case "$f" in --*) case "$HELP" in *"$f"*) ;; *) nl_fail "flag $f not in this build";; esac;; esac; done
$BIN --version 2>&1 | grep -q "839a572" && nl_say "  ok: llama.cpp 839a572" || nl_fail "llama.cpp build changed or missing (expected 839a572)"
HV=$(cd "$CHECKOUT" && git rev-parse --short HEAD 2>/dev/null)
case "$HV" in "$HERMES_PIN"*) nl_say "  ok: Hermes checkout $HV (pin $HERMES_PIN)";; *) nl_fail "Hermes checkout $HV != pin $HERMES_PIN";; esac
for d in installs tools; do [ -d "/home/witcheer/.hermes/$d" ] || nl_fail "main Hermes root has no $d/ to share with the agent homes"; done
nl_say "  params: src='$PARAMS_SRC' flags='$SAMPLING_FLAGS'"
nl_check_disk 20
for p in $PORT 8191 8192 8193 8194; do ss -ltnH "sport = :$p" 2>/dev/null | grep -q . && nl_fail "port $p is already in use"; done
for a in a b c d; do tmux has-session -t "=f80-$a" 2>/dev/null && nl_fail "tmux session f80-$a already exists"; done
# the leak scan must not find the hidden check names in other queued items (n/a here: no hidden criteria); the prompt must carry no path of the account
grep -qE '/home/|/Users/|witcheer' "$PROMPT" && nl_fail "prompt file names a home path or the account"
if nl_skip_exact "$ART"; then nl_say "nothing to run, item complete ($ART exists)"; exit 0; fi
DL=$(nl_deadline_epoch); LEFT=$(( (DL-$(date +%s))/60 ))
nl_say "  planned RUN: $SLUG (ladder: $LADDER; wall $WALL min + ~8 min load/probe/slack; card ~$((WALL+10)) min; hard stop $(date -d "@$DL" '+%a %H:%M') local, $LEFT min away)"
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
      --cache-type-k $KV --cache-type-v $KV --alias "$ALIAS" --metrics --reasoning-budget $RB --reasoning-budget-message "$RB_MSG" $SAMPLING_FLAGS "${EXTRA_ARR[@]}" \
      > "$SRVLOG" 2>&1 &
    SRV=$!
    for i in $(seq 1 180); do curl -s -m 3 127.0.0.1:$PORT/health | grep -qi ok && { UP=1; break; }; kill -0 $SRV 2>/dev/null || break; sleep 2; done
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); tot=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if [ "$UP" = 1 ]; then
      free=$(( tot - used ))
      echo "{\"rung\":\"$r\",\"up\":true,\"used_mib\":$used,\"free_mib\":$free,\"mtp\":\"${EXTRA_ARR[*]:-off}\"}" >> "$FIT"
      if [ "$free" -ge 800 ]; then CTX_USED=$c; RUNG=$r; nl_say "  server up at rung $r: ${used} MiB used, ${free} MiB free"; break; fi
      nl_say "  rung $r loads but leaves only ${free} MiB free (< 800): next rung"; UP=0
    else
      echo "{\"rung\":\"$r\",\"up\":false,\"used_mib\":${used:-0},\"why\":\"$(grep -m1 -iE 'out of memory|failed|error' "$SRVLOG" | head -c 160 | tr '"' "'")\"}" >> "$FIT"
      nl_say "  rung $r did not come up: $(grep -m1 -iE 'out of memory|failed|error' "$SRVLOG" | head -c 200)"
    fi
    killsrv; sleep 5
  done
  [ -n "$RUNG" ] || rungfail serve "no ladder rung came up with >= 800 MiB free (see $RUN/fit.jsonl)"
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
UPSTREAM=$UPSTREAM ALIAS=$ALIAS PROMPT=$PROMPT WALL_S=$WALL_S CTXCFG=65536 FR_PY=$RPY FR_CHECKOUT=$CHECKOUT bash "$FA/four_race.sh" "$RUN" >> "$RUN/race.log" 2>&1; rrc=$?
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
nl_say "DONE $SLUG"
exit 0
