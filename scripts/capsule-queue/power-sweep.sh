#!/bin/bash
# power-sweep.sh: RTX 5090 power-limit sweep on Donald's model (2026-10-22, mercury; operator yes in the GPU thread).
# question: what does an 80% power cap (and 500 / 400 W) cost in prefill and decode, and how many watts does it save?
# run BY THE OPERATOR, once, from an ssh terminal (needs his sudo password for nvidia-smi -pl):
#     sudo bash /home/witcheer/power-sweep/power-sweep.sh
# order 575 -> 460 -> 400 -> 500 -> 575 (the second 575 is the drift check). per limit: llama-bench pp512 and tg128 at
# depth 0, then pp512 and tg128 at depth 32768, 3 reps each, separate invocations so each gets its own power samples.
# safety: refuses inside the idle-queue blocks / the 08:00 brief / while another GPU job runs; EXIT trap puts the limit
# back to 575 W and restarts Donald's service, then checks :8090/health.
set -uo pipefail
[ "$(id -u)" = 0 ] || { echo "run it with sudo: sudo bash $0"; exit 2; }
U=witcheer; H=/home/$U
BIN=$H/llama.cpp-bolt/build/bin/llama-bench
M=$H/models/qwen38-27b/Qwen3.8-27B-Q6_K.gguf
OUT=$H/power-sweep/run-$(date +%m%d-%H%M); mkdir -p "$OUT"
LOG=$OUT/sweep.log
say(){ echo "[psweep $(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
DRY=${DRYRUN:-0}
hm=${HM_OVERRIDE:-$(date +%H%M)}   # HM_OVERRIDE only for guard tests
inwin(){ [ "$hm" -ge "$1" ] && [ "$hm" -lt "$2" ]; }
if inwin 0105 0615 || inwin 1405 1715 || inwin 0755 0815; then say "REFUSED: inside an idle-queue block or the 08:00 brief window ($hm). try after 17:15."; exit 3; fi
[ -x "$BIN" ] && [ -f "$M" ] || { say "REFUSED: missing $BIN or $M"; exit 2; }
DPID=$(systemctl show -p MainPID --value llama-server.service 2>/dev/null)
kids=$(pgrep -P "${DPID:-0}" 2>/dev/null | tr '\n' ' ')
foreign=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | while read p; do case " $DPID $kids " in *" $p "*) ;; *) echo "$p";; esac; done)
[ -z "$foreign" ] || { say "REFUSED: another GPU job holds the card (pids: $foreign)"; exit 4; }
busy=$(sudo -u $U tmux ls 2>/dev/null | cut -d: -f1 | grep -E '^(idleq|vx-|voxel-|prz|dgz|hkv4|flint)' | tr '\n' ' ')
[ -z "$busy" ] || { say "REFUSED: busy tmux sessions: $busy"; exit 4; }
[ -f $H/sparkagent-BUSY ] && { say "REFUSED: ~/sparkagent-BUSY present"; exit 4; }
say "preflight ok. model $(basename $M) ($(stat -c %s $M) B), bench $($H/llama.cpp-bolt/build/bin/llama-server --version 2>&1 | grep -m1 version)"
[ "$DRY" = 1 ] && { say "DRYRUN ok, nothing changed"; exit 0; }

restore(){
  nvidia-smi -pl 575 >/dev/null 2>&1; say "power limit back to $(nvidia-smi --query-gpu=power.limit --format=csv,noheader)"
  [ -n "${SP:-}" ] && kill $SP 2>/dev/null
  systemctl start llama-server.service; sleep 5
  say "Donald :8090/health -> $(curl -s -m5 http://127.0.0.1:8090/health | head -c 40)"
  chown -R $U:$U "$H/power-sweep"
  say "SWEEP_END (results in $OUT)"
}
trap restore EXIT
systemctl stop llama-server.service
for _ in $(seq 1 60); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '); [ "$m" -lt 1024 ] && break; sleep 2; done
say "Donald drained, card ${m} MiB"
[ "$m" -lt 1024 ] || { say "card did not free, abort"; exit 5; }
# 200 ms sampler for the whole sweep: epoch_ms, power W, SM MHz, mem MHz, temp C, util %
nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,power.limit --format=csv,noheader,nounits -lms 200 > "$OUT/samples.csv" 2>&1 & SP=$!
run(){ # $1 limit  $2 tag  $3.. bench args
  local pl=$1 tag=$2; shift 2
  local t0=$(date +%s.%N)
  sudo -u $U "$BIN" -m "$M" -ngl 99 -fa on -r 3 -o jsonl "$@" > "$OUT/${pl}W-$tag.jsonl" 2> "$OUT/${pl}W-$tag.err"
  local rc=$? t1=$(date +%s.%N)
  echo "$pl $tag $t0 $t1 $rc" >> "$OUT/windows.txt"
  say "  ${pl} W $tag rc=$rc $(python3 -c "import json,sys;r=[json.loads(l) for l in open('$OUT/${pl}W-$tag.jsonl') if l.strip()];print(' '.join(f\"{x.get('avg_ts',0):.1f}t/s\" for x in r))" 2>/dev/null)"
}
n=0
for pl in 575 460 400 500 575; do
  n=$((n+1)); nvidia-smi -pl $pl >/dev/null || { say "nvidia-smi -pl $pl failed"; exit 6; }
  sleep 3; say "step $n: limit $(nvidia-smi --query-gpu=power.limit --format=csv,noheader)"
  tagb=""; [ $n = 5 ] && tagb="-b"
  run $pl pp512-d0$tagb  -p 512 -n 0 -d 0
  run $pl tg128-d0$tagb  -p 0 -n 128 -d 0
  run $pl pp512-d32k$tagb -p 512 -n 0 -d 32768
  run $pl tg128-d32k$tagb -p 0 -n 128 -d 32768
done
say "SWEEP_DONE"
