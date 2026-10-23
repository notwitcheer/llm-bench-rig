#!/bin/bash
# w09z7-enqueue.sh: moves 09z7-duck-velmixslope into ~/idle-queue only once 66y-voxel-muse has STARTED (09z7 sorts before 66y in
# `ls ~/idle-queue/*.sh | sort`, and 66y must run first at thu 2026-10-22 14:10: its 15:25 readout cron depends on it).
# why this is safe: idle-queue-drain.sh writes ~/idleq-active.json for an item right before its DRYRUN and adds it to its "tried"
# list; after each item it re-globs ~/idle-queue/*.sh | sort and runs the first untried one, so a 09z7 dropped in while 66y runs
# trains right after 66y in the same block (if >= 30 min are left), never before it.
# started = idleq-active.json names 66y with "started" at/after this block's 14:10 (12:10Z), or 66y in done/; a 66y that leaves the
# queue without either (moved out by a sibling session) also releases 09z7, but only after 14:12 local today.
# 66y never starting today (runner skipped the block) -> keep waiting across blocks: 09z7 still enters only after 66y starts.
# gives up sat 2026-10-24 00:05 local. liveness: a heartbeat line every 30 min in the log + touch ~/Pollen/w09z7.alive every loop.
# run in tmux "w09z7" with "nogpu" in the start command (the runner ignores such sessions). log ~/Pollen/w09z7-enqueue.log
exec >> /home/witcheer/Pollen/w09z7-enqueue.log 2>&1
Q=/home/witcheer/idle-queue; SRC=/home/witcheer/Pollen/queue-staging/09z7-duck-velmixslope.sh; DST=$Q/09z7-duck-velmixslope.sh
Y=66y-voxel-muse; GO=$(date -d '2026-10-22 14:12' +%s); STOP=$(date -d '2026-10-24 00:05' +%s); SINCE=2026-10-22T12:10
echo "$(date -u +%FT%TZ) watcher start (pid $$): waiting for $Y to start (active.json started >= ${SINCE}Z) or leave the queue after 14:12"
hb=0; why=""
while :; do
  now=$(date +%s); touch /home/witcheer/Pollen/w09z7.alive
  [ "$now" -lt "$STOP" ] || { echo "$(date -u +%FT%TZ) GIVE UP: $Y never started by sat 00:05, 09z7 NOT queued"; exit 1; }
  [ $(( now - hb )) -ge 1800 ] && { echo "$(date -u +%FT%TZ) alive, $Y in queue: $([ -f $Q/$Y.sh ] && echo yes || echo no)"; hb=$now; }
  a=$(cat /home/witcheer/idleq-active.json 2>/dev/null)
  st=$(echo "$a" | grep -o '"started":"[^"]*"' | cut -d'"' -f4)
  case "$a" in *"$Y.sh"*) [ -n "$st" ] && [[ "$st" > "$SINCE" || "$st" == "$SINCE"* ]] && { why="active.json: $Y started $st"; break; };; esac
  [ -f "$Q/done/$Y.sh" ] && { why="$Y in done/"; break; }
  [ "$now" -ge "$GO" ] && [ ! -f "$Q/$Y.sh" ] && { why="$Y left the queue without a start record (moved by hand?)"; break; }
  sleep 20
done
echo "$(date -u +%FT%TZ) release: $why"; sleep 60
[ -f "$DST" ] && { echo "$(date -u +%FT%TZ) already queued, nothing to do"; exit 0; }
cp $SRC $DST.new && bash -n $DST.new && mv $DST.new $DST && echo "$(date -u +%FT%TZ) 09z7 queued: $(ls $Q/*.sh | sort | xargs -n1 basename | tr '\n' ' ')" || echo "$(date -u +%FT%TZ) ENQUEUE FAILED"
