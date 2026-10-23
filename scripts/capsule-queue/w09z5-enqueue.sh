#!/bin/bash
# w09z5-enqueue.sh: moves 09z5-duck-velslope into ~/idle-queue only AFTER the thu 2026-10-15 14:10-17:10 block, so 09z4 alone
# rides that duck block and the duck lane still has an item queued once 09z4 exports. by the lane alternation fri 01:10 is
# bench (09z5 yields there, ly = fri 06:10), so 09z5 trains fri 2026-10-16 14:10. run in tmux "w09z5" with "nogpu" in the start
# command (the runner ignores such sessions). log ~/Pollen/w09z5-enqueue.log. sibling of w09z4-enqueue.sh.
exec >> /home/witcheer/Pollen/w09z5-enqueue.log 2>&1
Q=/home/witcheer/idle-queue; SRC=/home/witcheer/Pollen/queue-staging/09z5-duck-velslope.sh; DST=$Q/09z5-duck-velslope.sh
GO=$(date -d '2026-10-15 17:12' +%s)
echo "$(date -u +%FT%TZ) watcher start, enqueue at $(date -d @$GO)"
while [ "$(date +%s)" -lt "$GO" ]; do sleep 60; done
# never enqueue into a live drain mid-block (the 17:10 hard stop should have ended it; wait up to 10 min more if not)
for i in $(seq 1 20); do tmux has-session -t =idleq 2>/dev/null || break; sleep 30; done
[ -f "$DST" ] && { echo "$(date -u +%FT%TZ) already queued, nothing to do"; exit 0; }
cp $SRC $DST.new && bash -n $DST.new && mv $DST.new $DST && echo "$(date -u +%FT%TZ) 09z5 queued: $(ls $Q)" || echo "$(date -u +%FT%TZ) ENQUEUE FAILED"
