#!/bin/bash
# w09z4-enqueue.sh: moves 09z4-duck-velpush into ~/idle-queue only AFTER the wed 2026-10-14 14:10-17:10 block, so 09z3 alone
# rides today's duck block and the duck lane still has an item queued once 09z3 exports. by the lane alternation thu 01:10 is
# bench, so 09z4 trains thu 14:10. run in tmux "w09z4" with "nogpu" in the start command (the runner ignores such sessions).
# log ~/Pollen/w09z4-enqueue.log. sibling of w09z3-enqueue.sh.
exec >> /home/witcheer/Pollen/w09z4-enqueue.log 2>&1
Q=/home/witcheer/idle-queue; SRC=/home/witcheer/Pollen/queue-staging/09z4-duck-velpush.sh; DST=$Q/09z4-duck-velpush.sh
GO=$(date -d '2026-10-14 17:12' +%s)
echo "$(date -u +%FT%TZ) watcher start, enqueue at $(date -d @$GO)"
while [ "$(date +%s)" -lt "$GO" ]; do sleep 60; done
# never enqueue into a live drain mid-block (the 17:10 hard stop should have ended it; wait up to 10 min more if not)
for i in $(seq 1 20); do tmux has-session -t =idleq 2>/dev/null || break; sleep 30; done
[ -f "$DST" ] && { echo "$(date -u +%FT%TZ) already queued, nothing to do"; exit 0; }
cp $SRC $DST.new && bash -n $DST.new && mv $DST.new $DST && echo "$(date -u +%FT%TZ) 09z4 queued: $(ls $Q)" || echo "$(date -u +%FT%TZ) ENQUEUE FAILED"
