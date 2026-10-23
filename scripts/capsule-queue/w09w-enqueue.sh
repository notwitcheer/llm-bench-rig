#!/bin/bash
# w09w-enqueue.sh: moves 09w into ~/idle-queue only once 09v has left the queue AND the drain has handed the card to another
# item (59's last leg), so 09w cannot jump ahead of 59 in the tue 14:10 block. run in tmux "w09w". log ~/Pollen/w09w-enqueue.log
exec >> /home/witcheer/Pollen/w09w-enqueue.log 2>&1
Q=/home/witcheer/idle-queue; SRC=/home/witcheer/Pollen/queue-staging/09w-duck-carpetbl.sh
echo "$(date -u +%FT%TZ) watcher start"
while [ -f $Q/09v-duck-carpetdeep.sh ]; do sleep 30; done
echo "$(date -u +%FT%TZ) 09v left the queue"; sleep 60
for i in $(seq 1 60); do
  a=$(grep -o '"script":"[^"]*"' /home/witcheer/idleq-active.json 2>/dev/null)
  case "$a" in *09v-duck-carpetdeep*|"") sleep 30;; *) break;; esac
done
echo "$(date -u +%FT%TZ) active: $a"
cp $SRC $Q/09w-duck-carpetbl.sh.new && bash -n $Q/09w-duck-carpetbl.sh.new && mv $Q/09w-duck-carpetbl.sh.new $Q/09w-duck-carpetbl.sh && echo "$(date -u +%FT%TZ) 09w queued" || echo "$(date -u +%FT%TZ) ENQUEUE FAILED"
