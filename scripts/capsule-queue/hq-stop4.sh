#!/usr/bin/env bash
# stop hmrun4 right after its q_nv_humming leg (operator card-sharing pause 10:30 CEST); its EXIT trap restores Donald.
L=/home/witcheer/humming-run4.log
until grep -q "quality q_nv_humming done" "$L"; do tmux has-session -t =hmrun4 2>/dev/null || exit 0; sleep 2; done
P=$(pgrep -f "^bash /home/witcheer/humming-run4.sh" | head -1)
echo "[hqstop] $(date +%T) q_nv_humming done, TERM main pid $P" | tee -a "$L"
kill -TERM $P
