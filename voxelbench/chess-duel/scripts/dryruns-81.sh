#!/bin/bash
# dryruns-81.sh: DRYRUN matrix for item 81 (CPU only, throwaway manifest). logs: ~/voxel-stage/dry-81*.log
# green: in place now; window boundary 166 min ok / 164 min -> rc 3; red: missing model rc 2, leaked criterion name on the agent path rc 2, busy port rc 2, existing tmux name rc 2, bad flag rc 2
cd ~/voxel-stage || exit 1; I=~/voxel-stage/81-chess-duel.sh
export MANIFEST_OVERRIDE=/tmp/chess81-dry.json
NOW=$(date +%s)
DRYRUN=1 bash $I > dry-81.log 2>&1; echo "green (in place, real window) rc=$? (want 0, or 3 when fewer than 165 min are left in a block)"
NL_HARD_STOP_EPOCH=$(( NOW + 166*60 )) DRYRUN=1 bash $I > dry-81-window-ok.log 2>&1; echo "166 min left rc=$? (want 0)"
NL_HARD_STOP_EPOCH=$(( NOW + 164*60 )) DRYRUN=1 bash $I > dry-81-red-window.log 2>&1; echo "164 min left, dry rc=$? (want 3)"
NL_HARD_STOP_EPOCH=$(( NOW + 164*60 )) bash $I > dry-81-red-window-real.log 2>&1; echo "164 min left, REAL path rc=$? (want 3, before any drain)"
sed 's#^A_FILE=.*#A_FILE=/home/witcheer/models/qwen38-27b/NOPE.gguf; A_BYTES=1; A_ALIAS=x#' $I > /tmp/chess81-red1.sh; NL_HARD_STOP_EPOCH=$(( NOW + 200*60 )) DRYRUN=1 bash /tmp/chess81-red1.sh > dry-81-red-model.log 2>&1; echo "missing model rc=$? (want 2)"
sed 's#^A_SAMP=.*#A_SAMP="--temp 1.0 --bogus-flag 3"; A_EXTRA=()#' $I > /tmp/chess81-red2.sh; NL_HARD_STOP_EPOCH=$(( NOW + 200*60 )) DRYRUN=1 bash /tmp/chess81-red2.sh > dry-81-red-flag.log 2>&1; echo "bogus flag rc=$? (want 2)"
sed 's#A_BYTES=22884408288#A_BYTES=12345#' $I > /tmp/chess81-red3.sh; NL_HARD_STOP_EPOCH=$(( NOW + 200*60 )) DRYRUN=1 bash /tmp/chess81-red3.sh > dry-81-red-size.log 2>&1; echo "model size mismatch rc=$? (want 2)"
# leaked criterion name on a path the agent can read (a marker file in the visible tree, emptied after the test)
printf 'chess_final_%s\n' gate > ~/voxelbench/chess-duel/zz-leak-test-marker; NL_HARD_STOP_EPOCH=$(( NOW + 200*60 )) DRYRUN=1 bash $I > dry-81-red-leak.log 2>&1; echo "leaked name in the visible tree rc=$? (want 2)"; : > ~/voxelbench/chess-duel/zz-leak-test-marker
# a busy port and an existing session name
tmux new-session -d -s chess81-dryport "python3 -m http.server 8581 --bind 127.0.0.1 > /dev/null 2>&1"; sleep 2
NL_HARD_STOP_EPOCH=$(( NOW + 200*60 )) DRYRUN=1 bash $I > dry-81-red-port.log 2>&1; echo "port 8581 busy rc=$? (want 2)"; tmux kill-session -t =chess81-dryport
tmux new-session -d -s chess81cpu "sleep 25"; NL_HARD_STOP_EPOCH=$(( NOW + 200*60 )) DRYRUN=1 bash $I > dry-81-red-session.log 2>&1; echo "tmux chess81cpu exists rc=$? (want 2)"; tmux kill-session -t =chess81cpu
# EXPECT_RUN mismatch
EXPECT_RUN=1 NL_HARD_STOP_EPOCH=$(( NOW + 200*60 )) DRYRUN=1 bash $I > dry-81-red-expect.log 2>&1; echo "EXPECT_RUN=1 but 2 planned rc=$? (want 2)"
rm -f /tmp/chess81-red?.sh /tmp/chess81-dry.json
