#!/bin/bash
# usage: runsweep.sh out N mode alpha start steps win "etas" "seeds"
out=$1; N=$2; mode=$3; al=$4; start=$5; steps=$6; win=$7; etas=$8; seeds=$9
for eta in $etas; do for seed in $seeds; do echo "$N 10 $al $seed $eta $start $steps $win $mode"; done; done | nice -n 19 xargs -P 8 -L1 bash -c "r=\$(./vicp \$0 \$1 \$2 \$3 \$4 \$5 \$6 \$7 \$8); echo \"mode=\$8 N=\$0 alpha=\$2 seed=\$3 eta=\$4 start=\$5 steps=\$6 -> \$r\"" >> $out
echo "DONE $out" >> $out
