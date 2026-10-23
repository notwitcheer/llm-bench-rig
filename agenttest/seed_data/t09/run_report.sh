#!/bin/bash
set -e
INPUT={WORK}/t09/data/input.txt
OUT={WORK}/t09/out
[ -f "$INPUT" ] || { echo "error: input file $INPUT not found" >&2; exit 1; }
[ -d "$OUT" ] || { echo "error: output directory $OUT does not exist" >&2; exit 1; }
echo "report ok $(cat $INPUT)" > $OUT/report.txt
echo done
