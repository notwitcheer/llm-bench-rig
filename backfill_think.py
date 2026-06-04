#!/usr/bin/env python3
"""One-time (idempotent) backfill: write meta['think'] into existing result dirs
from the verified think-mode sets in lib/board.py. Runs going forward are
self-describing (bench.py records think), so this only matters for pre-existing
results. Run on each machine that holds a results/ mirror.

    python3 backfill_think.py            # writes results/*/meta.json
    python3 backfill_think.py --dry-run
"""
import json
import sys
from pathlib import Path

from lib.board import think_mode

RESULTS = Path("results")
dry = "--dry-run" in sys.argv

changed = skipped = unknown = 0
for d in sorted(RESULTS.iterdir()):
    mf = d / "meta.json"
    if not (d.is_dir() and mf.exists()):
        continue
    meta = json.loads(mf.read_text())
    tm = think_mode(meta, d.name)
    if tm is None:
        print(f"  UNKNOWN think mode (left as-is): {d.name}")
        unknown += 1
        continue
    if meta.get("think") == tm:
        skipped += 1
        continue
    meta["think"] = tm
    if not dry:
        mf.write_text(json.dumps(meta, indent=2))
    print(f"  {'would set' if dry else 'set'} think={tm}  {d.name}")
    changed += 1

print(f"\n{'[dry-run] ' if dry else ''}changed={changed} unchanged={skipped} unknown={unknown}")
