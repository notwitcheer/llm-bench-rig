#!/usr/bin/env python3
"""rc_select.py <out.json> [--tids a,b,c] [--held N] [--near N]: the frozen selection (ids only, never question text).
Same RNG draws as the real selection; --tids filters it (mock runs). The real selection is written once and sha256-frozen."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rc_common as C

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--tids", default="")
ap.add_argument("--held", type=int, default=10)
ap.add_argument("--near", type=int, default=10)
a = ap.parse_args()
sel = C.make_selection(20, a.held, a.near)
if a.tids:
    keep = [int(x) for x in a.tids.split(",")]
    sel["templates"] = [t for t in sel["templates"] if t["tid"] in keep]
    sel["mock_subset"] = True
json.dump(sel, open(a.out, "w"), indent=1)
print("selection:", len(sel["templates"]), "templates,", sum(len(t["held"]) for t in sel["templates"]), "held-out,", sum(len(t["near"]) for t in sel["templates"]), "near-miss ->", a.out)
