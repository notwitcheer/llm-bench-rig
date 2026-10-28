#!/usr/bin/env python3
"""print the L1 rows for the queue item (bash reads them): one pipe-separated line per row, in run order.
fields: slug|path|bytes|arch|est_min|temp|top_p|top_k|min_p|moe_ladder|profile|label
usage: row_info.py [--rows rows.json] [--only slug1,slug2]   |   row_info.py --total-est"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--rows", default=os.path.join(HERE, "rows.json"))
ap.add_argument("--only", default="")
ap.add_argument("--total-est", action="store_true")
a = ap.parse_args()
d = json.load(open(a.rows))
only = set(x for x in a.only.split(",") if x)
rows = [r for r in d["rows"] if not only or r["slug"] in only]
if a.total_est:
    print(sum(r["est_min"] for r in rows))
    raise SystemExit
for r in rows:
    prof = "at2" + "".join(c for c in r["slug"] if c.isalnum())
    assert len(prof) <= 40 and prof.isalnum(), prof
    print("|".join(str(x) for x in (r["slug"], r["path"], r["bytes"], r["arch"], r["est_min"], r["temp"], r["top_p"], r["top_k"], r["min_p"], r["moe_ladder"], prof, r["label"])))
