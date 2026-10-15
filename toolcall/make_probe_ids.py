"""Pick the probe items (5 per set) from the frozen toolcall-v1 manifest, seeded. Probe items are a
plumbing/pace sample, not a score. usage: python make_probe_ids.py <manifest.json> <out.json>"""
import json
import random
import sys

SEED = 20261015 + 100
man = json.load(open(sys.argv[1]))
items = man["items"]
pick = {}
pools = {
    "S1": [d["id"] for d in items if "S1" in d["sets"]],
    "S2": [d["id"] for d in items if "S2" in d["sets"] and "S1" not in d["sets"]],
    "S3": [d["id"] for d in items if "S3" in d["sets"]],
    "S4": [d["id"] for d in items if "S4" in d["sets"]],
}
for k, (s, pool) in enumerate(pools.items()):
    pick[s] = random.Random(SEED + k).sample(pool, 5)
# S5: one or two per stratum from phase 1, shortest stratum first so a budget cut still leaves finished entries
s5 = []
for k, (st, n) in enumerate((("<=4", 2), ("5-8", 2), (">=9", 1))):
    pool = [d["id"] for d in items if "S5" in d["sets"] and d.get("stratum") == st]
    s5 += random.Random(SEED + 10 + k).sample(pool, n)
pick["S5"] = s5
json.dump(pick, open(sys.argv[2], "w"), indent=1)
print(json.dumps(pick))
