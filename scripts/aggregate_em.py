"""Aggregate the t073 eval matrix into the decomposition: how the base->EM gain splits into
eval-protocol artifact, format-elicitation, and genuine EM. Reads the graded results/em/*.json
(not the *.gens.json)."""
import glob
import json

R = {}
for f in glob.glob("results/em/*.json"):
    if f.endswith(".gens.json"):
        continue
    d = json.load(open(f))
    R[(d["label"], d["variant"], d["benchmark"])] = d


def acc(lab, var, bm):
    return R.get((lab, var, bm), {}).get("acc", float("nan"))


for bm in ["math500", "amc23"]:
    bp, bfair, bfmt = acc("base", "paper", bm), acc("base", "fair", bm), acc("base", "format", bm)
    em10, em15 = acc("em-step10", "paper", bm), acc("em-step15", "paper", bm)
    print(f"\n[{bm}]  base/paper={bp:.1f}  base/fair={bfair:.1f}  base/format={bfmt:.1f}  "
          f"EM@10={em10:.1f}  EM@15={em15:.1f}")
    print(f"  decomposition vs EM@10:  protocol={bfair - bp:+.1f}  format={bfmt - bfair:+.1f}  "
          f"genuine-EM={em10 - bfmt:+.1f}  total={em10 - bp:+.1f}")
    for lab in ["base", "em-step10", "em-step15"]:
        d = R.get((lab, "paper", bm))
        if d:
            print(f"  {lab}/paper: acc={d['acc']:.1f}  rep={d['rep_rate']:.3f}  len={d['mean_len']:.0f}")
