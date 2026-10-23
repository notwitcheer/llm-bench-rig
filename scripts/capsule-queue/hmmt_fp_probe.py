import collections
from datasets import load_dataset
from lib.evals.matharena import grade_with_timeout
ds = load_dataset("MathArena/hmmt_feb_2026_outputs", split="train")
print("cols", ds.column_names[:20])
fp = collections.Counter()
for r in ds:
    if bool(r["correct"]): continue
    pa = r.get("parsed_answer"); pa = None if pa in (None, "", "None") else str(pa)
    ours, to = grade_with_timeout(pa, str(r["gold_answer"]), secs=10)
    if ours: fp[(r.get("problem_idx"), str(r["gold_answer"])[:60], (pa or "")[:90])] += 1
print("FP total", sum(fp.values()), "distinct", len(fp))
for k, v in fp.most_common(): print(v, k)
