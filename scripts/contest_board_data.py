#!/usr/bin/env python3
"""build contest-board data (AIME 2026 + HMMT Feb 2026, think-on 32k, veto on) + MATH-500 per row.
runs on the rig from the repo root; prints json to stdout. a leg with no detail json is read from its progress json and flagged final=false."""
import json, os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
ROWS = [  # slug, label, math500 dir suffix
    ("gemma-4-31b-q4-0-it", "Gemma 4 31B Q4_0", "-thinkon"),
    ("qwen3-8-27b-q6-k", "Qwen3.8-27B Q6_K", "-thinkon"),
    ("qwen3-8-27b-ud-iq3-xxs", "Qwen3.8-27B UD-IQ3_XXS (12 GB)", "-thinkon"),
    ("ornith-1-5-35b-q4-k-m", "Ornith 1.5 35B Q4_K_M", "-thinkon"),
    ("qwen3-6-27b-q6-k", "Qwen3.6-27B Q6_K", "-thinkon-32k"),
    ("qwen3-6-35b-a3b-ud-q5-k-m", "Qwen3.6-35B-A3B UD-Q5_K_M", "-thinkon-32k"),
    ("nvidia-nemotron-3-5-lightning-30b-a3b-q4-k-m", "Nemotron 3.5 Lightning 30B-A3B Q4_K_M", "-thinkon"),
    ("ornith-9b", "Ornith 1.5 9B Q6_K", "-thinkon"),
    ("qwable-27b-q4-k-m", "Qwable 27B Q4_K_M", "-thinkon-32k"),
]
TOTALS = {"ma_aime_2026": 30, "ma_hmmt_feb_2026": 33}
out = []
for slug, label, msuf in ROWS:
    d = f"{R}/{slug}-matharena-32k"
    row = {"slug": slug, "label": label, "legs": {}}
    for s, tot in TOTALS.items():
        prog = json.load(open(f"{d}/{s}_progress.json"))["completed"]
        it = list(prog.values())
        det = f"{d}/{s}_detail.json"
        final = os.path.exists(det)
        leg = {
            "final": final, "total": tot, "done": len(it),
            "correct": sum(1 for i in it if i.get("correct")),
            "capped": sum(1 for i in it if i.get("capped")),
            "capped_correct": sum(1 for i in it if i.get("capped") and i.get("correct")),
            "finished": sum(1 for i in it if not i.get("capped")),
            "finished_correct": sum(1 for i in it if not i.get("capped") and i.get("correct")),
            "vetoed": sum(1 for i in it if i.get("exact_vetoed")),
        }
        if final:
            j = json.load(open(det))
            leg.update(score=j["score"], det_correct=j["correct"], det_total=j["total"],
                       parse_failures=j["parse_failures"], capped_count=j.get("capped_count"),
                       median_tokens=j.get("completion_tokens_median"), tpc=j.get("tokens_per_correct"))
            assert j["correct"] == leg["correct"] and j["total"] == tot == leg["done"], (slug, s)
        row["legs"][s] = leg
    m = json.load(open(f"{R}/{slug}{msuf}/math500_detail.json"))
    row["math500"] = {"score": m["score"], "capped_rate": m.get("capped_rate"),
                      "budget": "32k" if msuf.endswith("32k") else "16k", "dir": f"{slug}{msuf}"}
    out.append(row)
print(json.dumps(out, indent=1))
