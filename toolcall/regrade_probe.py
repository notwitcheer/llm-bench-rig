"""Re-grade stored single-turn probe generations with the current parser (no GPU). Multi-turn entries
cannot be re-graded: a parser change alters the trajectory (a malformed step ends the turn), so S5 is
only re-run, never re-scored. usage: python regrade_probe.py <leg dir> [<leg dir> ...]"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import bfcl_items as B  # noqa: E402
import qwen38_xml as X  # noqa: E402
from driver import grade_single  # noqa: E402

for d in map(Path, sys.argv[1:]):
    rows, agg, flips = [], {}, []
    for line in open(d / "items.jsonl"):
        r = json.loads(line)
        e = B.entry(r["id"])
        tools = B.tools_for(e)
        p = X.parse_reply(r["generated"], tools, True, r["stop_type"])
        oc, chk, lo = grade_single(r["id"], e, tools, p, r["set"] == "S4")
        n = {"id": r["id"], "set": r["set"], "parser": X.PARSER_VERSION, "bucket": p.bucket, "flags": p.flags,
             "outcome": oc, "detail": p.detail, "bfcl": chk, "outcome_parser1": r["outcome"]}
        rows.append(n)
        agg.setdefault(r["set"], {}).setdefault(oc, 0)
        agg[r["set"]][oc] += 1
        if oc != r["outcome"]:
            flips.append(f"{r['id']}: {r['outcome']} -> {oc}")
    out = {"leg": d.name, "parser": X.PARSER_VERSION, "single_turn": agg, "flips_vs_parser1": flips,
           "python_bool_replies": sum(1 for x in rows if "python_bool" in x["flags"])}
    (d / "items_parser2.jsonl").write_text("".join(json.dumps(x, default=str) + "\n" for x in rows))
    (d / "regrade_parser2.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))
