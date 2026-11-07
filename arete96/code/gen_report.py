#!/usr/bin/env python3
"""gen_report.py RUNDIR : CPU digest of the three GPU legs of item 96-arete-eval -> RUNDIR/gen-report.json (+ one printed line per file). Frozen.
Per file: rows, finish reasons, length-capped, empty answers, completion tokens, per-request decode speed (server timings: mean/median/p10 of predicted_per_second), aggregate
tok/s of the run (status file), wall; and a PROTOCOL check: any row whose server timings report speculative drafting (draft_n > 0) is counted (expected 0: no --spec flag was passed).
Board: rows per task and the leg log's wall seconds. Reads only files the item wrote."""
import json, os, statistics, sys

run = sys.argv[1]


def rl(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


rep = {}
for tag in ("base", "arete"):
    rows = rl(os.path.join(run, "completions-%s.jsonl" % tag))
    sp = [r["timings"]["predicted_per_second"] for r in rows if r.get("timings", {}).get("predicted_per_second")]
    st = json.load(open(os.path.join(run, "gen-status-%s.json" % tag))) if os.path.exists(os.path.join(run, "gen-status-%s.json" % tag)) else {}
    toks = [int(r.get("completion_tokens") or 0) for r in rows]
    rep[tag] = dict(rows=len(rows), finish=dict((k, sum(1 for r in rows if r.get("finish_reason") == k)) for k in sorted({r.get("finish_reason") for r in rows})),
                    length_capped=sum(1 for r in rows if r.get("finish_reason") == "length"), empty=sum(1 for r in rows if not (r.get("response") or "").strip()),
                    completion_tokens=sum(toks), mean_tokens=round(sum(toks) / max(len(toks), 1), 1), max_tokens_seen=max(toks or [0]),
                    decode_tok_s_per_request=dict(mean=round(statistics.mean(sp), 1) if sp else None, median=round(statistics.median(sp), 1) if sp else None,
                                                  p10=round(sorted(sp)[len(sp) // 10], 1) if sp else None),
                    draft_rows=sum(1 for r in rows if (r.get("timings", {}).get("draft_n") or 0) > 0), status=st)
bd = {}
for t in ("gpqa", "he", "arc", "mmlu"):
    rs = rl(os.path.join(run, "board", "new-q6k", t + ".jsonl"))
    bd[t] = dict(rows=len(rs), correct=sum(1 for r in rs if r.get("ok")), mean_s=round(sum(r.get("s", 0) for r in rs) / max(len(rs), 1), 3))
rep["board_arete"] = bd
rep["protocol_warnings"] = [("%s: %d rows report speculative drafting" % (t, rep[t]["draft_rows"])) for t in ("base", "arete") if rep[t]["draft_rows"]]
json.dump(rep, open(os.path.join(run, "gen-report.json"), "w"), indent=1)
for t in ("base", "arete"):
    r = rep[t]
    print("GEN %s: %d rows, %d length-capped, %d empty, %d tokens (mean %.0f, max %d), decode per request median %s tok/s, run aggregate %s tok/s" % (
        t, r["rows"], r["length_capped"], r["empty"], r["completion_tokens"], r["mean_tokens"], r["max_tokens_seen"], r["decode_tok_s_per_request"]["median"], r["status"].get("this_run_agg_tok_s")))
print("GEN board(Arete): " + ", ".join("%s %d/%d" % (t, bd[t]["correct"], bd[t]["rows"]) for t in bd) + ("  WARNINGS: " + "; ".join(rep["protocol_warnings"]) if rep["protocol_warnings"] else ""))
