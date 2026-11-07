#!/usr/bin/env python3
"""applies the PREREG verdict rule. inputs: results dir with <model>.jsonl (generation rows) + checks.jsonl. writes aggregate.json + readout.md."""
import argparse, json, os, sys, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glbcommon as G
ap = argparse.ArgumentParser(); ap.add_argument("--res", required=True); ap.add_argument("--out", required=True); ap.add_argument("--cells", type=int, default=0); ap.add_argument("--broken-note", default="")
a = ap.parse_args()
C = G.cfg(); P = G.prompts(); NC = a.cells or len(P) * len(C["seeds"]); M = C["verdict"]["ahead_margin_cells"]
def rows(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []
chk = {}
for r in rows(a.res + "/checks.jsonl"): chk[(r["model"], r["prompt"], r["seed"])] = r
out = {"models": {}, "n_cells_planned": NC}
for m in C["models"]:
    k = m["key"]; gen = rows(f"{a.res}/{k}.jsonl")
    cs = [chk[(k, g["prompt"], g["seed"])] for g in gen if (k, g["prompt"], g["seed"]) in chk]
    toks = [g["completion_tokens"] for g in gen if g.get("completion_tokens")]
    d = {"label": m["label"], "size": m["size"], "n_generated": len(gen), "n_scored": len(cs),
         "html": sum(c["html"] for c in cs), "load_ok": sum(c["load_ok"] for c in cs), "nonempty": sum(c["nonempty"] for c in cs),
         "page_valid": sum(c["page_valid"] for c in cs), "glb_valid": sum(c["glb_valid"] for c in cs), "full_pass": sum(c["full_pass"] for c in cs),
         "harness_faults": sum(1 for c in cs if c.get("harness_fault")),
         "truncated": sum(1 for g in gen if g.get("finish_reason") == "length"),
         "tokens_median": int(statistics.median(toks)) if toks else None, "tokens_total": sum(toks),
         "per_prompt": {p["id"]: [int(chk[(k, p["id"], s)]["full_pass"]) if (k, p["id"], s) in chk else None for s in C["seeds"]] for p in P},
         "fail_reasons": {}}
    for c in cs:
        if not c["full_pass"]: d["fail_reasons"][c.get("reason") or "?"] = d["fail_reasons"].get(c.get("reason") or "?", 0) + 1
    out["models"][k] = d
sub = out["models"][C["models"][0]["key"]]
comp = [m["key"] for m in C["models"][1:]]
fault_models = [k for k, v in out["models"].items() if v["n_scored"] and v["harness_faults"] / v["n_scored"] > 0.2]
complete = all(v["n_scored"] >= NC for v in out["models"].values())
if a.broken_note and os.path.exists(a.broken_note): overall = "BROKEN"; why = "the 4B did not load: " + open(a.broken_note).read().strip()[:160]
elif fault_models: overall = "INVALID"; why = "checker fault rate > 20% for " + ",".join(fault_models)
elif sub["n_scored"] >= NC and sub["page_valid"] == 0: overall = "BROKEN"; why = "the 4B has 0 valid pages over all %d cells" % NC
elif not complete: overall = "PROVISIONAL"; why = "scored cells: " + ", ".join("%s %d/%d" % (k, v["n_scored"], NC) for k, v in out["models"].items())
else:
    lab = {}
    for k in comp:
        dd = sub["full_pass"] - out["models"][k]["full_pass"]
        lab[k] = "4B AHEAD" if dd >= M else ("4B BEHIND" if dd <= -M else "LEVEL")
    s = set(lab.values())
    overall = "AHEAD OF BOTH" if s == {"4B AHEAD"} else "BEHIND BOTH" if s == {"4B BEHIND"} else "LEVEL WITH BOTH" if s == {"LEVEL"} else "MIXED"
    out["per_comparator"] = lab; why = "; ".join("%s: %s (d=%+d)" % (k, lab[k], sub["full_pass"] - out["models"][k]["full_pass"]) for k in comp)
# a BROKEN 4B before all cells are scored stays PROVISIONAL by design (needs all its cells)
out["overall"] = overall; out["why"] = why; out["provisional"] = overall == "PROVISIONAL"
ks = [m["key"] for m in C["models"]]
b, q, o = [out["models"][k] for k in ks]
out["line1"] = ("on 12 absurd prompts a 4B model's Three.js page loaded, drew a scene and exported a valid GLB in %d/%d tries; Qwen3.8-27B %d/%d, Ornith-1.5-35B %d/%d "
                "(same system prompt, script checks only)." % (b["full_pass"], NC, q["full_pass"], NC, o["full_pass"], NC)) if overall not in ("BROKEN", "INVALID") else "no post: " + overall
json.dump(out, open(a.out + "/aggregate.json", "w"), indent=1)
L = ["# GLB-3D-Coder readout (script output, not edited)", "", "overall: **%s** (%s)" % (overall, why), "", "line 1: " + out["line1"], "",
     "| model | scored | html | load_ok | nonempty | page_valid | glb_valid | full_pass | trunc | tok median | tok total |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for k, v in out["models"].items():
    L.append("| %s | %d/%d | %d | %d | %d | %d | %d | **%d** | %d | %s | %d |" % (v["label"], v["n_scored"], NC, v["html"], v["load_ok"], v["nonempty"], v["page_valid"], v["glb_valid"], v["full_pass"], v["truncated"], v["tokens_median"], v["tokens_total"]))
L += ["", "## per prompt (full_pass per seed; 1 pass, 0 fail, . not run)", "", "| prompt | " + " | ".join(out["models"][k]["size"] for k in ks) + " |", "|---|" + "---|" * len(ks)]
for p in P:
    L.append("| %s | " % p["id"] + " | ".join("".join("." if x is None else str(x) for x in out["models"][k]["per_prompt"][p["id"]]) for k in ks) + " |")
L += ["", "## top failure reasons"]
for k, v in out["models"].items(): L.append("- %s: %s" % (v["label"], "; ".join("%s x%d" % kv for kv in sorted(v["fail_reasons"].items(), key=lambda x: -x[1])[:5]) or "none"))
open(a.out + "/readout.md", "w").write("\n".join(L) + "\n")
print(overall, "|", why)
