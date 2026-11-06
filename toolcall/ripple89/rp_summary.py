#!/usr/bin/env python3
"""rp_summary.py <rundir> <slug> <regime>  > summary.json        |        rp_summary.py --selftest

Item 89 artefact writer AND the pre-registered verdict rule (this file is sha256-frozen in PREREG.md before any counted output; the item refuses to run when the hash differs).
Per take: engine end reason, finish seconds from the take's own Enter, tool calls (state.db), gate verdict (gate.json written by the pre-registered pass test), server-counted tokens,
engine timings (decode / prompt tok/s from llama-server's own `timings`), GPU power and VRAM during the take (nvidia-smi, 1 Hz). Then the per-file and overall verdict.

RULE (one place, applied as written):
  take CLEAN   = ended done or wall AND >= 3 tool calls in state.db AND request errors <= 20% of requests.
  take PASS    = CLEAN AND gate.pass (pristine tests 5 passed, 0 failed, 0 errors, nothing under tests/ touched) AND the agent's tool calls never named the harness dir (peeked_outside false).
  file BROKEN  = the server did not load, or its tool-call smoke failed, or none of its takes was CLEAN.
  overall:
    BROKEN         every file BROKEN
    REPRODUCED     the primary file (IQ3_XXS) ran >= 2 takes and every one of them is a PASS
    PARTIAL        not REPRODUCED, and at least one take on any file is a PASS
    NOT REPRODUCED at least one CLEAN take, none a PASS
    INCOMPLETE     fewer than 2 primary takes ran (window cut), shown as PROVISIONAL, the item exits 3 and resumes
"""
import glob
import json
import os
import re
import sqlite3
import sys

PRIMARY = "iq3xxs"


def verdict(takes, files):
    """takes: {name: row}; files: {tag: {"loaded": bool}}"""
    by = {}
    for n, t in takes.items():
        by.setdefault(t["tag"], []).append(t)
    fv = {}
    for tag, f in files.items():
        ts = by.get(tag, [])
        clean = [t for t in ts if t["clean"]]
        broken = (not f.get("loaded")) or (not clean)
        fv[tag] = {"takes_run": len(ts), "clean": len(clean), "pass": sum(1 for t in ts if t["pass"]), "broken": broken}
    prim = fv.get(PRIMARY, {"takes_run": 0, "pass": 0, "clean": 0, "broken": True})
    if all(v["broken"] for v in fv.values()):
        return "BROKEN", fv
    if prim["takes_run"] < 2:
        return "INCOMPLETE", fv
    if prim["pass"] == prim["takes_run"]:
        return "REPRODUCED", fv
    if any(v["pass"] for v in fv.values()):
        return "PARTIAL", fv
    return "NOT REPRODUCED", fv


def mmss(s):
    return "%d:%02d" % (int(s // 60), int(s % 60))


def marks_of(d):
    mk = {}
    p = d + "/rec/events-a.jsonl"
    if os.path.exists(p):
        for l in open(p):
            j = json.loads(l)
            if "mark" in j:
                mk.setdefault(j["mark"], j["t"])
                if j["mark"] == "end":
                    mk["end_reason"] = j.get("reason")
    return mk


def take_row(d, tag):
    mk = marks_of(d)
    row = {"dir": os.path.basename(d), "tag": tag, "end": mk.get("end_reason")}
    if "enter" in mk:
        row["finish_s"] = round((mk["last_change"] if mk.get("end_reason") == "done" and "last_change" in mk else mk.get("end", mk["enter"])) - mk["enter"], 1)
    hp = d + "/hud/hud.jsonl"
    row["requests"], row["request_errors"] = 0, 0
    if os.path.exists(hp) and "enter" in mk:
        hud = [json.loads(l) for l in open(hp) if l.strip()]
        if hud:
            a = hud[-1]["agents"]["a"]
            row["tokens_server"] = hud[-1].get("srv_tokens")
            row["tokens_exact_from_usage"] = a["exact_pred"]
            row["requests"], row["request_errors"] = a["reqs"], a["err"]
            row["vram_mib_max"] = max([r.get("vram_mib") or 0 for r in hud] or [0])
    rp = d + "/hud/requests.jsonl"
    if os.path.exists(rp):
        rq = [json.loads(l) for l in open(rp) if l.strip()]
        dq = [r for r in rq if r.get("pred_ps") and r.get("predicted_n")]
        pq = [r for r in rq if r.get("prompt_ps") and r.get("prompt_n") and r["prompt_n"] > 200]
        if dq:
            row["decode_tok_s_engine"] = round(sum(r["predicted_n"] for r in dq) / sum(r["predicted_n"] / r["pred_ps"] for r in dq), 1)
        if pq:
            row["prompt_tok_s_engine"] = round(sum(r["prompt_n"] for r in pq) / sum(r["prompt_n"] / r["prompt_ps"] for r in pq), 1)
        row["prompt_tokens_total"] = sum(r.get("prompt_n") or 0 for r in rq)
    pw = d + "/hud/power.csv"
    if os.path.exists(pw) and "enter" in mk:
        w = []
        for l in open(pw):
            p = l.strip().split(",")
            try:
                t, x = float(p[0]), float(p[1])
            except Exception:
                continue
            if mk["enter"] <= t <= mk["enter"] + row.get("finish_s", 0):
                w.append(x)
        if w:
            row["gpu_w_mean"], row["gpu_w_max"] = round(sum(w) / len(w), 1), round(max(w), 1)
    sp = d + "/state/state.db"
    row["tool_calls"] = None
    if os.path.exists(sp):
        try:
            c = sqlite3.connect(sp)
            n = 0
            for (tc,) in c.execute("select tool_calls from messages where role='assistant' and tool_calls is not null and tool_calls != ''"):
                try:
                    n += len(json.loads(tc))
                except Exception:
                    n += 1
            row["tool_calls"] = n
            blob = " ".join(x or "" for (x,) in c.execute("select tool_calls from messages where role='assistant'"))
            row["peeked_outside"] = bool(re.search(r"voxelbench|ripple89|/ref/|rp_gate|PREREG", blob))
            tr = [x for (x,) in c.execute("select content from messages where role='tool' order by id") if x and "passed" in x]
            m = re.findall(r"(\d+ failed[^\n\"]*|\d+ passed[^\n\"]*)", tr[-1]) if tr else []
            row["agent_last_pytest"] = (m[-1] if m else None)
        except Exception as e:
            row["tool_calls"] = "unreadable: %s" % str(e)[:60]
    gp = d + "/gate/gate.json"
    if os.path.exists(gp):
        g = json.load(open(gp))
        row["gate_pass"] = bool(g.get("pass"))
        row["gate_passed"], row["gate_failed"] = g.get("passed"), g.get("failed")
        row["tests_touched"] = g.get("tests_touched")
        row["files_changed"] = g.get("changed_files")
    tc = row["tool_calls"] if isinstance(row["tool_calls"], int) else 0
    errs = row["request_errors"] <= 0.2 * max(row["requests"], 1)
    row["clean"] = bool(row["end"] in ("done", "wall") and tc >= 3 and errs and row["requests"] > 0)
    row["pass"] = bool(row["clean"] and row.get("gate_pass") and not row.get("peeked_outside"))
    return row


def selftest():
    def T(tag, clean, ps):
        return {"tag": tag, "clean": clean, "pass": ps}
    ok = {"iq3xxs": {"loaded": True}, "iq2s": {"loaded": True}}
    cases = [
        ({"a1": T("iq3xxs", True, True), "a2": T("iq3xxs", True, True), "b1": T("iq2s", True, False)}, ok, "REPRODUCED"),
        ({"a1": T("iq3xxs", True, True), "a2": T("iq3xxs", True, False)}, ok, "PARTIAL"),
        ({"a1": T("iq3xxs", True, False), "a2": T("iq3xxs", True, False), "b1": T("iq2s", True, True)}, ok, "PARTIAL"),
        ({"a1": T("iq3xxs", True, False), "a2": T("iq3xxs", True, False), "b1": T("iq2s", True, False)}, ok, "NOT REPRODUCED"),
        ({"a1": T("iq3xxs", False, False), "a2": T("iq3xxs", False, False), "b1": T("iq2s", False, False)}, ok, "BROKEN"),
        ({}, {"iq3xxs": {"loaded": False}, "iq2s": {"loaded": False}}, "BROKEN"),
        ({"a1": T("iq3xxs", True, True)}, ok, "INCOMPLETE"),
        ({"a1": T("iq3xxs", True, True), "a2": T("iq3xxs", True, True), "b1": T("iq2s", False, False)}, ok, "REPRODUCED"),
        ({"b1": T("iq2s", True, True), "b2": T("iq2s", True, True)}, {"iq3xxs": {"loaded": False}, "iq2s": {"loaded": True}}, "INCOMPLETE"),
    ]
    bad = 0
    for takes, files, want in cases:
        got = verdict(takes, files)[0]
        print("selftest", "OK " if got == want else "BAD", want, "got", got)
        bad += got != want
    sys.exit(1 if bad else 0)


if "--selftest" in sys.argv:
    selftest()

run, slug = sys.argv[1], sys.argv[2]
regime = sys.argv[3] if len(sys.argv) > 3 else ""
takes = {}
for d in sorted(glob.glob(run + "/takes/*")):
    name = os.path.basename(d)
    takes[name] = take_row(d, name.rsplit("-", 1)[0])
files = {}
for tag in ("iq3xxs", "iq2s"):
    fp = run + "/fit-%s.json" % tag
    files[tag] = json.load(open(fp)) if os.path.exists(fp) else {"loaded": False}
ov, fv = verdict(takes, files)
race = json.load(open(run + "/out/race.mp4.summary.json")) if os.path.exists(run + "/out/race.mp4.summary.json") else {}
pair = json.load(open(run + "/pair.json")) if os.path.exists(run + "/pair.json") else {}


def line1():
    p = [t for t in takes.values() if t["tag"] == PRIMARY and t["clean"]]
    if ov in ("BROKEN", "INCOMPLETE") or not p:
        return None
    t1 = sorted(p, key=lambda t: t["dir"])[0]
    nrun = fv[PRIMARY]["takes_run"]
    base = "Ripple-65B Preview (IQ3_XXS GGUF, llama.cpp PR build) driving Hermes Agent on one RTX 5090, every token local: "
    mid = "%d of %d takes ended with 5/5 tests passing and the tests untouched" % (fv[PRIMARY]["pass"], nrun)
    num = ", first take %s, %s tool calls, %s W mean GPU power" % (mmss(t1["finish_s"]), t1["tool_calls"], t1.get("gpu_w_mean", "?"))
    if ov == "NOT REPRODUCED":
        mid = "none of %d takes got the 4-bug repo to 5/5" % nrun
    return base + mid + num


out = {"slug": slug, "regime": regime, "verdict": ov, "provisional": ov == "INCOMPLETE", "files": fv, "files_loaded": files, "takes": takes, "clip_pair": pair, "race": race,
       "final_clip": os.path.exists(run + "/out/ripple-final.mp4"), "line1": line1()}
json.dump(out, sys.stdout, indent=1)
