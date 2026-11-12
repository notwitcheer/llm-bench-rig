#!/usr/bin/env python3
"""wf_summary.py <rundir> <slug> <regime>  > summary.json        |        wf_summary.py --selftest

Item 99z (kv-fp8): Qwen3.8-27B UD-Q4_K_M v3.1 file, f16 KV cache vs FP8 f8_e4m3 KV cache, ONE llama.cpp build (09338b85, FP8 PRs 28392 + 28413),
same task (fix-shopcart, 4 planted bugs, 5 pristine tests), same sampling, same ctx, f16 KV.
Take rule = mc_summary.py's (CLEAN / PASS), copied unchanged. Arm rule and verdict below are this item's (pre-registered, PREREG.md, sha256 in FROZEN.sha256).

VERDICT (applied as written):
  BROKEN         an arm did not load, its tool-call smoke failed, or none of its takes was CLEAN
  INCOMPLETE     an arm has fewer than MIN_CLEAN (2) CLEAN takes: provisional, exit 3, resumes
  TIE            both arms have the same number of PASS takes (of their CLEAN takes)  -> "fp8 kv did not change it on our task"
  CHANGED        the pass counts differ. direction is stated only when the difference is >= 2 passes AND the extension legs
                 reached EXT_TAKES per arm; otherwise the line says "changed, n too small for a direction"
pass counts are the headline; decode tok/s is reported per arm, never gating.
"""
import glob
import json
import os
import re
import sqlite3
import sys

ARMS = ("f16", "fp8")
MIN_CLEAN = 2
EXT_TAKES = 8


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
        row["pred_n_total"] = sum(r["predicted_n"] for r in dq)
        row["pred_s_total"] = round(sum(r["predicted_n"] / r["pred_ps"] for r in dq), 3)
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
            row["peeked_outside"] = bool(re.search(r"moecache99a|voxel-stage|/ref/|rp_gate|PREREG", blob))
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



def arm_counts(takes, tag):
    ts = [t for t in takes.values() if t["tag"] == tag]
    clean = [t for t in ts if t["clean"]]
    return {"takes_run": len(ts), "clean": len(clean), "pass": sum(1 for t in clean if t["pass"]), "decode_tok_s": arm_decode(takes, tag)}


def arm_decode(takes, tag):
    n = t = 0.0
    for r in takes.values():
        if r["tag"] == tag and r["clean"] and r.get("pred_n_total") and r.get("pred_s_total"):
            n += r["pred_n_total"]
            t += r["pred_s_total"]
    return round(n / t, 2) if t > 0 else None


def verdict(takes, files):
    fv = {}
    for tag in ARMS:
        c = arm_counts(takes, tag)
        c["broken"] = (not files.get(tag, {}).get("loaded")) or c["clean"] == 0
        fv[tag] = c
    o, n = fv["f16"], fv["fp8"]
    if o["broken"] or n["broken"]:
        return "BROKEN", fv
    if o["clean"] < MIN_CLEAN or n["clean"] < MIN_CLEAN:
        return "INCOMPLETE", fv
    if o["pass"] == n["pass"]:
        return "TIE", fv
    if abs(o["pass"] - n["pass"]) >= 2 and min(o["takes_run"], n["takes_run"]) >= EXT_TAKES:
        return ("NEW MORE PASSES" if n["pass"] > o["pass"] else "NEW FEWER PASSES"), fv
    return "CHANGED", fv


def gbs(b):
    return "%.2f GB" % (b / 1e9) if b else "size n/a"


def selftest():
    def T(tag, clean, ps):
        return {"tag": tag, "clean": clean, "pass": ps, "pred_n_total": 1000.0, "pred_s_total": 25.0}
    ok = {"f16": {"loaded": True}, "fp8": {"loaded": True}}
    cases = [
        ({"o1": T("f16", True, True), "o2": T("f16", True, True), "o3": T("f16", True, False), "n1": T("fp8", True, True), "n2": T("fp8", True, True), "n3": T("fp8", True, False)}, ok, "TIE"),
        ({"o1": T("f16", True, True), "o2": T("f16", True, True), "o3": T("f16", True, True), "n1": T("fp8", True, False), "n2": T("fp8", True, False), "n3": T("fp8", True, False)}, ok, "CHANGED"),
        ({"o1": T("f16", True, True), "o2": T("f16", True, True), "o3": T("f16", True, False), "n1": T("fp8", True, True), "n2": T("fp8", True, True), "n3": T("fp8", True, True)}, ok, "CHANGED"),
        ({"o1": T("f16", True, True), "o2": T("f16", True, True), "n1": T("fp8", True, True)}, ok, "INCOMPLETE"),
        ({"o1": T("f16", True, True), "o2": T("f16", True, True), "n1": T("fp8", True, True), "n2": T("fp8", False, False)}, ok, "INCOMPLETE"),
        ({"o1": T("f16", True, True), "o2": T("f16", True, True)}, {"f16": {"loaded": True}, "fp8": {"loaded": False}}, "BROKEN"),
        ({"o1": T("f16", False, False), "n1": T("fp8", True, True), "n2": T("fp8", True, True)}, ok, "BROKEN"),
    ]
    bad = 0
    for takes, files, want in cases:
        got = verdict(takes, files)[0]
        print("selftest", "OK " if got == want else "BAD", want, "got", got)
        bad += got != want
    # the extension branch: 8 takes per arm, 6 vs 2 passes
    big = {}
    for k in range(8):
        big["o%d" % k] = T("f16", True, k < 6)
        big["n%d" % k] = T("fp8", True, k < 2)
    g = verdict(big, ok)[0]
    print("selftest", "OK " if g == "NEW FEWER PASSES" else "BAD", "NEW FEWER PASSES", "got", g)
    bad += g != "NEW FEWER PASSES"
    sys.exit(1 if bad else 0)


if "--selftest" in sys.argv:
    selftest()

run, slug = sys.argv[1], sys.argv[2]
regime = sys.argv[3] if len(sys.argv) > 3 else ""
takes = {}
for d in sorted(glob.glob(run + "/takes/*")):
    name = os.path.basename(d)
    if ".partial-" in name:
        continue
    takes[name] = take_row(d, name.rsplit("-", 1)[0])
files = {}
for tag in ARMS:
    fp = run + "/fit-%s.json" % tag
    files[tag] = json.load(open(fp)) if os.path.exists(fp) else {"loaded": False}
ov, fv = verdict(takes, files)
s_bytes = files.get("f16", {}).get("bytes") or files.get("fp8", {}).get("bytes")
pair = json.load(open(run + "/pair.json")) if os.path.exists(run + "/pair.json") else {}
sizes = {tag: files[tag].get("vram_mib") for tag in ARMS}
p, n_ = fv["f16"], fv["fp8"]
line1 = None
if ov not in ("BROKEN", "INCOMPLETE"):
    line1 = ("on one RTX 5090 (32 GB), the same Hermes task on Qwen3.8-27B Q6_K (%s): f16 KV passed %d of %d takes, "
             "fp8 (f8_e4m3) KV passed %d of %d (%s)") % (gbs(s_bytes), p["pass"], p["clean"], n_["pass"], n_["clean"],
             "fp8 KV did not change it on our task" if ov == "TIE" else ov.lower())
out = {"slug": slug, "regime": regime, "verdict": ov, "provisional": ov == "INCOMPLETE", "arms": fv, "files_loaded": files, "vram_mib": sizes,
       "takes": takes, "clip_pair": pair, "final_clip": os.path.exists(run + "/out/whichfile-final.mp4"), "line1": line1}
json.dump(out, sys.stdout, indent=1)
