#!/usr/bin/env python3
"""mc_summary9a1.py <rundir> <slug> <regime>  > summary.json        |        mc_summary9a1.py --selftest

Item 9a1 (ttf TQ1_P ternary Qwen3.8-27B, no adapters, 6.74 GB) vs stock Qwen3.8-27B Q6_K (22.88 GB): one llama-server per arm,
the same Hermes Agent task (fix-shopcart, 4 planted bugs, 5 pristine tests). Take parsing is item 99y's wf_summary.py take_row (unchanged
except the arm tags and the peek regex). The arm rule and verdict are PREREG.md's (sha256 in FROZEN.sha256).

take CLEAN = ended done or wall, >= 3 tool calls in state.db, request errors <= 20% of requests.
take PASS  = CLEAN and the pristine gate passes and the agent's tool calls never named the harness dir.
arm BROKEN = the server did not load, or its tool-call smoke failed, or none of its takes was CLEAN.
verdict, in order:
  BROKEN (ttf)        the ternary arm is BROKEN
  BROKEN (baseline)   the stock arm is BROKEN
  INCOMPLETE          either arm has fewer than 2 CLEAN takes (provisional: exit 3, resumes)
  HOLDS               ttf PASS count >= stock PASS count - 1
  FALLS BEHIND        ttf PASS count <= stock PASS count - 2
GSM8K (first 20, thinking off) and decode tok/s are reported, never gating.
"""
import glob
import json
import os
import re
import sqlite3
import sys

ARMS = ("stock", "ttf")
MIN_CLEAN = 2
FILE_BYTES_DEFAULT = {"stock": 22884408288, "ttf": 6741621728}
PEEK = r"voxel-stage|9a1-ttf-ternary|9a1-runs|/ref/|rp_gate|PREREG|moecache99a|whichfile99y|ripple89"


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
        row["finish_s"] = round((mk["last_change"] if mk.get("end_reason") == "done" and "last_change" in mk
                                 else mk.get("end", mk["enter"])) - mk["enter"], 1)
    hp = d + "/hud/hud.jsonl"
    row["requests"], row["request_errors"] = 0, 0
    if os.path.exists(hp) and "enter" in mk:
        hud = [json.loads(l) for l in open(hp) if l.strip()]
        if hud:
            a = hud[-1]["agents"]["a"]
            row["tokens_server"] = hud[-1].get("srv_tokens")
            row["requests"], row["request_errors"] = a["reqs"], a["err"]
            row["vram_mib_max"] = max([r.get("vram_mib") or 0 for r in hud] or [0])
    rp = d + "/hud/requests.jsonl"
    if os.path.exists(rp):
        rq = [json.loads(l) for l in open(rp) if l.strip()]
        dq = [r for r in rq if r.get("pred_ps") and r.get("predicted_n")]
        if dq:
            row["decode_tok_s_engine"] = round(sum(r["predicted_n"] for r in dq) / sum(r["predicted_n"] / r["pred_ps"] for r in dq), 1)
        row["pred_n_total"] = sum(r["predicted_n"] for r in dq)
        row["pred_s_total"] = round(sum(r["predicted_n"] / r["pred_ps"] for r in dq), 3)
    row["tool_calls"] = None
    sp = d + "/state/state.db"
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
            row["peeked_outside"] = bool(re.search(PEEK, blob))
            c.close()
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
    return {"takes_run": len(ts), "clean": len(clean), "pass": sum(1 for t in clean if t["pass"]),
            "decode_tok_s": arm_decode(takes, tag)}


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
    s, t = fv["stock"], fv["ttf"]
    if t["broken"]:
        return "BROKEN (ttf)", fv
    if s["broken"]:
        return "BROKEN (baseline)", fv
    if s["clean"] < MIN_CLEAN or t["clean"] < MIN_CLEAN:
        return "INCOMPLETE", fv
    if t["pass"] >= s["pass"] - 1:
        return "HOLDS", fv
    if t["pass"] <= s["pass"] - 2:
        return "FALLS BEHIND", fv
    return "HOLDS", fv


def line1(v, s, t, bytes_stock, bytes_ttf):
    x = "%.1f" % ((bytes_stock - bytes_ttf) / 1e9)
    if v == "HOLDS":
        return ("a 27b ternary gguf at 6.7 GB held the same hermes task: %d of %d passes against %d of %d for the 23 GB file, %s GB less on disk"
                % (t["pass"], t["clean"], s["pass"], s["clean"], x))
    if v == "FALLS BEHIND":
        return ("a 27b ternary gguf at 6.7 GB fell behind the 23 GB file on the same hermes task: %d of %d passes against %d of %d, %s GB less on disk"
                % (t["pass"], t["clean"], s["pass"], s["clean"], x))
    return "verdict %s: no line 1 (provisional or broken; see the status note)" % v


def gsm8k_line(g):
    if not g:
        return "gsm8k: not run"
    holds = g["ttf_correct"] >= g["stock_correct"] - 2
    return ("gsm8k first %d (thinking off): ttf %d/%d vs stock %d/%d (%s)"
            % (g["n"], g["ttf_correct"], g["n"], g["stock_correct"], g["n"],
               ("holds on these %d" % g["n"]) if holds else ("behind on these %d" % g["n"])))


def selftest():
    def T(tag, ended=True, gate=True, peek=False, tc=5, re_=0, rt=10):
        # clean is derived from the take rule (as take_row does), never set by hand
        end = "done" if ended else "error"
        clean = bool(end in ("done", "wall") and tc >= 3 and re_ <= 0.2 * max(rt, 1) and rt > 0)
        r = {"tag": tag, "clean": clean, "gate_pass": gate, "peeked_outside": peek, "tool_calls": tc,
             "request_errors": re_, "requests": rt, "end": end}
        r["pass"] = bool(clean and gate and not peek)
        return r

    def mk(stock, ttf):
        d = {}
        for i, t in enumerate(stock):
            d["stock-%d" % (i + 1)] = t
        for i, t in enumerate(ttf):
            d["ttf-%d" % (i + 1)] = t
        return d

    ok = {"stock": {"loaded": True}, "ttf": {"loaded": True}}
    cases = [
        (mk([T("stock")] * 3, [T("ttf")] * 3), ok, "HOLDS"),
        (mk([T("stock")] * 3, [T("ttf", gate=False)] * 3), ok, "FALLS BEHIND"),
        (mk([T("stock")] * 3, [T("ttf", gate=False), T("ttf"), T("ttf")]), ok, "HOLDS"),
        (mk([T("stock")] * 3, [T("ttf", gate=False)] * 2 + [T("ttf")]), ok, "FALLS BEHIND"),
        (mk([T("stock")] * 3, [T("ttf", peek=True)] * 3), ok, "FALLS BEHIND"),
        (mk([T("stock")] * 3, [T("ttf")] * 3), {"stock": {"loaded": True}, "ttf": {"loaded": False}}, "BROKEN (ttf)"),
        (mk([T("stock", ended=False)] * 3, [T("ttf")] * 3), ok, "BROKEN (baseline)"),
        (mk([T("stock")] * 3, [T("ttf")] * 1), ok, "INCOMPLETE"),
        (mk([T("stock")] * 3, [T("ttf", tc=2)] * 3), ok, "BROKEN (ttf)"),
        (mk([T("stock")] * 3, [T("ttf", tc=2), T("ttf")]), ok, "INCOMPLETE"),
        (mk([T("stock", re_=3, rt=10)] * 3, [T("ttf")] * 3), ok, "BROKEN (baseline)"),
    ]
    bad = 0
    for i, (takes, files, want) in enumerate(cases):
        got, _ = verdict(takes, files)
        good = got == want
        bad += not good
        print("case %d: got %r want %r %s" % (i, got, want, "ok" if good else "FAIL"))
    _, fv = verdict(mk([T("stock")] * 3, [T("ttf")] * 3), ok)
    l1 = line1("HOLDS", fv["stock"], fv["ttf"], 22884408288, 6741621728)
    print("line1:", l1)
    want_l1 = ("a 27b ternary gguf at 6.7 GB held the same hermes task: 3 of 3 passes against 3 of 3 for the 23 GB file, 16.1 GB less on disk")
    if l1 != want_l1:
        bad += 1
        print("line1 FAIL")
    print("SELFTEST", "PASS" if bad == 0 else "FAIL (%d)" % bad)
    return bad


def main():
    run = sys.argv[1]
    slug = sys.argv[2] if len(sys.argv) > 2 else ""
    takes = {}
    for d in sorted(glob.glob(os.path.join(run, "takes", "*"))):
        name = os.path.basename(d)
        if ".partial-" in name or not os.path.isdir(d):
            continue
        takes[name] = take_row(d, name.rsplit("-", 1)[0])
    files = {}
    for tag in ARMS:
        fp = os.path.join(run, "fit-%s.json" % tag)
        files[tag] = json.load(open(fp)) if os.path.exists(fp) else {"loaded": False}
    v, fv = verdict(takes, files)
    sizes = {tag: files[tag].get("bytes") or FILE_BYTES_DEFAULT[tag] for tag in ARMS}
    l1 = None if (v == "INCOMPLETE" or v.startswith("BROKEN")) else line1(v, fv["stock"], fv["ttf"], sizes["stock"], sizes["ttf"])
    g = json.load(open(os.path.join(run, "gsm8k.json"))) if os.path.exists(os.path.join(run, "gsm8k.json")) else None
    pp = os.path.join(run, "pair.json")
    pair = json.load(open(pp)) if os.path.exists(pp) else {}
    out = {"slug": slug, "regime": sys.argv[3] if len(sys.argv) > 3 else "", "verdict": v,
           "provisional": v == "INCOMPLETE", "arms": fv, "files_loaded": files, "file_bytes": sizes,
           "takes": takes, "gsm8k": g, "gsm8k_line": gsm8k_line(g), "clip_pair": pair,
           "final_clip": os.path.exists(os.path.join(run, "out", "whichfile-final.mp4")), "line1": l1}
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        sys.exit(selftest())
    main()
