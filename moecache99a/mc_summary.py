#!/usr/bin/env python3
"""mc_summary.py <rundir> <slug> <regime>  > summary.json        |        mc_summary.py --selftest

Item 99a artefact writer AND the pre-registered verdict rule (sha256-frozen in FROZEN.sha256 before any counted output; the item refuses to run when the hash differs).
Derived from item 89's rp_summary.py. Two arms on ONE file (Ornith-1.5-122B UD-IQ3_XXS), one llama.cpp build:
  plain = `--n-cpu-moe 24` (the best offload setting of our 25 Oct sweep), cache = `-cmoe --moe-cache-mib N --moe-cache-cpu-share F` (N, F = the tuner's recommendation).
Per take: engine end reason, finish seconds from the take's own Enter, tool calls (state.db), gate verdict (pre-registered pass test), engine timings, GPU power and VRAM.

RULE (one place, applied as written):
  take CLEAN = ended done or wall AND >= 3 tool calls in state.db AND request errors <= 20% of requests.
  take PASS  = CLEAN AND gate.pass (pristine tests 5 passed, 0 failed, 0 errors, nothing under tests/ touched) AND the agent's tool calls never named the harness dir.
  arm BROKEN = the server did not load, or its tool-call smoke failed, or none of its takes was CLEAN.
  arm decode = sum(predicted_n) / sum(predicted_n / predicted_per_second) over every request of the arm's CLEAN takes (llama-server's own timings, token-weighted).
  ratio r    = cache decode / plain decode.
  verdict:
    BROKEN        plain arm BROKEN (nothing to compare against)
    CACHE FAILED  plain fine, cache arm BROKEN (suspect our flags/harness first: CPU diagnosis before any rerun)
    INCOMPLETE    either arm has fewer than 2 CLEAN takes (window cut): provisional, the item exits 3 and resumes
    FASTER        r >= 1.10
    SLOWER        r <= 0.90
    TIE           0.90 < r < 1.10   (the band is wide on purpose: the two arms run different agent trajectories; our fixed-prompt depth rows spread <= 0.6%)
  pass counts per arm are REPORTED, never gating.
"""
import glob
import json
import os
import re
import sqlite3
import sys

ARMS = ("plain", "cache")
FASTER_AT, SLOWER_AT, MIN_CLEAN = 1.10, 0.90, 2


def arm_decode(takes, tag):
    n = t = 0.0
    for r in takes.values():
        if r["tag"] == tag and r["clean"] and r.get("pred_n_total") and r.get("pred_s_total"):
            n += r["pred_n_total"]
            t += r["pred_s_total"]
    return round(n / t, 2) if t > 0 else None


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
        fv[tag] = {"takes_run": len(ts), "clean": len(clean), "pass": sum(1 for t in ts if t["pass"]), "broken": broken, "decode_tok_s": arm_decode(takes, tag)}
    p, c = fv.get("plain", {"broken": True}), fv.get("cache", {"broken": True})
    if p["broken"]:
        return "BROKEN", fv, None
    if c["broken"]:
        return "CACHE FAILED", fv, None
    if p["clean"] < MIN_CLEAN or c["clean"] < MIN_CLEAN:
        return "INCOMPLETE", fv, None
    if not p["decode_tok_s"] or not c["decode_tok_s"]:
        return "BROKEN", fv, None
    r = round(c["decode_tok_s"] / p["decode_tok_s"], 3)
    if r >= FASTER_AT:
        return "FASTER", fv, r
    if r <= SLOWER_AT:
        return "SLOWER", fv, r
    return "TIE", fv, r


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


def selftest():
    def T(tag, clean, ps, n=1000.0, tps=40.0):
        return {"tag": tag, "clean": clean, "pass": ps, "pred_n_total": n, "pred_s_total": n / tps}
    ok = {"plain": {"loaded": True}, "cache": {"loaded": True}}
    cases = [
        ({"p1": T("plain", True, True, tps=40), "p2": T("plain", True, True, tps=40), "c1": T("cache", True, True, tps=60), "c2": T("cache", True, True, tps=60)}, ok, "FASTER"),
        ({"p1": T("plain", True, True, tps=40), "p2": T("plain", True, True, tps=40), "c1": T("cache", True, True, tps=44), "c2": T("cache", True, True, tps=44)}, ok, "FASTER"),
        ({"p1": T("plain", True, True, tps=40), "p2": T("plain", True, True, tps=40), "c1": T("cache", True, True, tps=43.9), "c2": T("cache", True, True, tps=43.9)}, ok, "TIE"),
        ({"p1": T("plain", True, True, tps=40), "p2": T("plain", True, True, tps=40), "c1": T("cache", True, False, tps=38), "c2": T("cache", True, True, tps=38)}, ok, "TIE"),
        ({"p1": T("plain", True, True, tps=40), "p2": T("plain", True, True, tps=40), "c1": T("cache", True, True, tps=36), "c2": T("cache", True, True, tps=36)}, ok, "SLOWER"),
        ({"p1": T("plain", True, True, tps=40), "p2": T("plain", True, True, tps=40), "c1": T("cache", True, True, tps=36.4), "c2": T("cache", True, True, tps=36.4)}, ok, "TIE"),
        ({"p1": T("plain", True, True), "p2": T("plain", True, True), "c1": T("cache", False, False), "c2": T("cache", False, False)}, ok, "CACHE FAILED"),
        ({"p1": T("plain", True, True), "p2": T("plain", True, True)}, {"plain": {"loaded": True}, "cache": {"loaded": False}}, "CACHE FAILED"),
        ({"c1": T("cache", True, True), "c2": T("cache", True, True)}, {"plain": {"loaded": False}, "cache": {"loaded": True}}, "BROKEN"),
        ({"p1": T("plain", False, False), "p2": T("plain", False, False), "c1": T("cache", True, True), "c2": T("cache", True, True)}, ok, "BROKEN"),
        ({"p1": T("plain", True, True), "c1": T("cache", True, True), "c2": T("cache", True, True)}, ok, "INCOMPLETE"),
        ({"p1": T("plain", True, True), "p2": T("plain", True, True), "c1": T("cache", True, True)}, ok, "INCOMPLETE"),
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
    if ".partial-" in name:
        continue
    takes[name] = take_row(d, name.rsplit("-", 1)[0])
files = {}
for tag in ARMS:
    fp = run + "/fit-%s.json" % tag
    files[tag] = json.load(open(fp)) if os.path.exists(fp) else {"loaded": False}
ov, fv, ratio = verdict(takes, files)
race = json.load(open(run + "/out/race.mp4.summary.json")) if os.path.exists(run + "/out/race.mp4.summary.json") else {}
pair = json.load(open(run + "/pair.json")) if os.path.exists(run + "/pair.json") else {}
tune = json.load(open(run + "/tune.json")) if os.path.exists(run + "/tune.json") else {}
probe = {}
for tag in ARMS:
    pp = run + "/probe-%s.json" % tag
    if os.path.exists(pp):
        probe[tag] = json.load(open(pp))


def line1():
    if ov in ("BROKEN", "INCOMPLETE", "CACHE FAILED") or ratio is None:
        return None
    dp, dc = fv["plain"]["decode_tok_s"], fv["cache"]["decode_tok_s"]
    base = "on one RTX 5090 (32 GB) a 122B model, run by Hermes Agent: "
    if ov == "FASTER":
        return base + "llama.cpp's new MoE cache took decode from %.1f to %.1f tok/s (%.2fx) with the same model file" % (dp, dc, ratio)
    if ov == "SLOWER":
        return base + "llama.cpp's new MoE cache decoded at %.1f tok/s against %.1f without it (%.2fx), same model file" % (dc, dp, ratio)
    return base + "llama.cpp's new MoE cache changed decode from %.1f to %.1f tok/s (%.2fx), inside the +-10%% tie band, same model file" % (dp, dc, ratio)


out = {"slug": slug, "regime": regime, "verdict": ov, "ratio_cache_over_plain": ratio, "provisional": ov == "INCOMPLETE", "arms": fv, "files_loaded": files, "takes": takes, "clip_pair": pair,
       "tune": tune, "probe": probe, "race": race, "final_clip": os.path.exists(run + "/out/moecache-final.mp4"), "line1": line1()}
json.dump(out, sys.stdout, indent=1)
