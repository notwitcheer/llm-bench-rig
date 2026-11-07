#!/usr/bin/env python3
"""b93_summary.py <rundir> <slug> <regime>  > summary.json      |      b93_summary.py --selftest

Item 93 artefact writer AND the pre-registered verdict rule (sha256-frozen in PREREG.md/FROZEN.sha256 before any counted output; the item refuses to run when a hash differs).

Rows (HTTP, one harness for both engines, the engine's own `timings`): per arm, 3 prompts (conv, prose, code) x 3 repetitions at 700 max tokens; decode tok/s of an arm =
 sum(predicted_n) / sum(predicted_n / predicted_per_second) over its runs (token-weighted). Prefill rows: one cold 16K and one cold 64K prompt, tok/s = timings.prompt_per_second.
 arms: llama_plain (mainline llama.cpp at the best --n-cpu-moe of the pre-registered ladder), llama_ngram (same n, --spec-type ngram-mod), basalt_spec0 (--spec 0), basalt_default (the pack's MTP drafter).
 R0 = basalt_spec0 / llama_plain          (no-draft ratio, THE headline)
 R1 = basalt_default / max(llama_plain, llama_ngram)   (drafting ratio, each engine's best configuration)
VERDICT (on R0, applied as written):
  BROKEN       an engine never produced a clean row set (did not load, or fewer than 6 of 9 runs clean)
  BASALT AHEAD   R0 >= 1.25
  LEVEL          0.80 <= R0 < 1.25
  LLAMA.CPP AHEAD  R0 < 0.80
  INCOMPLETE   the rows exist but fewer than 2 clean takes ran on an engine (exit 3, resumable)
A take is CLEAN = ended done or wall, >= 3 tool calls, request errors <= 20%. A take PASSES = CLEAN and the pristine tests go 5/5 and tests/ untouched (gate.json), reported, never gating the verdict.
"""
import glob
import json
import os
import re
import sqlite3
import sys

ARMS = ("llama_plain", "llama_ngram", "basalt_spec0", "basalt_default")


def verdict_of(r0, clean):
    """clean: dict arm -> number of clean rows (of 9) ; takes_ok: handled by caller"""
    if clean.get("llama_plain", 0) < 6 or clean.get("basalt_spec0", 0) < 6 or r0 is None:
        return "BROKEN"
    if r0 >= 1.25:
        return "BASALT AHEAD"
    if r0 >= 0.80:
        return "LEVEL"
    return "LLAMA.CPP AHEAD"


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
            row["peeked_outside"] = bool(re.search(r"basalt93|b93_|/ref/|PREREG", blob))
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
    cases = [(2.9, {"llama_plain": 9, "basalt_spec0": 9}, "BASALT AHEAD"), (1.25, {"llama_plain": 9, "basalt_spec0": 9}, "BASALT AHEAD"), (1.249, {"llama_plain": 9, "basalt_spec0": 9}, "LEVEL"),
             (0.8, {"llama_plain": 9, "basalt_spec0": 9}, "LEVEL"), (0.799, {"llama_plain": 9, "basalt_spec0": 9}, "LLAMA.CPP AHEAD"), (2.0, {"llama_plain": 5, "basalt_spec0": 9}, "BROKEN"),
             (None, {"llama_plain": 9, "basalt_spec0": 9}, "BROKEN"), (2.0, {"llama_plain": 9, "basalt_spec0": 0}, "BROKEN")]
    bad = 0
    for r0, clean, want in cases:
        got = verdict_of(r0, clean)
        print("selftest", "OK " if got == want else "BAD", want, "got", got)
        bad += got != want
    sys.exit(1 if bad else 0)


def arm_rows(d):
    """d = rows dir of one arm: *.json from oai_bench (decode runs) and pp16k/pp48k"""
    out = {"runs": [], "pp": {}}
    for f in sorted(glob.glob(d + "/dec-*.json")):
        j = json.load(open(f))
        tm = j.get("timings") or {}
        ok = bool(tm) and not j.get("error") and (tm.get("predicted_n") or 0) >= 100 and (tm.get("predicted_per_second") or 0) > 0
        out["runs"].append({"f": os.path.basename(f), "ok": ok, "predicted_n": tm.get("predicted_n"), "tok_s": tm.get("predicted_per_second"), "prompt_n": tm.get("prompt_n"),
                            "draft_n": tm.get("draft_n"), "draft_acc": tm.get("draft_n_accepted"), "finish": j.get("finish_reason")})
    for k in ("pp16k", "pp48k"):
        f = "%s/%s.json" % (d, k)
        if os.path.exists(f):
            j = json.load(open(f))
            out["pp"][k] = {"tok_s": j.get("prefill_tok_s"), "prompt_n": j.get("prompt_n"), "error": j.get("error")}
    good = [r for r in out["runs"] if r["ok"]]
    out["clean"] = len(good)
    out["decode_tok_s"] = round(sum(r["predicted_n"] for r in good) / sum(r["predicted_n"] / r["tok_s"] for r in good), 1) if good else None
    acc = [(r["draft_acc"], r["draft_n"]) for r in good if r.get("draft_n")]
    out["draft_accept"] = round(sum(a for a, n in acc) / max(1, sum(n for a, n in acc)), 3) if acc else None
    return out


def main():
    run, slug = sys.argv[1], sys.argv[2]
    regime = sys.argv[3] if len(sys.argv) > 3 else ""
    arms = {a: arm_rows(run + "/rows/" + a) for a in ARMS if os.path.isdir(run + "/rows/" + a)}
    g = lambda a: (arms.get(a) or {}).get("decode_tok_s")
    r0 = round(g("basalt_spec0") / g("llama_plain"), 2) if g("basalt_spec0") and g("llama_plain") else None
    best_llama = max([x for x in (g("llama_plain"), g("llama_ngram")) if x] or [0]) or None
    r1 = round(g("basalt_default") / best_llama, 2) if g("basalt_default") and best_llama else None
    clean = {a: arms[a]["clean"] for a in arms}
    takes = {}
    for d in sorted(glob.glob(run + "/takes/*")):
        if ".partial" in d:
            continue
        name = os.path.basename(d)
        takes[name] = take_row(d, name.rsplit("-", 1)[0])
    tk = {t: sum(1 for x in takes.values() if x["tag"] == t and x["clean"]) for t in ("llama", "basalt")}
    ov = verdict_of(r0, clean)
    incomplete = ov != "BROKEN" and (tk["llama"] < 2 or tk["basalt"] < 2)
    ladder = json.load(open(run + "/ladder.json")) if os.path.exists(run + "/ladder.json") else {}
    comm = json.load(open(run + "/community.row.json")) if os.path.exists(run + "/community.row.json") else None
    prefill = json.load(open(run + "/prefill.json")) if os.path.exists(run + "/prefill.json") else {}
    race = json.load(open(run + "/out/race.mp4.summary.json")) if os.path.exists(run + "/out/race.mp4.summary.json") else {}
    out = {"slug": slug, "regime": regime, "verdict": ov, "provisional": bool(incomplete), "r0_nodraft_ratio": r0, "r1_draft_ratio": r1, "arms": arms, "clean_rows": clean,
           "best_n_cpu_moe": ladder.get("best_n"), "ladder": ladder, "community_row": comm, "prefill": prefill, "takes": takes, "clean_takes": tk, "race": race,
           "take_config_llama": ladder.get("take_config"), "reference_own_post_tok_s": 59.7,
           "final_clip": os.path.exists(run + "/out/basalt-final.mp4"), "line1": None}
    if ov != "BROKEN" and not incomplete:
        L0, B0 = g("llama_plain"), g("basalt_spec0")
        t1 = {t: sorted([x for x in takes.values() if x["tag"] == t and x["clean"]], key=lambda x: x["dir"])[0] for t in ("llama", "basalt")}
        fin = lambda x: mmss(x["finish_s"]) if x.get("finish_s") is not None else "-"
        tail_ = "; the same Hermes Agent task took %s on Basalt and %s on llama.cpp" % (fin(t1["basalt"]), fin(t1["llama"]))
        if ov == "BASALT AHEAD":
            out["line1"] = "On the identical Qwen3.8-Flash-Next Q2_0 file, Basalt v0.3.0 decodes %.0f tok/s without drafting against llama.cpp's %.0f on one RTX 5090 with 59 GB of RAM: %.1fx%s" % (B0, L0, r0, ("; %.1fx with each engine's drafting" % r1) if r1 else "") + tail_
        elif ov == "LEVEL":
            out["line1"] = "On the identical Qwen3.8-Flash-Next Q2_0 file, Basalt v0.3.0 and llama.cpp decode within a quarter of each other without drafting on one RTX 5090 with 59 GB of RAM (%.0f against %.0f tok/s, %.2fx)%s" % (B0, L0, r0, tail_)
        else:
            out["line1"] = "On the identical Qwen3.8-Flash-Next Q2_0 file, llama.cpp decodes %.0f tok/s without drafting against Basalt v0.3.0's %.0f on one RTX 5090 with 59 GB of RAM: Basalt at %.2fx%s" % (L0, B0, r0, tail_)
    json.dump(out, sys.stdout, indent=1)


if "--selftest" in sys.argv:
    selftest()
main()
