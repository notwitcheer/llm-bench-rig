#!/usr/bin/env python3
"""c94_summary.py <rundir> <slug> <regime> > summary.json   |   c94_summary.py --selftest
Item 94 artefact writer AND the pre-registered verdict rule (sha256-frozen via FROZEN.sha256). See PREREG.md.
R255 = qsa decode at 255K / stock decode at 255K (lru task, seeds 11,12, runs with predicted_n >= 100). QSA AHEAD >= 1.5 | LEVEL 0.8..1.5 | STOCK AHEAD < 0.8 | BROKEN | INCOMPLETE (a Hermes session missing)."""
import glob
import json
import os
import re
import sqlite3
import sys

DEPTHS = (0, 64000, 128000, 255000)
AUTHOR_5090_DECODE = 141.8


def verdict_of(r255, ok_qsa, ok_stock, sessions_ok):
    if r255 is None or ok_qsa < 1 or ok_stock < 1:
        return "BROKEN"
    if not sessions_ok:
        return "INCOMPLETE"
    if r255 >= 1.5:
        return "QSA AHEAD"
    if r255 >= 0.8:
        return "LEVEL"
    return "STOCK AHEAD"


def claim_of(q255):
    if q255 is None:
        return None
    return "ABOVE" if q255 > 156.0 else ("HELD" if q255 >= 127.6 else "BELOW")


def selftest():
    cases = [((2.1, 2, 2, True), "QSA AHEAD"), ((1.5, 2, 2, True), "QSA AHEAD"), ((1.49, 2, 2, True), "LEVEL"), ((0.8, 2, 2, True), "LEVEL"), ((0.79, 2, 2, True), "STOCK AHEAD"),
             ((2.0, 0, 2, True), "BROKEN"), ((None, 2, 2, True), "BROKEN"), ((2.0, 2, 2, False), "INCOMPLETE")]
    bad = 0
    for args, want in cases:
        got = verdict_of(*args)
        print("selftest", "OK " if got == want else "BAD", want, "got", got)
        bad += got != want
    for q, want in ((141.8, "HELD"), (127.6, "HELD"), (127.5, "BELOW"), (156.0, "HELD"), (156.1, "ABOVE")):
        got = claim_of(q)
        print("selftest", "OK " if got == want else "BAD", "claim", q, want, "got", got)
        bad += got != want
    sys.exit(1 if bad else 0)


def hm(s):
    return "%d:%02d" % (int(s // 60), int(s % 60)) if s is not None else "-"


def rows_of(d):
    out = {}
    for f in sorted(glob.glob(d + "/dec-*.json")):
        m = re.match(r"dec-(\w+)-d(\d+)-s(\d+)\.json", os.path.basename(f))
        if not m:
            continue
        j = json.load(open(f))
        tm = j.get("timings") or {}
        ok = bool(tm) and not j.get("error") and (tm.get("predicted_per_second") or 0) > 0
        out.setdefault((m.group(1), int(m.group(2))), []).append({"seed": int(m.group(3)), "ok": ok, "n": tm.get("predicted_n") or 0, "tok_s": tm.get("predicted_per_second"), "pp": tm.get("prompt_per_second"),
                                                                  "prompt_n": tm.get("prompt_n"), "early": ok and (tm.get("predicted_n") or 0) < 16, "clean": ok and (tm.get("predicted_n") or 0) >= 100,
                                                                  "draft_n": tm.get("draft_n"), "draft_acc": tm.get("draft_n_accepted")})
    return out


def agg(rs):
    g = [r for r in rs if r["clean"]]
    if not g:
        return None
    return round(sum(r["n"] for r in g) / sum(r["n"] / r["tok_s"] for r in g), 1)


def arm(run, name):
    rows = rows_of(run + "/rows/" + name)
    a = {"depths": {}, "clean_255": 0}
    for dpt in DEPTHS:
        rs = rows.get(("lru", dpt), [])
        pp = [r["pp"] for r in rs if r["ok"] and r["pp"] and dpt > 0]
        a["depths"][str(dpt)] = {"decode_tok_s": agg(rs), "runs": len(rs), "clean": sum(r["clean"] for r in rs), "early": sum(r["early"] for r in rs),
                                 "prefill_tok_s": round(sum(pp) / len(pp), 0) if pp else None, "prompt_n": (rs[0]["prompt_n"] if rs else None),
                                 "draft_accept": round(sum(r["draft_acc"] or 0 for r in rs) / max(1, sum(r["draft_n"] or 0 for r in rs)), 3) if rs and any(r["draft_n"] for r in rs) else None}
    a["clean_255"] = a["depths"]["255000"]["clean"]
    pr = rows.get(("essay", 255000), [])
    a["prose_255"] = agg(pr)
    probe = []
    for f in sorted(glob.glob(run + "/rows/" + name + "/probe-*.json")):
        j = json.load(open(f))
        tm = j.get("timings") or {}
        if tm:
            probe.append(tm.get("predicted_n") or 0)
    a["early_stop_probe"] = {"requests": len(probe), "under_16_tokens": sum(1 for n in probe if n < 16)}
    return a


def score_take(td, keys, questions):
    """answers = the assistant text after each typed question (state.db), matched to the key as a whole number"""
    sp = td + "/state/state.db"
    if not os.path.exists(sp):
        return None
    c = sqlite3.connect(sp)
    msgs = [(r, (x or "")) for r, x in c.execute("select role, content from messages order by id")]
    turns, cur = [], None
    for r, x in msgs:
        if r == "user":
            cur = {"q": x, "a": ""}
            turns.append(cur)
        elif r == "assistant" and cur is not None and x:
            cur["a"] += " " + x
    res, byq = [], {}
    for i, q in enumerate(questions):
        if i >= len(turns):
            res.append({"id": q["id"], "answered": False, "ok": False})
            continue
        want = keys[q["id"]]["value"]
        nums = re.findall(r"(?<![\d.])\d[\d_,]*(?:\.\d+)?", turns[i]["a"])
        norm = lambda s: s.replace(",", "").replace("_", "")
        try:
            ok = any(abs(float(norm(n)) - float(norm(want))) < 1e-9 for n in nums)
        except ValueError:
            ok = False
        res.append({"id": q["id"], "answered": True, "ok": ok, "depth": q["depth"]})
    byd = {}
    for x in res:
        if x.get("depth") is not None:
            byd.setdefault("early" if x["depth"] < 0.45 else "late", []).append(x["ok"])
    return {"asked": len(turns), "answered": sum(1 for x in res if x["answered"]), "correct": sum(1 for x in res if x["ok"]), "of": len(questions),
            "by_depth": {k: "%d/%d" % (sum(v), len(v)) for k, v in byd.items()}}


def take_info(td):
    """finish second, tokens, wall from the recorder marks + proxy hud"""
    ev = td + "/rec/events-a.jsonl"
    mk = {}
    if os.path.exists(ev):
        for l in open(ev):
            j = json.loads(l)
            if "mark" in j:
                mk.setdefault(j["mark"], j["t"])
                if j["mark"] == "end":
                    mk["end_reason"] = j.get("reason")
    row = {"dir": os.path.basename(td), "end": mk.get("end_reason")}
    if "enter" in mk:
        row["finish_s"] = round((mk["last_change"] if mk.get("end_reason") == "done" and "last_change" in mk else mk.get("end", mk["enter"])) - mk["enter"], 1)
    hp = td + "/hud/hud.jsonl"
    if os.path.exists(hp):
        hud = [json.loads(l) for l in open(hp) if l.strip()]
        if hud:
            row["requests"], row["request_errors"] = hud[-1]["agents"]["a"]["reqs"], hud[-1]["agents"]["a"]["err"]
            row["ctx_tokens_max"] = max([r.get("ctx_tokens") or 0 for r in hud] or [0])
            row["vram_mib_max"] = max([r.get("vram_mib") or 0 for r in hud] or [0])
    rp = td + "/hud/requests.jsonl"
    if os.path.exists(rp):
        rq = [json.loads(l) for l in open(rp) if l.strip()]
        first = [r for r in rq if r.get("prompt_n") and r["prompt_n"] > 100000]
        if first:
            row["first_prefill_s"] = round(first[0]["t1"] - first[0]["t0"], 1)
            row["first_prompt_n"] = first[0]["prompt_n"]
        dq = [r for r in rq if r.get("pred_ps") and r.get("predicted_n")]
        if dq:
            row["decode_tok_s_engine"] = round(sum(r["predicted_n"] for r in dq) / sum(r["predicted_n"] / r["pred_ps"] for r in dq), 1)
        row["cache_hit_tokens"] = sum(r.get("cache_n") or 0 for r in rq)
    pw = td + "/hud/power.csv"
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
            row["gpu_w_mean"] = round(sum(w) / len(w), 1)
    row["clean"] = bool(row["end"] in ("done", "wall") and row.get("requests", 0) > 0 and row.get("request_errors", 0) <= 0.2 * max(row.get("requests", 1), 1))
    return row


def main():
    run, slug = sys.argv[1], sys.argv[2]
    regime = sys.argv[3] if len(sys.argv) > 3 else ""
    H = os.path.dirname(os.path.abspath(__file__))
    qs = json.load(open(H + "/qa/questions.json"))
    keys = json.load(open(H + "/qa/keys/keys.json"))
    arms = {n: arm(run, n) for n in ("qsa", "stock") if os.path.isdir(run + "/rows/" + n)}
    g = lambda n, d: ((arms.get(n) or {}).get("depths", {}).get(str(d)) or {}).get("decode_tok_s")
    q255, s255 = g("qsa", 255000), g("stock", 255000)
    r255 = round(q255 / s255, 2) if q255 and s255 else None
    sess = {}
    for tag in ("qsa", "stock"):
        ds = sorted(glob.glob(run + "/takes/%s-*" % tag))
        ds = [d for d in ds if ".partial" not in d]
        if ds:
            r = take_info(ds[0])
            r["score"] = score_take(ds[0], keys, qs)
            sess[tag] = r
    sessions_ok = all(sess.get(t, {}).get("clean") for t in ("qsa", "stock"))
    ov = verdict_of(r255, (arms.get("qsa") or {}).get("clean_255", 0), (arms.get("stock") or {}).get("clean_255", 0), sessions_ok)
    out = {"slug": slug, "regime": regime, "verdict": ov, "provisional": ov == "INCOMPLETE", "r255": r255, "claim_check": claim_of(q255), "author_5090_decode": AUTHOR_5090_DECODE, "arms": arms, "sessions": sess,
           "final_clip": os.path.exists(run + "/out/qsa-final.mp4"), "line1": None}
    if ov in ("QSA AHEAD", "LEVEL", "STOCK AHEAD") and sess.get("qsa", {}).get("score"):
        sc = sess["qsa"]["score"]
        tail = "; Hermes Agent answered %d of %d questions about the whole docs set from one prompt" % (sc["correct"], sc["of"])
        pre = sess["qsa"].get("first_prefill_s")
        if ov == "QSA AHEAD":
            out["line1"] = "On one RTX 5090, Qwen3.8-27B with Qwen Sparse Attention decodes %.0f tok/s with 255K tokens of context against %.0f tok/s for the dense-attention build of the same model on stock llama.cpp: %.1fx%s" % (q255, s255, r255, tail)
        elif ov == "LEVEL":
            out["line1"] = "On one RTX 5090, Qwen3.8-27B with Qwen Sparse Attention and the dense-attention build decode at about the same speed with 255K tokens of context (%.0f against %.0f tok/s, %.2fx)%s" % (q255, s255, r255, tail)
        else:
            out["line1"] = "On one RTX 5090 the dense-attention build of Qwen3.8-27B on stock llama.cpp decodes %.0f tok/s with 255K tokens of context against %.0f for the QSA fork (%.2fx)%s" % (s255, q255, r255, tail)
    json.dump(out, sys.stdout, indent=1)


if "--selftest" in sys.argv:
    selftest()
main()
