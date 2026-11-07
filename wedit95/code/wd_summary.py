#!/usr/bin/env python3
"""wd_summary.py <RUNDIR> : the pre-registered verdict rule of item 95 (PREREG.md), applied to the saved arm files. Nothing is worded by hand.
RUNDIR/arms/<arm>-<set>.json  (wd_gen.py output: {summary, rows}); arms: base, published, scripted, editor ; sets: dev, held
RUNDIR/take/gate.json         (wd_gate.py: candidates the editor scored, the best by dev, flags)
Prints SUMMARY/VERDICT/LINE1, writes RUNDIR/summary.json when the verdict is not INCOMPLETE (exit 0), else exit 3 (resumes, nothing written).  --selftest : every verdict branch on synthetic counts (BAD on a miss).
Loop = the thinking never closed inside the cap (empty answer field). HELD = 48 GSM8K test problems, ONE try each, same seeds for every arm. "cured" = one-sided Fisher exact p < 0.05 for a LOWER loop rate than the base arm.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wd_common as C  # noqa: E402

ALPHA = 0.05
MIN_BASE_RATE = 0.40


def load_arm(rundir, arm, st):
    p = os.path.join(rundir, "arms", "%s-%s.json" % (arm, st))
    if not os.path.exists(p):
        return None
    s = json.load(open(p))["summary"]
    return s if s["n"] >= (C.N_HELD if st == "held" else C.N_DEV) else None


def decide(H, gate, fl=None):
    """H: {arm: summary|None} for the held-out set; gate: dict|None. -> (verdict, detail, extra)."""
    need = ["base", "published", "scripted"]
    if any(H.get(k) is None for k in need) or gate is None:
        return "INCOMPLETE", "a required arm or the editor take is missing", {}
    b = H["base"]
    if b["loop_rate"] < MIN_BASE_RATE:
        return "INVALID", "the base model's held-out loop rate is %.0f%% (< %.0f%%): the failure did not reproduce under this rig's sampler, nothing to cure" % (100 * b["loop_rate"], 100 * MIN_BASE_RATE), {}

    def cured(arm):
        s = H[arm]
        return C.fisher_less(s["loops"], s["n"], b["loops"], b["n"]) < ALPHA
    if not cured("published"):
        return "EDIT DID NOT REPRODUCE", "the published edit, applied bit-exact to our GGUF, does not lower the held-out loop rate (%d/%d vs %d/%d base): the setup, not the editors, is the finding" % (H["published"]["loops"], H["published"]["n"], b["loops"], b["n"]), {}
    e = H.get("editor")
    if e is None:
        return "NO CANDIDATE", "the editor take ended without a scored candidate (%s)" % gate.get("why", "no wd eval finished"), {}
    ec, sc = cured("editor"), cured("scripted")
    s = H["scripted"]
    e_lower = C.fisher_less(e["loops"], e["n"], s["loops"], s["n"]) < ALPHA
    s_lower = C.fisher_less(s["loops"], s["n"], e["loops"], e["n"]) < ALPHA
    ex = dict(editor_cured=ec, script_cured=sc, editor_sig_lower_than_script=e_lower, script_sig_lower_than_editor=s_lower)
    if ec and sc:
        if e_lower:
            return "EDITOR BEAT THE SCRIPT", "both cure; the editor's edit loops significantly less than the scripted one", ex
        if s_lower:
            return "EDITOR HELPED, SCRIPT BETTER", "both cure; the scripted edit loops significantly less than the editor's", ex
        return "EDITOR MATCHED THE SCRIPT", "both cure; no significant difference between them", ex
    if ec and not sc:
        return "EDITOR CURED, SCRIPT DID NOT", "the editor's edit lowers the loop rate significantly, the scripted one does not", ex
    if sc and not ec:
        return "EDITOR COULD NOT MATCH THE SCRIPT", "the scripted edit lowers the loop rate significantly, the editor's does not", ex
    return "NEITHER CURED IT", "neither the editor's nor the scripted edit lowers the held-out loop rate significantly", ex


def line1(verdict, H, gate):
    def f(arm):
        s = H.get(arm)
        return "%d/%d" % (s["loops"], s["n"]) if s else "?"
    flag = " (take flagged: %s)" % ", ".join(gate.get("flags", [])) if gate and gate.get("flags") else ""
    base = "a local Qwen3.8-27B in Hermes Agent edited a 0.8B model's weights to stop its thinking loops: held-out loops %s base -> %s after the editor's edit; the scripted pipeline: %s, the published edit: %s%s" % (f("base"), f("editor"), f("scripted"), f("published"), flag)
    return {"EDITOR MATCHED THE SCRIPT": base, "EDITOR BEAT THE SCRIPT": base, "EDITOR HELPED, SCRIPT BETTER": base, "EDITOR CURED, SCRIPT DID NOT": base,
            "EDITOR COULD NOT MATCH THE SCRIPT": base, "NEITHER CURED IT": base,
            "NO CANDIDATE": "a local Qwen3.8-27B in Hermes Agent was given a 0.8B model's weights to edit and ended without a scored candidate in the time box; the scripted pipeline on the same model: %s of 48 still loop vs %s base%s" % (f("scripted").split("/")[0], f("base").split("/")[0], flag),
            "EDIT DID NOT REPRODUCE": "the published thinking-loop edit applied to our 0.8B GGUF did not lower the loop rate on %d held-out problems (%s vs %s base): no editor comparison made" % (C.N_HELD, f("published"), f("base")),
            "INVALID": "the 0.8B model's thinking loops did not reproduce on this rig (%s of %d held-out tries looped): nothing to cure" % (f("base").split("/")[0], C.N_HELD)}.get(verdict, "weight-editor run incomplete")


def main(rundir):
    H = {a: load_arm(rundir, a, "held") for a in ("base", "published", "scripted", "editor")}
    D = {a: load_arm(rundir, a, "dev") for a in ("base", "published", "scripted", "editor")}
    gp = os.path.join(rundir, "take", "gate.json")
    gate = json.load(open(gp)) if os.path.exists(gp) else None
    v, d, ex = decide(H, gate)
    for a in ("base", "published", "scripted", "editor"):
        for st, T in (("held", H), ("dev", D)):
            s = T[a]
            if s:
                lo, hi = C.wilson(s["loops"], s["n"])
                print("SUMMARY %s %s: loops %d/%d = %.1f%% (95%% %.1f-%.1f), closed %d, correct %d, mean tokens %s" % (a, st, s["loops"], s["n"], 100 * s["loop_rate"], 100 * lo, 100 * hi, s["closed"], s["correct"], s["mean_tokens"]))
    if gate:
        print("SUMMARY take: candidates scored %s, best by dev %s, flags %s" % (gate.get("scored"), gate.get("best"), gate.get("flags")))
    print("VERDICT %s | %s" % (v, d))
    if v == "INCOMPLETE":
        print("LINE1 weight-editor run incomplete")
        return 3
    l1 = line1(v, H, gate)
    print("LINE1 " + l1)
    json.dump(dict(verdict=v, detail=d, extra=ex, held=H, dev=D, gate=gate, line1=l1), open(os.path.join(rundir, "summary.json"), "w"), indent=1)
    return 0


def selftest():
    def S(loops, n=48):
        return dict(n=n, loops=loops, loop_rate=loops / n, closed=n - loops, correct=n - loops, mean_tokens=5000)
    g = dict(scored=3, best="c3", flags=[])
    cases = [("matched", dict(base=S(44), published=S(10), scripted=S(12), editor=S(14)), g, "EDITOR MATCHED THE SCRIPT"),
             ("beat", dict(base=S(44), published=S(10), scripted=S(34), editor=S(8)), g, "EDITOR BEAT THE SCRIPT"),
             ("helped", dict(base=S(44), published=S(10), scripted=S(4), editor=S(30)), g, "EDITOR HELPED, SCRIPT BETTER"),
             ("editor-only", dict(base=S(44), published=S(10), scripted=S(43), editor=S(12)), g, "EDITOR CURED, SCRIPT DID NOT"),
             ("script-only", dict(base=S(44), published=S(10), scripted=S(12), editor=S(43)), g, "EDITOR COULD NOT MATCH THE SCRIPT"),
             ("neither", dict(base=S(44), published=S(10), scripted=S(43), editor=S(42)), g, "NEITHER CURED IT"),
             ("nocand", dict(base=S(44), published=S(10), scripted=S(12), editor=None), dict(scored=0, why="no wd eval finished", flags=[]), "NO CANDIDATE"),
             ("norepro", dict(base=S(44), published=S(43), scripted=S(12), editor=S(12)), g, "EDIT DID NOT REPRODUCE"),
             ("invalid", dict(base=S(10), published=S(2), scripted=S(2), editor=S(2)), g, "INVALID"),
             ("incomplete", dict(base=S(44), published=None, scripted=S(12), editor=S(12)), g, "INCOMPLETE"),
             ("notake", dict(base=S(44), published=S(10), scripted=S(12), editor=S(12)), None, "INCOMPLETE")]
    bad = 0
    for name, H, gate, want in cases:
        got = decide(H, gate)[0]
        print(("OK " if got == want else "BAD ") + "%s -> %s%s" % (name, got, "" if got == want else " (want %s)" % want))
        bad += got != want
    for v in ("EDITOR MATCHED THE SCRIPT", "NO CANDIDATE", "EDIT DID NOT REPRODUCE", "INVALID", "EDITOR COULD NOT MATCH THE SCRIPT"):
        l = line1(v, dict(base=S(44), published=S(10), scripted=S(12), editor=S(14)), dict(flags=["web"]))
        if not l or "None" in l:
            print("BAD line1 " + v)
            bad += 1
    print("selftest %s" % ("ALL OK" if not bad else "%d BAD" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        sys.exit(selftest())
    sys.exit(main(sys.argv[1]))
