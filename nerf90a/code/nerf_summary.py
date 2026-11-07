#!/usr/bin/env python3
"""nerf_summary.py <RUNDIR> : applies the pre-registered rule (PREREG.md) to the saved per-item results; nothing is worded by hand.
RUNDIR holds: results/<build>-<file>/<task>.jsonl (build old|new), facts.json (hashes, HF facts), and the day-0 comparators come from ../day0 via nerf_common.
Prints SUMMARY lines + VERDICT/LINE1 and writes RUNDIR/summary.json (only when the headline pass is complete). Exit 0 = verdict written, 3 = INCOMPLETE (nothing written).
--selftest : nine verdict branches on synthetic data (CPU, seconds); prints BAD on any miss.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nerf_common as N  # noqa: E402

P_SHIFT = 0.01          # a task counts as moved only when outside the day-0 band AND its test p is below this
HEAD = ("new", "q6k")   # headline pass: today's Donald build, the file Donald serves


def two_prop_p(c1, n1, c2, n2):
    import math
    p = (c1 + c2) / (n1 + n2)
    se = math.sqrt(max(p * (1 - p) * (1 / n1 + 1 / n2), 1e-12))
    z = abs(c1 / n1 - c2 / n2) / se
    return math.erfc(z / math.sqrt(2))


def task_stats(res, build, tag, task, d0):
    """res: key->rec for today's pass. d0: day0_task dict. -> stats dict or None when the pass has no items."""
    if not res:
        return None
    n_exp = d0["n"]
    c = sum(1 for r in res.values() if r["ok"])
    n = len(res)
    s = dict(task=task, n=n, n_expected=n_exp, complete=(n == n_exp), c=c, score=100.0 * c / n, d0_c=d0["c"], d0_n=d0["n"], d0_score=100.0 * d0["c"] / d0["n"])
    lo, hi = N.wilson(d0["c"], d0["n"])
    tl, th = N.wilson(c, n)
    s.update(d0_lo=100 * lo, d0_hi=100 * hi, lo=100 * tl, hi=100 * th)
    s["outside"] = bool(s["score"] < s["d0_lo"] or s["score"] > s["d0_hi"])
    s["direction"] = "down" if s["score"] < s["d0_lo"] else ("up" if s["score"] > s["d0_hi"] else "")
    if d0.get("items") is not None and n == n_exp:
        keys = sorted(d0["items"])
        b = sum(1 for k in keys if d0["items"][k] and not res[k]["ok"])      # day-0 right, today wrong
        cc = sum(1 for k in keys if (not d0["items"][k]) and res[k]["ok"])    # day-0 wrong, today right
        s.update(paired=True, b=b, c_only=cc, p=N.mcnemar_exact(b, cc), both_right=sum(1 for k in keys if d0["items"][k] and res[k]["ok"]),
                 both_wrong=sum(1 for k in keys if (not d0["items"][k]) and (not res[k]["ok"])))
        if task == "gpqa":
            same = sum(1 for k in keys if res[k]["p"] == d0["pred"][k])
            s["same_answer"] = same
        else:
            s["same_answer"] = s["both_right"] + s["both_wrong"]
    else:
        s.update(paired=False, p=two_prop_p(c, n, d0["c"], d0["n"]))
    s["moved"] = bool(s["outside"] and s["p"] < P_SHIFT)
    return s


def agreement(resA, resB):
    """same-answer count between two passes of the same task (predicted letter / pass-fail)."""
    ks = sorted(set(resA) & set(resB))
    return dict(n=len(ks), same=sum(1 for k in ks if resA[k]["p"] == resB[k]["p"]))


def analyse(rundir, results_dir=None):
    resdir = results_dir or os.path.join(rundir, "results")
    facts = json.load(open(os.path.join(rundir, "facts.json")))
    out = dict(passes={}, agreement={}, facts=facts)
    for tag in N.FILES:
        for build in ("new", "old"):
            tasks = {}
            for t in N.TASKS:
                res = N.read_arm(resdir, build, tag, t)
                st = task_stats(res, build, tag, t, N.day0_task(tag, t))
                if st:
                    tasks[t] = st
            if tasks:
                out["passes"]["%s-%s" % (build, tag)] = tasks
        for t in N.TASKS:
            a, b = N.read_arm(resdir, "new", tag, t), N.read_arm(resdir, "old", tag, t)
            if a and b:
                out["agreement"]["%s-%s" % (tag, t)] = agreement(a, b)
    return out


def pass_verdict(tasks, hash_ok):
    """-> (word, detail). tasks: {task: stats}."""
    if any(t not in tasks or not tasks[t]["complete"] for t in N.TASKS):
        return "INCOMPLETE", "not every item of the four tasks ran"
    if not hash_ok:
        return "CHANGED FILE", "sha256 differs from the LFS oid recorded for the day-0 commit"
    moved = [t for t in N.TASKS if tasks[t]["moved"]]
    if moved:
        dirs = {tasks[t]["direction"] for t in moved}
        word = "SHIFTED DOWN" if dirs == {"down"} else ("SHIFTED UP" if dirs == {"up"} else "SHIFTED MIXED")
        return word, "moved: " + ", ".join(moved)
    edge = [t for t in N.TASKS if tasks[t]["outside"]]
    return "SAME FILE, SAME SCORES", ("%d task(s) at the edge of the band, none significant" % len(edge)) if edge else "all four tasks inside the day-0 band"


def decide(an):
    f = an["facts"]
    hok = bool(f.get("hash_ok", {}).get("q6k"))
    head = an["passes"].get("%s-%s" % HEAD)
    if not head:
        return dict(verdict="INCOMPLETE", detail="headline pass (Q6_K on the Donald build) has no items")
    word, detail = pass_verdict(head, hok)
    r = dict(verdict=word, detail=detail)
    # attribution: only when the headline moved and the day-0 build re-run is complete
    old = an["passes"].get("old-q6k")
    if word.startswith("SHIFTED") and old and all(t in old and old[t]["complete"] for t in N.TASKS):
        ow, _ = pass_verdict(old, hok)
        r["attribution"] = ("the day-0 build re-run is %s: the difference is the runtime build, not the weights" % ow) if ow == "SAME FILE, SAME SCORES" else \
                           ("the day-0 build re-run is also %s: the difference is not the build alone" % ow)
    # secondary passes are reported with their own words, never gating
    r["secondary"] = {}
    for key in ("old-q6k", "new-q4km", "old-q4km"):
        if key in an["passes"]:
            tag = key.split("-")[1]
            r["secondary"][key] = pass_verdict(an["passes"][key], bool(f.get("hash_ok", {}).get(tag)))[0]
    return r


def line1(an, dec):
    h = an["passes"].get("%s-%s" % HEAD, {})
    g, he = h.get("gpqa"), h.get("he")
    parts = []
    if g:
        parts.append("GPQA %d/%d vs %d/%d on day 0" % (g["c"], g["n"], g["d0_c"], g["d0_n"]))
    if he:
        parts.append("HumanEval %d/%d vs %d/%d" % (he["c"], he["n"], he["d0_c"], he["d0_n"]))
    v = dec["verdict"]
    if v == "SAME FILE, SAME SCORES":
        return "Qwen3.8-27B was not nerfed: the Q6_K file is byte-identical to August and the same board gives the same scores (%s)." % ", ".join(parts)
    if v.startswith("SHIFTED"):
        way = {"DOWN": "lower", "UP": "higher", "MIXED": "mixed"}[v.split(" ", 1)[1]]
        return "Qwen3.8-27B file unchanged since August, yet today's runtime gives %s scores on the same items (%s): %s." % (way, ", ".join(parts), dec.get("attribution", "the weights are not the cause (hash)"))
    if v == "CHANGED FILE":
        return "the Qwen3.8-27B Q6_K on this box no longer matches the file we benchmarked in August (sha256 differs)."
    return "audit incomplete."


def main(rundir):
    an = analyse(rundir)
    dec = decide(an)
    for k, tasks in an["passes"].items():
        for t, s in tasks.items():
            print("SUMMARY %s %s: %d/%d = %.2f%% (day-0 %d/%d = %.2f%%, band %.2f-%.2f)%s %s" % (k, t, s["c"], s["n"], s["score"], s["d0_c"], s["d0_n"], s["d0_score"], s["d0_lo"], s["d0_hi"],
                  " paired same-answer %d/%d p=%.3f" % (s["same_answer"], s["n"], s["p"]) if s.get("paired") else " unpaired p=%.3f" % s["p"], "OUTSIDE" if s["outside"] else "inside"))
    for k, a in an["agreement"].items():
        print("SUMMARY agreement old-vs-new %s: same answer %d/%d" % (k, a["same"], a["n"]))
    print("VERDICT %s | %s" % (dec["verdict"], dec["detail"]))
    if dec.get("attribution"):
        print("ATTRIBUTION %s" % dec["attribution"])
    for k, w in dec.get("secondary", {}).items():
        print("SECONDARY %s %s" % (k, w))
    if dec["verdict"] == "INCOMPLETE":
        print("LINE1 audit incomplete")
        json.dump(dict(an=an, decision=dec), open(os.path.join(rundir, "summary.pre.json"), "w"), indent=1)
        return 3
    l1 = line1(an, dec)
    print("LINE1 " + l1)
    json.dump(dict(analysis=an, decision=dec, line1=l1), open(os.path.join(rundir, "summary.json"), "w"), indent=1)
    return 0


# ------------------------------------------------------------------------------------------------ self test
def selftest():
    import tempfile, random, shutil
    sys.path.insert(0, HERE)
    import make_mock_results as M
    ok = 0
    cases = [("same", "SAME FILE, SAME SCORES"), ("edge", "SAME FILE, SAME SCORES"), ("down", "SHIFTED DOWN"), ("up", "SHIFTED UP"), ("mixed", "SHIFTED MIXED"),
             ("changed", "CHANGED FILE"), ("partial", "INCOMPLETE"), ("buildshift", "SHIFTED DOWN"), ("nohead", "INCOMPLETE")]
    for name, want in cases:
        d = tempfile.mkdtemp(prefix="nerfst")
        M.make(d, name)
        an = analyse(d)
        got = decide(an)["verdict"]
        if got == want:
            ok += 1
            print("OK %s -> %s" % (name, got))
        else:
            print("BAD %s: got %s want %s" % (name, got, want))
        shutil.rmtree(d, ignore_errors=True)
    # attribution branch
    d = tempfile.mkdtemp(prefix="nerfst")
    M.make(d, "buildshift")
    dec = decide(analyse(d))
    if "runtime build" in dec.get("attribution", ""):
        print("OK attribution build"); ok += 1
    else:
        print("BAD attribution: %s" % dec.get("attribution"))
    shutil.rmtree(d, ignore_errors=True)
    print("selftest %d/10" % ok)
    return 0 if ok == 10 else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        sys.exit(selftest())
    sys.exit(main(sys.argv[1]))
