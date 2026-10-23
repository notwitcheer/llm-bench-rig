#!/usr/bin/env python3
"""hand solutions for the 10 agenttest tasks (what a correct agent would leave on disk).
used ONLY by prove_checkers.py to show every checker can pass. near_miss() writes plausible wrong
answers to show the checkers are not trivially satisfied."""
import os
import subprocess


def w(work, rel, s):
    p = os.path.join(work, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").write(s)


def solve(tid, work):
    if tid == "t01":
        w(work, "t01/hello.txt", "muse glimmer was here\n")
    elif tid == "t02":
        out = subprocess.run(["awk", "-F,", "NR>1 {s+=$3} END {printf \"%.2f\", s}", f"{work}/t02/sales.csv"],
                             capture_output=True, text=True, check=True).stdout
        w(work, "t02/total.txt", out)
    elif tid == "t03":
        out = subprocess.run(["grep", "-l", "blue heron protocol", "-r", f"{work}/t03/docs"],
                             capture_output=True, text=True).stdout.strip()
        w(work, "t03/answer.txt", os.path.basename(out) + "\n")
    elif tid == "t04":
        p = f"{work}/t04/stats.py"
        s = open(p).read().replace("sum(xs) / (len(xs) - 1)", "sum(xs) / len(xs)")
        open(p, "w").write(s)
    elif tid == "t05":
        for fn, word in (("a.txt", "alpha"), ("b.txt", "beta"), ("c.txt", "gamma")):
            w(work, f"t05/project/{fn}", word + "\n")
        w(work, "t05/project/index.txt", "a.txt\nb.txt\nc.txt\n")
    elif tid == "t06":
        w(work, "t06/fact.txt", "Hermes Agent\n")
    elif tid == "t07":
        out = subprocess.run(["grep", "-c", "-w", "ERROR", f"{work}/t07/app.log"], capture_output=True, text=True).stdout
        w(work, "t07/count.txt", out)
    elif tid == "t08":
        import json
        p = f"{work}/t08/config.json"
        d = json.load(open(p))
        d["server"]["port"] = 8443
        d["logging"]["level"] = "debug"
        open(p, "w").write(json.dumps(d, indent=2) + "\n")
    elif tid == "t09":
        os.makedirs(f"{work}/t09/out", exist_ok=True)
        subprocess.run(["bash", f"{work}/t09/run_report.sh"], check=True, capture_output=True)
    elif tid == "t10":
        w(work, "t10/summary.txt",
          "- launch is set for 14 november, screenshots due 3 november.\n"
          "- the approved budget for the quarter is 48,000 euros.\n"
          "- priya owns the database migration to the new cluster.\n")
    else:
        raise KeyError(tid)


def near_miss(tid, work):
    """a plausible wrong answer per task (each must FAIL its checker)."""
    if tid == "t01":
        w(work, "t01/hello.txt", "Muse glimmer was here.\n")
    elif tid == "t02":
        w(work, "t02/total.txt", "total: 6593.6\n")
    elif tid == "t03":
        w(work, "t03/answer.txt", f"{work}/t03/docs/note_douglas.txt\n")
    elif tid == "t04":
        p = f"{work}/t04/test.py"
        open(p, "w").write("print('all tests passed')\n")
    elif tid == "t05":
        for fn, word in (("a.txt", "alpha"), ("b.txt", "beta"), ("c.txt", "gamma")):
            w(work, f"t05/project/{fn}", word)
        w(work, "t05/project/index.txt", "c.txt\nb.txt\na.txt\n")
    elif tid == "t06":
        w(work, "t06/fact.txt", "Nous Research\n")
    elif tid == "t07":
        out = subprocess.run(["grep", "-c", "-i", "error", f"{work}/t07/app.log"], capture_output=True, text=True).stdout
        w(work, "t07/count.txt", out)
    elif tid == "t08":
        import json
        p = f"{work}/t08/config.json"
        d = json.load(open(p))
        d["server"]["port"] = "8443"
        d["logging"]["level"] = "debug"
        open(p, "w").write(json.dumps(d, indent=2) + "\n")
    elif tid == "t09":
        p = f"{work}/t09/run_report.sh"
        s = open(p).read().replace('[ -d "$OUT" ] ||', 'mkdir -p "$OUT"; [ -d "$OUT" ] ||')
        open(p, "w").write(s)
        subprocess.run(["bash", p], capture_output=True)
    elif tid == "t10":
        w(work, "t10/summary.txt",
          "- launch is set for 14 november.\n\n- the budget is 48,000 euros.\n- priya owns the migration.\n")
