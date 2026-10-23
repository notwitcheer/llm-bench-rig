#!/usr/bin/env python3
"""agenttest runner: one fresh Hermes Agent session per frozen task, scored by the python checkers.

Shape copied from the Muse Glimmer run (2026-10-20): isolated HERMES_HOME (provider custom, base_url
on an ssh -L tunnel to the capsule llama-server, 64K context), default approvals, non-TTY
`hermes chat -q <prompt> --format stream-json --source tool`, one run per task, seeds generated once
before the first task. Metrics come from the isolated home's state.db.

usage: run_agent.py --home /opt/data/tmp/sparkagent/home --out results/<run> [--smoke] [--tasks t01,t02]
"""
import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "checkers"))
import gen_seeds  # noqa: E402
import importlib  # noqa: E402

HERMES = "/opt/hermes/.venv/bin/hermes"
MALFORMED_MARKERS = ("does not exist. Available tools", "Invalid JSON arguments", "invalid_tool_call")


def is_tool_error(content: str) -> bool:
    c = content or ""
    try:
        d = json.loads(c)
    except Exception:  # noqa: BLE001
        d = None
    if isinstance(d, dict):
        if d.get("status") == "error" or d.get("success") is False:
            return True
        if d.get("error") not in (None, "", False):
            return True
        ec = d.get("exit_code")
        return isinstance(ec, int) and ec != 0
    if re.search(r'"error":\s*"[^"]', c):
        return True
    return c.startswith("Error") or any(m in c for m in MALFORMED_MARKERS)


def session_metrics(db, sid):
    cur = db.cursor()
    s = dict(zip([r[1] for r in cur.execute("pragma table_info(sessions)")],
                 cur.execute("select * from sessions where id=?", (sid,)).fetchone()))
    msgs = cur.execute("select role, content, tool_calls, tool_name, reasoning, reasoning_content, finish_reason "
                       "from messages where session_id=? order by id", (sid,)).fetchall()
    calls, names, errs, malformed, err_detail, reasoning_chars, assistant = 0, [], 0, 0, [], 0, 0
    final = ""
    for role, content, tc, tn, rs, rc, fin in msgs:
        if role == "assistant":
            assistant += 1
            reasoning_chars += len(rs or "") + len(rc or "")
            if tc:
                lst = json.loads(tc)
                calls += len(lst)
                names += [x.get("function", {}).get("name") for x in lst]
            elif content:
                final = content
                if "<tool_call" in content or re.search(r'"name"\s*:\s*"\w+"\s*,\s*"arguments"', content):
                    malformed += 1
                    err_detail.append({"tool": None, "malformed": True, "content": content[:300]})
        elif role == "tool":
            if any(m in (content or "") for m in MALFORMED_MARKERS):
                malformed += 1
            if is_tool_error(content):
                errs += 1
                err_detail.append({"tool": tn, "content": (content or "")[:300]})
    return {
        "session_id": sid, "tool_calls": calls, "tool_names": names, "tool_errors": errs,
        "malformed_calls": malformed, "error_detail": err_detail, "assistant_msgs": assistant,
        "api_calls": s.get("api_call_count"), "input_tokens": s.get("input_tokens"),
        "output_tokens": s.get("output_tokens"), "cache_read_tokens": s.get("cache_read_tokens"),
        "reasoning_tokens_reported": s.get("reasoning_tokens"), "reasoning_chars": reasoning_chars,
        "end_reason": s.get("end_reason"), "final_text": final[:600],
    }


def run_one(home, work, tid, prompt, out, budget):
    env = dict(os.environ, HERMES_HOME=home)
    cmd = [HERMES, "chat", "-q", prompt, "--format", "stream-json", "--source", "tool", "--run-budget", str(budget)]
    t0 = time.time()
    with open(f"{out}/{tid}.jsonl", "w") as so, open(f"{out}/{tid}.err", "w") as se:
        try:
            rc = subprocess.run(cmd, cwd=work, env=env, stdin=subprocess.DEVNULL, stdout=so, stderr=se,
                                timeout=budget + 120).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    return rc, t0, time.time()


def find_session(db, prompt, t0, t1):
    rows = db.execute("select s.id from sessions s join messages m on m.session_id=s.id where m.role='user' "
                      "and m.content=? and s.started_at between ? and ? order by s.started_at", (prompt, t0 - 5, t1 + 5)).fetchall()
    return rows[-1][0] if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--budget", type=int, default=900)
    ap.add_argument("--no-seed", action="store_true")
    a = ap.parse_args()
    spec = json.load(open(os.path.join(HERE, "tasks.json")))
    work = spec["work_root"]
    os.makedirs(a.out, exist_ok=True)
    base = re.sub(r"/v1/?$", "", open(os.path.join(a.home, "config.yaml")).read().split("base_url:")[1].split()[0])
    print("endpoint models:", urllib.request.urlopen(base + "/v1/models", timeout=10).read()[:200])
    if not a.no_seed:
        json.dump(gen_seeds.build(work), open(f"{a.out}/seed-manifest.json", "w"), indent=1)
    todo = [{"id": "smoke", "prompt": spec["smoke_prompt"], "label": "smoke"}] if a.smoke else []
    todo += [t for t in spec["tasks"] if a.tasks == "all" or t["id"] in a.tasks.split(",")]
    db_path = os.path.join(a.home, "state.db")
    results = []
    for t in todo:
        rc, t0, t1 = run_one(a.home, work, t["id"], t["prompt"], a.out, a.budget)
        db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        sid = find_session(db, t["prompt"], t0, t1)
        m = session_metrics(db, sid) if sid else {"session_id": None}
        db.close()
        if t["id"] == "smoke":
            ok, why = os.path.isfile(f"{work}/smoke/s.txt"), "smoke file exists"
        else:
            ok, why = importlib.import_module(f"check_{t['id']}").check(work)
        r = {"id": t["id"], "label": t["label"], "pass": ok, "check_reason": why, "rc": rc,
             "wall_s": round(t1 - t0, 1), "started_utc": time.strftime("%H:%M:%S", time.gmtime(t0)), **m}
        json.dump(r, open(f"{a.out}/{t['id']}.result.json", "w"), indent=1)
        results.append(r)
        print(f"{t['id']} {'PASS' if ok else 'FAIL'} wall={r['wall_s']}s calls={m.get('tool_calls')} "
              f"errs={m.get('tool_errors')} malformed={m.get('malformed_calls')} rc={rc} :: {why}", flush=True)
    graded = [r for r in results if r["id"] != "smoke"]
    if graded:
        summ = {"passed": sum(r["pass"] for r in graded), "n": len(graded),
                "tool_calls": sum(r.get("tool_calls") or 0 for r in graded),
                "tool_errors": sum(r.get("tool_errors") or 0 for r in graded),
                "malformed_calls": sum(r.get("malformed_calls") or 0 for r in graded),
                "wall_s_total": round(sum(r["wall_s"] for r in graded), 1), "tasks": graded}
        json.dump(summ, open(f"{a.out}/results.json", "w"), indent=1)
        print(f"SCORE {summ['passed']}/{summ['n']} calls={summ['tool_calls']} errs={summ['tool_errors']} "
              f"malformed={summ['malformed_calls']} wall={summ['wall_s_total']}s")


if __name__ == "__main__":
    main()
