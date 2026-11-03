#!/usr/bin/env python3
"""desk_run.py: run the 8 desktop-arena tasks for ONE entrant against an ALREADY RUNNING OpenAI-compatible server (the item script starts/stops the model).

usage: desk_run.py --home <HERMES_HOME made by desk_mkhome.py> --label <entrant slug> --out <dir> [--tasks t1,t2,..] [--wall 420] [--fps 3]
per task: restart the entrant's desktop (computer-use screen stop --force / start: a fresh Xfce), reset + seed the task state in the container (harness, docker exec),
grab frames of the agent's own display at --fps while `hermes -t computer_use chat -q <prompt> -Q` runs (hard wall cap, own process group only), then SCORE from the
resulting file state (desk_tasks.score, read through docker exec: the gate is outside the agent) and append one JSON line to <out>/results.jsonl:
  {label, task, pass, detail, secs, timeout, tool_calls, tool_errors, actions, session, frames}
frames land in <out>/frames/<task>/NNNNN.jpg (+ times.txt: frame index -> seconds since the agent started). Never starts/stops a model server, never touches :8090, never
kills by name: the only kill is the process group of the hermes child this script started.
"""
import argparse
import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import desk_lib as L  # noqa: E402
import desk_tasks as T  # noqa: E402

HBIN = os.environ.get("FR_HBIN", os.path.expanduser("~/.hermes/hermes-agent/.hermes/bin/hermes"))


def henv(home):
    e = dict(os.environ)
    e["HERMES_HOME"] = home
    e["HERMES_DISABLE_LAZY_INSTALLS"] = "1"
    return e


def hermes(home, *args, timeout=180):
    r = subprocess.run([HBIN] + list(args), env=henv(home), capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def restart_desktop(home):
    hermes(home, "computer-use", "screen", "stop", "--force", timeout=90)
    time.sleep(1.0)
    rc, out = hermes(home, "computer-use", "screen", "start", timeout=150)
    return rc == 0 and "running on DISPLAY" in out, out.strip()[-200:]


class Grabber(threading.Thread):
    def __init__(self, c, d, fps):
        super().__init__(daemon=True)
        self.c, self.d, self.dt, self.stop_ev, self.n = c, d, 1.0 / fps, threading.Event(), 0
        os.makedirs(d, exist_ok=True)
        self.t0 = time.time()
        self.times = open(os.path.join(d, "times.txt"), "w")

    def run(self):
        while not self.stop_ev.is_set():
            t = time.time()
            if L.grab(self.c, os.path.join(self.d, "%05d.jpg" % self.n)):
                self.times.write("%d %.2f\n" % (self.n, t - self.t0))
                self.times.flush()
                self.n += 1
            left = self.dt - (time.time() - t)
            if left > 0:
                self.stop_ev.wait(left)

    def finish(self):
        self.stop_ev.set()
        self.join(timeout=10)
        self.times.close()


def session_stats(home, sid):
    out = {"tool_calls": 0, "tool_errors": 0, "actions": {}}
    try:
        c = sqlite3.connect(os.path.join(home, "state.db"))
        for role, name, content, tcalls in c.execute("select role, tool_name, content, tool_calls from messages where session_id=? order by timestamp", (sid,)):
            if role == "tool":
                out["tool_calls"] += 1
                s = (content or "")[:600]
                if '"error"' in s[:200] or '"ok": false' in s[:200]:
                    out["tool_errors"] += 1
            if role == "assistant" and tcalls:
                try:
                    for tc in json.loads(tcalls):
                        a = json.loads(tc["function"]["arguments"]).get("action", "?")
                        out["actions"][a] = out["actions"].get(a, 0) + 1
                except Exception:
                    pass
    except Exception as e:
        out["stats_error"] = str(e)[:100]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--wall", type=int, default=420)
    ap.add_argument("--fps", type=float, default=3.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    want = [x for x in a.tasks.split(",") if x] or [t["id"] for t in T.TASKS]
    res_path = os.path.join(a.out, "results.jsonl")
    ok, msg = restart_desktop(a.home)
    if not ok:
        print("FAIL desktop did not start: %s" % msg)
        return 1
    c = L.cid(a.home)
    for t in T.TASKS:
        if t["id"] not in want:
            continue
        ok, msg = restart_desktop(a.home)
        if not ok:
            print("FAIL desktop restart before %s: %s" % (t["id"], msg))
            return 1
        c = L.cid(a.home)
        T.reset(c)
        t["seed"](c)
        time.sleep(3.0)
        fdir = os.path.join(a.out, "frames", t["id"])
        g = Grabber(c, fdir, a.fps)
        g.start()
        prompt = ("You are operating a Linux desktop (Xfce) with the computer_use tool. " + t["prompt"] + " When it is done, reply with one short sentence.")
        log = open(os.path.join(a.out, "agent-%s.log" % t["id"]), "w")
        t0 = time.time()
        p = subprocess.Popen([HBIN, "-t", "computer_use", "chat", "-q", prompt, "-Q"], env=henv(a.home), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        timeout = False
        try:
            p.wait(timeout=a.wall)
        except subprocess.TimeoutExpired:
            timeout = True
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
        secs = time.time() - t0
        time.sleep(1.0)
        g.finish()
        log.close()
        txt = open(os.path.join(a.out, "agent-%s.log" % t["id"])).read()
        m = re.search(r"session_id:\s*(\S+)", txt)
        sid = m.group(1) if m else ""
        if not sid:  # a wall-capped run is killed before it prints its session id: take the newest session this task started
            try:
                row = sqlite3.connect(os.path.join(a.home, "state.db")).execute("select id from sessions where started_at >= ? order by started_at desc limit 1", (t0 - 2,)).fetchone()
                sid = row[0] if row else ""
            except Exception:
                sid = ""
        passed, detail = t["score"](c)
        row = {"label": a.label, "task": t["id"], "title": t["title"], "pass": bool(passed), "detail": str(detail)[:200], "secs": round(secs, 1), "timeout": timeout,
               "session": sid, "frames": g.n}
        row.update(session_stats(a.home, sid) if sid else {"tool_calls": 0, "tool_errors": 0, "actions": {}})
        open(res_path, "a").write(json.dumps(row) + "\n")
        print("%s %s pass=%s %.0fs calls=%s err=%s timeout=%s" % (a.label, t["id"], passed, secs, row["tool_calls"], row["tool_errors"], timeout), flush=True)
        T.reset(c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
