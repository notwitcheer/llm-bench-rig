#!/usr/bin/env python3
"""agenttest v2 row runner (capsule side): one fresh real Hermes Agent session per frozen task against ONE llama-server row.

Called by the queue item (scripts/capsule-queue/73-agenttest-v2-c1.sh) once the row's server is up; also usable by hand for a smoke.
- profile: one Hermes profile per row (hermes profile create, never deleted; a raw HERMES_HOME outside ~/.hermes triggers a dependency build),
  config rewritten per run: provider custom -> the row's llama-server, context_length, memory off, approvals LEFT AT THE DEFAULT (smart; headless blocks are counted separately).
- per task: wipe + seed the task dir, `hermes -p <profile> chat -q <prompt> --format stream-json --source tool` (the per-task budget is agent.run_budget_seconds in the profile config: this install rejects --run-budget on the CLI) with cwd = the task dir, stdin closed,
  then the checker (try/except, a checker error is a fail), metrics from the profile's state.db.
- resumable per task: <out>/<tid>.result.json exists -> skipped. A task cut by the deadline writes no result and the runner exits 3 (the task reruns fresh).
- exit codes: 0 all tasks done (summary.json written), 3 stopped at the deadline (resumable), 4 server not healthy (caller restarts the server and re-invokes), 2 setup error.

usage: run_agent.py --row SLUG --profile NAME --model-id ID --port N --work-root DIR --out DIR [--effort high|unset] [--tasks t01,t02|all]
                    [--deadline-epoch E] [--hermes PATH] [--ctx 65536] [--meta FILE] [--gpu-samples FILE] [--budget-scale 1.0] [--smoke-note TEXT]"""
import argparse
import json
import os
import re
import signal
import sqlite3
import statistics
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402

MALFORMED_MARKERS = ("does not exist. Available tools", "Invalid JSON arguments", "invalid_tool_call")
BLOCK_MARKERS = ("BLOCKED", "blocked by", "requires approval", "approval required", "denied by approval")


def is_tool_error(content):
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
    return c.startswith("Error") or any(m in c for m in MALFORMED_MARKERS) or any(m in c for m in BLOCK_MARKERS)


def session_metrics(db, sid):
    cur = db.cursor()
    cols = [r[1] for r in cur.execute("pragma table_info(sessions)")]
    s = dict(zip(cols, cur.execute("select * from sessions where id=?", (sid,)).fetchone()))
    msgs = cur.execute("select role, content, tool_calls, tool_name, reasoning, reasoning_content, timestamp from messages where session_id=? order by id", (sid,)).fetchall()
    calls, names, errs, blocks, malformed, detail, rchars, first_asst = 0, [], 0, 0, 0, [], 0, None
    final = ""
    for role, content, tc, tn, rs, rc, ts in msgs:
        if role == "assistant":
            rchars += len(rs or "") + len(rc or "")
            if first_asst is None:
                first_asst = ts
            if tc:
                lst = json.loads(tc)
                calls += len(lst)
                names += [x.get("function", {}).get("name") for x in lst]
            elif content:
                final = content
                if "<tool_call" in content or re.search(r'"name"\s*:\s*"\w+"\s*,\s*"arguments"', content):
                    malformed += 1
                    detail.append({"tool": None, "malformed": True, "content": content[:200]})
        elif role == "tool":
            c = content or ""
            if any(m in c for m in MALFORMED_MARKERS):
                malformed += 1
            if is_tool_error(c):
                if any(m in c for m in BLOCK_MARKERS):
                    blocks += 1
                    detail.append({"tool": tn, "approval_block": True, "content": c[:200]})
                else:
                    errs += 1
                    detail.append({"tool": tn, "content": c[:200]})
    return {"session_id": sid, "tool_calls": calls, "tool_names": names, "tool_errors_other": errs, "approval_blocks": blocks, "malformed_calls": malformed,
            "error_detail": detail[:12], "api_calls": s.get("api_call_count"), "input_tokens": s.get("input_tokens"), "output_tokens": s.get("output_tokens"),
            "cache_read_tokens": s.get("cache_read_tokens"), "reasoning_chars": rchars, "end_reason": s.get("end_reason"),
            "first_assistant_s": round(first_asst - s["started_at"], 2) if first_asst and s.get("started_at") else None, "final_text": final[:400]}


def find_session(db, prompt, t0, t1):
    rows = db.execute("select s.id from sessions s join messages m on m.session_id=s.id where m.role='user' and m.content=? and s.started_at between ? and ? order by s.started_at",
                      (prompt, t0 - 5, t1 + 5)).fetchall()
    return rows[-1][0] if rows else None


def profile_home(name):
    return os.path.join(os.path.expanduser("~"), ".hermes", "profiles", name)


def write_config(home, a, budget):
    """per-task config: Hermes on this install rejects --run-budget on the CLI ("use agent.run_budget_seconds in config.yaml"), so the budget lives in the profile config."""
    cfg = f"""model:
  default: {a.model_id}
  provider: custom
  base_url: http://127.0.0.1:{a.port}/v1
  api_key: local-key
  context_length: {a.ctx}
toolsets:
  - hermes-cli
memory:
  memory_enabled: false
  user_profile_enabled: false
telemetry:
  shared_metrics:
    enabled: false
    send: false
    offer_version: 2
agent:
  run_budget_seconds: {budget}
"""
    if a.effort == "high":
        cfg += "  reasoning_effort: high\n"
    open(os.path.join(home, "config.yaml"), "w").write(cfg)


def setup_profile(a):
    home = profile_home(a.profile)
    if not os.path.isdir(home):
        p = subprocess.run([a.hermes, "profile", "create", a.profile, "--no-alias", "--description", f"agenttest v2 L1 row {a.row} (isolated, bench)"],
                           capture_output=True, text=True, timeout=300)
        if not os.path.isdir(home):
            print("profile create failed:", p.stdout[-300:], p.stderr[-300:])
            sys.exit(2)
    write_config(home, a, 600)
    return home


def run_one(a, T, tid, prompt, budget, timeout_s):
    env = dict(os.environ)
    env.pop("HERMES_HOME", None)
    write_config(profile_home(a.profile), a, budget)
    cmd = [a.hermes, "-p", a.profile, "chat", "-q", prompt, "--format", "stream-json", "--source", "tool"]
    t0 = time.time()
    with open(f"{a.out}/{tid}.jsonl", "w") as so, open(f"{a.out}/{tid}.err", "w") as se:
        p = subprocess.Popen(cmd, cwd=T, env=env, stdin=subprocess.DEVNULL, stdout=so, stderr=se, start_new_session=True)
        try:
            rc = p.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            rc = "timeout"
    return rc, t0, time.time()


def healthy(port):
    try:
        return b"ok" in urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5).read().lower()
    except Exception:  # noqa: BLE001
        return False


def vram_stats(path):
    try:
        vals = [int(x.split(",")[0].strip()) for x in open(path) if x.strip() and x.split(",")[0].strip().isdigit()]
        return {"vram_peak_mib": max(vals), "vram_samples": len(vals)} if vals else {}
    except OSError:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--effort", default="unset", choices=["unset", "high"])
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--deadline-epoch", type=int, default=0)
    ap.add_argument("--hermes", default=os.path.expanduser("~/.local/bin/hermes"))
    ap.add_argument("--ctx", type=int, default=65536)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--gpu-samples", default=None)
    ap.add_argument("--budget-scale", type=float, default=1.0)
    ap.add_argument("--smoke-note", default="")
    a = ap.parse_args()
    spec = json.load(open(os.path.join(HERE, "tasks.json")))
    reg = {c.id: c for c in common.loaded_tasks()}
    os.makedirs(a.out, exist_ok=True)
    home = setup_profile(a)
    want = None if a.tasks == "all" else set(a.tasks.split(","))
    todo = [t for t in spec["tasks"] if want is None or t["id"] in want]
    results, cold = [], not any(f.endswith(".result.json") for f in os.listdir(a.out))
    stopped = False
    for t in todo:
        tid = t["id"]
        rp = f"{a.out}/{tid}.result.json"
        if os.path.isfile(rp):
            results.append(json.load(open(rp)))
            print(f"{tid} SKIP (result present)", flush=True)
            continue
        budget = max(60, int(t["budget_s"] * a.budget_scale))
        if a.deadline_epoch and a.deadline_epoch - time.time() < budget + 120:
            print(f"{tid} DEADLINE: {int(a.deadline_epoch - time.time())} s left, task budget {budget} s: stopping (resumable)", flush=True)
            stopped = True
            break
        if not healthy(a.port):
            print(f"{tid} server on :{a.port} not healthy", flush=True)
            sys.exit(4)
        T = os.path.join(a.work_root, tid)
        common.rm_tree(T)
        os.makedirs(T)
        reg[tid].seed(T)
        prompt = t["prompt"].replace("{T}", T)
        timeout_s = budget + 150
        if a.deadline_epoch:
            timeout_s = min(timeout_s, max(30, int(a.deadline_epoch - time.time())))
        rc, t0, t1 = run_one(a, T, tid, prompt, budget, timeout_s)
        if rc == "timeout" and a.deadline_epoch and time.time() >= a.deadline_epoch - 5:
            json.dump({"id": tid, "interrupted_by_deadline": True, "wall_s": round(t1 - t0, 1)}, open(f"{a.out}/{tid}.interrupted.json", "w"))
            print(f"{tid} cut by the deadline after {t1 - t0:.0f} s (no result, reruns fresh)", flush=True)
            stopped = True
            break
        m = {"session_id": None}
        try:
            db = sqlite3.connect(f"file:{os.path.join(home, 'state.db')}?mode=ro", uri=True)
            sid = find_session(db, prompt, t0, t1)
            if sid:
                m = session_metrics(db, sid)
            db.close()
        except Exception as e:  # noqa: BLE001
            m = {"session_id": None, "metrics_error": f"{type(e).__name__}: {e}"}
        try:
            ok, why, frac = reg[tid].check(T)
        except Exception as e:  # noqa: BLE001
            ok, why, frac = False, f"checker error {type(e).__name__}: {str(e)[:120]}", 0.0
        res = {"id": tid, "name": t["name"], "cat": t["cat"], "diff": t["diff"], "budget_s": budget, "pass": bool(ok), "frac": frac, "check_reason": why,
               "rc": rc, "timed_out": rc == "timeout", "wall_s": round(t1 - t0, 1), "cold_start": cold, "started_epoch": int(t0), **m}
        cold = False
        json.dump(res, open(rp, "w"), indent=1)
        results.append(res)
        print(f"{tid} {'PASS' if ok else 'FAIL'} frac={frac} wall={res['wall_s']}s calls={m.get('tool_calls')} errs={m.get('tool_errors_other')} blocks={m.get('approval_blocks')} "
              f"malformed={m.get('malformed_calls')} rc={rc} :: {why[:140]}", flush=True)
    if stopped:
        sys.exit(3)
    if want is not None:
        print(f"SMOKE-SUBSET {sum(r['pass'] for r in results)}/{len(results)} tasks {a.smoke_note}")
        return
    done = [r for r in results if r.get("id") in {t["id"] for t in spec["tasks"]}]
    if len(done) < len(spec["tasks"]):
        sys.exit(3)
    walls = [r["wall_s"] for r in done]
    by = {}
    for key in ("diff", "cat"):
        for r in done:
            d = by.setdefault(key + ":" + r[key], [0, 0])
            d[0] += r["pass"]
            d[1] += 1
    meta = json.load(open(a.meta)) if a.meta and os.path.isfile(a.meta) else {}
    fa = [r["first_assistant_s"] for r in done if r.get("first_assistant_s") is not None]
    summ = {"row": a.row, "n": len(done), "passed": sum(r["pass"] for r in done), "mean_frac": round(statistics.mean(1.0 if r["pass"] else r["frac"] for r in done), 4),
            "tool_calls": sum(r.get("tool_calls") or 0 for r in done), "tool_errors_other": sum(r.get("tool_errors_other") or 0 for r in done),
            "approval_blocks": sum(r.get("approval_blocks") or 0 for r in done), "malformed_calls": sum(r.get("malformed_calls") or 0 for r in done),
            "timeouts": sum(1 for r in done if r.get("timed_out")), "wall_s_total": round(sum(walls), 1), "wall_s_median": round(statistics.median(walls), 1),
            "first_assistant_s_cold": next((r.get("first_assistant_s") for r in done if r.get("cold_start")), None),
            "first_assistant_s_median": statistics.median(fa) if fa else None,
            "by_group": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by.items())}, "per_task": {r["id"]: [int(r["pass"]), r["frac"]] for r in done},
            "meta": meta, **vram_stats(a.gpu_samples or ""), "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(summ, open(f"{a.out}/summary.json", "w"), indent=1)
    print(f"ROW-SUMMARY {a.row} {summ['passed']}/{summ['n']} blocks={summ['approval_blocks']} other_errs={summ['tool_errors_other']} malformed={summ['malformed_calls']} wall={summ['wall_s_total']}s")


if __name__ == "__main__":
    main()
