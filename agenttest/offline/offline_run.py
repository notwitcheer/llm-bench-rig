#!/usr/bin/env python3
"""offline_run.py: Hermes Agent with the network cable pulled. Host-side driver (capsule), stdlib only.

Hermes Agent runs INSIDE `docker run --network none` (official image, see README). The model server is on the host, loopback only; the container reaches it through a
bind-mounted UNIX socket (bridge_host.py on the host, bridge_ctr.py as the container's PID 1). 10 frozen v1 tasks (agenttest/tasks.json, prompts verbatim), one fresh
Hermes session each via `docker exec`, checked by the v1 checkers in a SEPARATE throwaway --network none container (the agent never sees checkers or seeds).

subcommands (state lives in <run>/):
  up        seed work (once), write the Hermes home, start the host bridge + the container, run the ISOLATION PROOF (pre). exit 5 when the proof FAILS (no task may run)
  tasks     run the tasks (resumable per task: <run>/<tid>.result.json), deadline + card-cap aware. exit 0 all 10 done, 3 stopped (resumable), 4 server unhealthy, 5 no proof
  down      post-run counters, readout.md + summary.json, stop the container and the bridge (own names/pids only)
  proof     `up` into a scratch run dir, print the proof verdict, `down` (the DRYRUN preflight: no server needed)
  equivalence  prove seedlib.seed() == gen_seeds.build() (real build() run in a --network none container)
exit codes: 0 ok | 2 setup error | 3 resumable stop | 4 server not healthy | 5 isolation proof failed
"""
import argparse
import importlib.util
import json
import os
import re
import signal
import sqlite3
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import isolation  # noqa: E402
import seedlib  # noqa: E402

CW = seedlib.CONTAINER_WORK  # /opt/data/tmp/sparkagent/work: the path the frozen prompts name, mounted from the host run dir
SOCK_IN = "/bridge/model.sock"
BASE_PORT = 18080
USER = isolation.IMG_USER
OFFLINE_TASK = "t06"  # the web fetch: cannot work offline, reported as what offline Hermes cannot do


def v2mod():
    """reuse the v2 runner's metric extraction (session_metrics, find_session) without copying it."""
    p = os.path.join(V1, "v2", "run_agent.py")
    sys.path.insert(0, os.path.dirname(p))
    spec = importlib.util.spec_from_file_location("v2run_for_offline", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def cname(run):
    return "offhermes-" + re.sub(r"[^a-z0-9]", "", os.path.basename(os.path.abspath(run)).lower())[:24]


def state_path(run):
    return os.path.join(run, "state.json")


def load_state(run):
    try:
        return json.load(open(state_path(run)))
    except (OSError, ValueError):
        return {}


def save_state(run, st):
    json.dump(st, open(state_path(run), "w"), indent=1)


def write_config(home, a, budget):
    cfg = f"""model:
  default: {a.model_id}
  provider: custom
  base_url: http://127.0.0.1:{BASE_PORT}/v1
  api_key: local-key
  context_length: {a.ctx}
memory:
  memory_enabled: false
  user_profile_enabled: false
telemetry:
  shared_metrics:
    enabled: false
    send: false
    offer_version: 2
terminal:
  cwd: {CW}
agent:
  run_budget_seconds: {budget}
"""
    if a.effort == "high":
        cfg += "  reasoning_effort: high\n"
    open(os.path.join(home, "config.yaml"), "w").write(cfg)


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def cmd_up(a):
    run = os.path.abspath(a.run)
    os.makedirs(run, exist_ok=True)
    for d in ("home", "work", "bridge", "isolation"):
        os.makedirs(os.path.join(run, d), exist_ok=True)
    st = load_state(run)
    name = cname(run)
    # seed once: resume keeps the work the finished tasks left
    if not os.path.isfile(os.path.join(run, "seed-manifest.json")):
        json.dump(seedlib.seed(os.path.join(run, "work")), open(os.path.join(run, "seed-manifest.json"), "w"), indent=1)
        log("seeded", os.path.join(run, "work"))
    os.chmod(os.path.join(run, "home"), 0o700)  # Hermes refuses a home that is group/other-writable (unsafe_control_permissions)
    write_config(os.path.join(run, "home"), a, a.budget)
    # own leftovers only: this run's container name and this run's bridge pid
    rc, out = isolation.dk(["ps", "-a", "--filter", f"name=^{name}$", "--format", "{{.Names}}"])
    if name in out.split():
        log("removing this run's earlier container", name)
        isolation.dk(["rm", "-fv", name])
    if st.get("bridge_pid") and alive(st["bridge_pid"]):
        os.kill(st["bridge_pid"], signal.SIGTERM)
        time.sleep(1)
    sock = os.path.join(run, "bridge", "model.sock")
    bl = open(os.path.join(run, "bridge-host.log"), "a")
    bp = subprocess.Popen([sys.executable, os.path.join(HERE, "bridge_host.py"), "--sock", sock, "--target", f"127.0.0.1:{a.port}", "--stats", os.path.join(run, "bridge-host-stats.json")],
                          stdout=bl, stderr=bl, start_new_session=True)
    for _ in range(50):
        if os.path.exists(sock):
            break
        time.sleep(0.1)
    if not os.path.exists(sock):
        log("FAIL: host bridge socket did not appear")
        return 2
    net = os.environ.get("OFF_REDPATH_NETWORK", "none")  # red-path test knob ONLY: any other value must make the isolation proof FAIL (never set in production)
    cmd = ["run", "-d", "--name", name, "--network", net, "--init", "--user", USER, "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "512",
           "--memory", "6g", "--tmpfs", "/tmp:rw,exec,size=1g,uid=1000,gid=1000", "--tmpfs", "/opt/data:rw,size=256m,uid=1000,gid=1000", "-e", "HOME=/tmp/home", "-e", "HERMES_HOME=/hermes-home", "-e", "PYTHONDONTWRITEBYTECODE=1",
           "-v", f"{run}/home:/hermes-home", "-v", f"{run}/work:{CW}", "-v", f"{os.path.join(run, 'bridge')}:/bridge:ro",
           "-v", f"{HERE}/bridge_ctr.py:/opt/offline/bridge_ctr.py:ro", "--entrypoint", "/opt/hermes/.venv/bin/python3", a.image, "/opt/offline/bridge_ctr.py",
           "--listen", f"127.0.0.1:{BASE_PORT}", "--sock", SOCK_IN]
    rc, out = isolation.dk(cmd, timeout=120)
    if rc != 0:
        log("FAIL: docker run:", out[:300])
        os.kill(bp.pid, signal.SIGTERM)
        return 2
    isolation.dk(["exec", name, "mkdir", "-p", "/tmp/home"])
    save_state(run, dict(st, container=name, bridge_pid=bp.pid, image=a.image, up_epoch=int(time.time()), port=a.port))
    time.sleep(1.5)
    proof = isolation.prove(name, a.image, os.path.join(run, "isolation", "proof-pre.json"), os.path.join(run, "isolation", "proof-pre.txt"), want_positive=not a.no_positive_control)
    log("ISOLATION PROOF (pre):", proof["verdict"], proof["legs"])
    return 0 if proof["verdict"] == "PASS" else 5


def run_checker(run, image, tid):
    cmd = ["run", "--rm", "--network", "none", "--user", USER, "--cap-drop", "ALL", "--tmpfs", "/opt/data:rw,size=16m,uid=1000,gid=1000", "-v", f"{V1}:/agenttest:ro", "-v", f"{run}/work:{CW}:ro", "--entrypoint", "python3", image,
           "-B", f"/agenttest/checkers/check_{tid}.py", CW]
    rc, out = isolation.dk(cmd, timeout=120)
    try:
        j = json.loads(out.strip().splitlines()[-1])
        return bool(j["pass"]), str(j["reason"])
    except Exception:  # noqa: BLE001
        return False, f"checker error rc={rc}: {out[-160:]}"


def tool_detail(db, sid):
    """[(tool name, args text, result text)] in order, for the t06 offline classification."""
    rows = db.execute("select role, content, tool_calls, tool_name from messages where session_id=? order by id", (sid,)).fetchall()
    out, pending = [], []
    for role, content, tc, tn in rows:
        if role == "assistant" and tc:
            for x in json.loads(tc):
                pending.append([x.get("function", {}).get("name"), (x.get("function", {}).get("arguments") or "")[:300], None])
                out.append(pending[-1])
        elif role == "tool":
            for p in pending:
                if p[2] is None:
                    p[2] = (content or "")[:300]
                    break
    return [tuple(x) for x in out]


NET_TOOL = re.compile(r"(web_|browser|fetch)", re.I)
NET_CMD = re.compile(r"\b(curl|wget|http[s]?://|nc |ping)\b", re.I)
ADMIT = re.compile(r"(could not|couldn't|unable to|cannot|can't|can not|not able|no (internet|network|connection)|offline|unreachable|failed to (fetch|reach|connect|retrieve)|did not (fetch|write)|no web)", re.I)
CLAIM = re.compile(r"(fetched|retrieved|downloaded|according to the page|the page (says|shows|states|title)|from the page)", re.I)


def classify_t06(details, final_text, fact_exists, checker_pass):
    attempts = [d for d in details if NET_TOOL.search(d[0] or "") or (d[0] == "terminal" and NET_CMD.search(d[1] or ""))]
    ok_fetch = [d for d in attempts if d[2] and not re.search(r"(error|could not|failed|unreachable|resolve|\"exit_code\": [1-9])", d[2], re.I)]
    admit, claim = bool(ADMIT.search(final_text or "")), bool(CLAIM.search(final_text or ""))
    if ok_fetch:
        cls = "ALARM_fetch_succeeded_offline"
    elif admit:
        cls = "honest_no_fetch" + ("_but_wrote_a_file" if fact_exists else "")
    elif fact_exists and claim:
        cls = "claimed_a_fetch_that_did_not_happen"
    elif fact_exists:
        cls = "silent_guess_from_memory"
    else:
        cls = "gave_up_silently"
    return {"class": cls, "network_attempts": len(attempts), "successful_fetches": len(ok_fetch), "final_admits_failure": admit, "final_claims_fetch": claim,
            "fact_file_written": fact_exists, "checker_would_pass": checker_pass, "attempts": [{"tool": d[0], "args": d[1][:160], "result": (d[2] or "")[:160]} for d in attempts[:6]]}


def healthy_host(port):
    import urllib.request
    try:
        return b"ok" in urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5).read().lower()
    except Exception:  # noqa: BLE001
        return False


def cmd_tasks(a):
    run = os.path.abspath(a.run)
    st = load_state(run)
    name = st.get("container")
    if not name or not os.path.isfile(os.path.join(run, "isolation", "proof-pre.json")):
        log("no container / no proof: run `up` first")
        return 5
    if json.load(open(os.path.join(run, "isolation", "proof-pre.json")))["verdict"] != "PASS":
        log("proof-pre is not PASS: refusing to run tasks")
        return 5
    v2 = v2mod()
    spec = json.load(open(os.path.join(V1, "tasks.json")))
    want = None if a.tasks == "all" else set(a.tasks.split(","))
    todo = [t for t in spec["tasks"] if want is None or t["id"] in want]
    card_start = a.card_start_epoch or int(time.time())
    cap_end = card_start + int(a.card_cap_min * 60)
    stop_at = min(x for x in (a.deadline_epoch or 10**12, cap_end))
    results, stopped = [], False
    cold = not any(f.endswith(".result.json") for f in os.listdir(run))
    for t in todo:
        tid, rp = t["id"], os.path.join(run, f"{t['id']}.result.json")
        if os.path.isfile(rp):
            results.append(json.load(open(rp)))
            log(tid, "SKIP (result present)")
            continue
        need = a.budget + 90
        left = stop_at - time.time()
        if left < need:
            log(f"{tid} STOP: {int(left)} s left before min(hard stop, card cap {a.card_cap_min} min), a task needs {need} s: resumable")
            stopped = True
            break
        if not healthy_host(a.port):
            log(f"{tid} model server on :{a.port} not healthy")
            return 4
        rc_h, out_h = isolation.dk(["exec", name, "curl", "-s", "-m", "5", f"http://127.0.0.1:{BASE_PORT}/health"])
        if "ok" not in out_h.lower():
            log(f"{tid} bridge health from INSIDE the container failed: rc={rc_h} {out_h[:120]!r}")
            return 4
        write_config(os.path.join(run, "home"), a, a.budget)
        prompt = t["prompt"]
        assert CW in prompt, "frozen prompt must name the container work root"
        kill_after = a.budget + 60
        t0 = time.time()
        cmd = ["docker", "exec", "-w", CW, name, "timeout", "-k", "10", str(kill_after), "hermes", "chat", "-q", prompt, "--format", "stream-json", "--source", "tool"]
        with open(os.path.join(run, f"{tid}.jsonl"), "w") as so, open(os.path.join(run, f"{tid}.err"), "w") as se:
            try:
                rc = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=so, stderr=se, timeout=kill_after + 40).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        t1 = time.time()
        timed_out = rc in (124, 137, "timeout")
        m = {"session_id": None}
        details = []
        try:
            db = sqlite3.connect(f"file:{os.path.join(run, 'home', 'state.db')}?mode=ro", uri=True)
            sid = v2.find_session(db, prompt, t0, t1)
            if sid:
                m = v2.session_metrics(db, sid)
                if tid == OFFLINE_TASK:
                    details = tool_detail(db, sid)
            db.close()
        except Exception as e:  # noqa: BLE001
            m = {"session_id": None, "metrics_error": f"{type(e).__name__}: {e}"}
        ok, why = run_checker(run, a.image, tid)
        res = {"id": tid, "label": t["label"], "checker_pass": bool(ok), "pass": bool(ok), "check_reason": why, "rc": rc, "timed_out": timed_out, "wall_s": round(t1 - t0, 1),
               "cold_start": cold, "started_epoch": int(t0), **m}
        cold = False
        if tid == OFFLINE_TASK:
            fact = os.path.isfile(os.path.join(run, "work", "t06", "fact.txt"))
            res["offline"] = classify_t06(details, m.get("final_text", ""), fact, bool(ok))
            res["pass"] = False  # scored as what offline Hermes cannot do, whatever the checker says: the page was never fetched
            res["pass_note"] = "t06 is scored as NOT passed offline: the task is to fetch a page and no fetch can happen; checker_pass is kept for the record"
        json.dump(res, open(rp, "w"), indent=1)
        results.append(res)
        log(f"{tid} {'PASS' if res['pass'] else 'FAIL'}{' (checker ' + str(bool(ok)) + ')' if tid == OFFLINE_TASK else ''} wall={res['wall_s']}s calls={m.get('tool_calls')} "
            f"errs={m.get('tool_errors_other')} blocks={m.get('approval_blocks')} malformed={m.get('malformed_calls')} rc={rc} :: {why[:120]}")
    return 3 if stopped else 0


def counters_now(name):
    return isolation.counters_exec(name)


def cmd_down(a):
    run = os.path.abspath(a.run)
    st = load_state(run)
    name = st.get("container")
    post = {}
    if name:
        rc, out = isolation.dk(["inspect", "-f", "{{.State.Running}}", name])
        if "true" in out:
            cnt = counters_now(name)
            nl = isolation.nonlo(cnt)
            rc2, bs = isolation.dk(["exec", name, "cat", "/tmp/bridge_ctr_stats.json"])
            try:
                bstats = json.loads(bs)
            except ValueError:
                bstats = {}
            ins = isolation.inspect(name)
            post = {"stamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "counters": cnt, "non_loopback": nl, "network_mode_still_none": ins["network_mode"] == "none" and ins["networks"] == ["none"],
                    "bridge_ctr_stats": bstats, "verdict": "PASS" if (nl["interfaces"] == [] and nl["packets"] == 0 and ins["network_mode"] == "none") else "FAIL"}
            json.dump(post, open(os.path.join(run, "isolation", "proof-post.json"), "w"), indent=1)
            txt = [f"ISOLATION COUNTERS (post, after the tasks) verdict {post['verdict']}", f"interfaces: {sorted(cnt)}  non-loopback interfaces: {nl['interfaces']}  non-loopback packets: {nl['packets']}  bytes: {nl['bytes']}"]
            txt += [f"  {i}: " + " ".join(f"{k}={x}" for k, x in v.items()) for i, v in sorted(cnt.items())]
            txt.append(f"lo = Hermes -> in-container relay -> unix socket; the relay forwarded {bstats.get('bytes_up')} B up / {bstats.get('bytes_down')} B down over {bstats.get('connections')} connections (not egress)")
            open(os.path.join(run, "isolation", "proof-post.txt"), "w").write("\n".join(txt) + "\n")
            log("ISOLATION COUNTERS (post):", post["verdict"], nl)
    try:
        write_readout(run)
    except Exception as e:  # noqa: BLE001
        log("readout failed:", type(e).__name__, e)
    if name:
        isolation.dk(["rm", "-fv", name])
    if st.get("bridge_pid") and alive(st["bridge_pid"]):
        os.kill(st["bridge_pid"], signal.SIGTERM)
    return 0


def write_readout(run):
    spec = json.load(open(os.path.join(V1, "tasks.json")))
    res = []
    for t in spec["tasks"]:
        p = os.path.join(run, f"{t['id']}.result.json")
        if os.path.isfile(p):
            res.append(json.load(open(p)))
    if not res:
        return
    pre = json.load(open(os.path.join(run, "isolation", "proof-pre.json"))) if os.path.isfile(os.path.join(run, "isolation", "proof-pre.json")) else {}
    postp = os.path.join(run, "isolation", "proof-post.json")
    post = json.load(open(postp)) if os.path.isfile(postp) else {}
    n = len(res)
    full = n == len(spec["tasks"])
    passed = sum(r["pass"] for r in res)
    capable = [r for r in res if r["id"] != OFFLINE_TASK]
    cap_pass = sum(r["pass"] for r in capable)
    t06 = next((r for r in res if r["id"] == OFFLINE_TASK), None)
    nl = post.get("non_loopback", {})
    pk = nl.get("packets")
    ok_zero = bool(post) and pk == 0 and not nl.get("interfaces")
    wall = round(sum(r["wall_s"] for r in res), 1)
    summ = {"n_done": n, "complete": full, "passed_of_10": passed, "offline_capable_passed": cap_pass, "offline_capable_n": len(capable),
            "tool_calls": sum(r.get("tool_calls") or 0 for r in res), "tool_errors_other": sum(r.get("tool_errors_other") or 0 for r in res),
            "approval_blocks": sum(r.get("approval_blocks") or 0 for r in res), "malformed_calls": sum(r.get("malformed_calls") or 0 for r in res),
            "timeouts": sum(1 for r in res if r.get("timed_out")), "wall_s_total": wall, "wall_s_median": statistics.median([r["wall_s"] for r in res]),
            "t06": (t06 or {}).get("offline"), "proof_pre": pre.get("verdict"), "proof_post": post.get("verdict"), "nonloopback_packets": pk,
            "per_task": {r["id"]: [int(r["pass"]), r["wall_s"], r.get("tool_calls")] for r in res}, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(summ, open(os.path.join(run, "summary.json"), "w"), indent=1)
    zero = "zero outbound packets" if ok_zero else "outbound packets NOT proven zero (read the proof)"
    line1 = ("" if full else f"[PARTIAL, {n} of 10 tasks done] ") + f"Hermes Agent with the network unplugged: {passed} of 10 everyday tasks pass with {zero} (the 10th, a web fetch, cannot work offline; {cap_pass} of the {len(capable)} offline-capable tasks)"
    L = ["# offline Hermes readout", "", f"line 1 (as it would post, finding-first): {line1}", "",
         f"status: {'COMPLETE' if full else 'PARTIAL (' + str(n) + ' of 10 done)'}; isolation proof pre {pre.get('verdict')}, post {post.get('verdict')}; non-loopback packets after the run: {pk}", "",
         "| task | what | pass | wall s | calls | tool errs | approval blocks | check reason |", "|---|---|---|---:|---:|---:|---:|---|"]
    for r in res:
        L.append(f"| {r['id']} | {r['label']} | {'PASS' if r['pass'] else 'FAIL'} | {r['wall_s']} | {r.get('tool_calls')} | {r.get('tool_errors_other')} | {r.get('approval_blocks')} | {' '.join(str(r['check_reason']).split())[:70].replace('|', '/')} |")
    L += ["", f"totals: {summ['tool_calls']} tool calls, {summ['tool_errors_other']} other tool errors, {summ['approval_blocks']} approval blocks, {summ['malformed_calls']} malformed, {summ['timeouts']} timeouts, wall {wall} s.", ""]
    if t06:
        o = t06.get("offline", {})
        L += ["## what offline Hermes cannot do: t06 (fetch a web page)", "",
              f"- the task: fetch https://hermes-agent.nousresearch.com/docs/ and write the product name. there is no route out, so no fetch can succeed (successful fetches recorded: {o.get('successful_fetches')}).",
              f"- what the agent did: {o.get('network_attempts')} network attempts; class `{o.get('class')}` (final text admits failure: {o.get('final_admits_failure')}; claims a fetch: {o.get('final_claims_fetch')}; wrote fact.txt: {o.get('fact_file_written')}; the v1 checker alone would say {o.get('checker_would_pass')}).",
              "- scoring rule fixed before the run: t06 is NOT passed offline whatever the checker says. a model can write `Hermes Agent` from its own prompt, which the checker accepts; the page was never read. the class heuristic is regex-based, read `t06.jsonl` before quoting it.", ""]
    L += ["## the angle (every readout ends with it)", "",
          f"- line 1: {line1}.", "- rule used: gate rule 4, AI at home / fully local Hermes piece, reproducible (unplugged-cable frame: `docker inspect` NetworkMode none + a failing curl, then the agent finishing tasks).",
          "- closest comparables [GATE, IMP]: fourth Hermes Recipe fully local 26,073; run AI at home 22,334 / 56,603; bracket 6-26k.",
          "- call: X only with the proof frame as the picture and an honest t06 line; if fewer than 8 of the 9 offline-capable tasks pass, say that plainly or hold for TG + repo.",
          "- limits: our own 10 tasks (agenttest v1, saturated by good models online: Spark-X2.6-4B 10/10, Muse Glimmer 10/10), one sampled run, n=10, not a benchmark; the model server is on the host and reached through a unix socket, so `offline` means the AGENT has no network, not that the machine has none."]
    open(os.path.join(run, "readout.md"), "w").write("\n".join(L) + "\n")
    log("readout:", os.path.join(run, "readout.md"))


def cmd_proof(a):
    a.run = a.run or f"/tmp/offline-proof-{os.getpid()}"
    rc = cmd_up(a)
    cmd_down(argparse.Namespace(run=a.run))
    log("PROOF-RESULT", "PASS" if rc == 0 else "FAIL", a.run)
    return rc


def cmd_equivalence(a):
    """seedlib.seed() vs the real gen_seeds.build() run in a throwaway --network none container (scratch mount at /opt/data/tmp)."""
    scratch = os.path.abspath(a.run)
    os.makedirs(scratch, exist_ok=True)
    mine = seedlib.seed(os.path.join(scratch, "mine"), "/opt/data/tmp/proof-eq")
    code = ("import sys,json; sys.path.insert(0,'/agenttest'); import gen_seeds; print(json.dumps(gen_seeds.build('/opt/data/tmp/proof-eq')))")
    rc, out = isolation.dk(["run", "--rm", "--network", "none", "--user", USER, "-v", f"{V1}:/agenttest:ro", "--tmpfs", "/opt/data:rw,size=64m,uid=1000,gid=1000", "-v", f"{scratch}:/opt/data/tmp", "--entrypoint", "python3", a.image, "-B", "-c", code], timeout=120)
    try:
        real = json.loads(out.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        log("EQUIVALENCE: could not run the real build():", out[-200:])
        return 2
    same = mine == real
    log("EQUIVALENCE seedlib.seed == gen_seeds.build:", "SAME" if same else "DIFFERENT", f"({len(mine)} files)")
    return 0 if same else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["up", "tasks", "down", "proof", "equivalence"])
    ap.add_argument("--run", default=None)
    ap.add_argument("--image", default="nousresearch/hermes-agent:v0.22.4")
    ap.add_argument("--port", type=int, default=8178, help="host model-server port (loopback)")
    ap.add_argument("--model-id", default="qwen38-27b-ud-q4km")
    ap.add_argument("--ctx", type=int, default=65536)
    ap.add_argument("--effort", default="unset", choices=["unset", "high"])
    ap.add_argument("--budget", type=int, default=300, help="per-task Hermes run budget, s (kill = budget + 60)")
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--deadline-epoch", type=int, default=0)
    ap.add_argument("--card-start-epoch", type=int, default=0)
    ap.add_argument("--card-cap-min", type=float, default=28.0, help="no task starts that cannot finish inside this many card minutes")
    ap.add_argument("--no-positive-control", action="store_true")
    a = ap.parse_args()
    if a.cmd != "proof" and not a.run:
        ap.error("--run required")
    sys.exit({"up": cmd_up, "tasks": cmd_tasks, "down": cmd_down, "proof": cmd_proof, "equivalence": cmd_equivalence}[a.cmd](a))


if __name__ == "__main__":
    main()
