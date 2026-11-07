#!/usr/bin/env python3
"""wd_gen.py: serve ONE 0.8B GGUF with llama-server (own port), run a problem set (dev|held) against it, save the rows, stop the server.
usage: wd_gen.py --gguf G --set dev|held --out rows.json [--port 8281] [--slots 8] [--cap 8192] [--limit N] [--ngl 99] [--seed 1] [--tag T]
Stops ONLY the server pid it started. Never touches :8090. exit 0 ok, 1 load/run failure."""
import argparse, json, os, signal, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wd_common as C  # noqa: E402

SRV = os.environ.get("WD_SRV", os.path.expanduser("~/llama.cpp-donald/build/bin/llama-server"))
ap = argparse.ArgumentParser()
ap.add_argument("--gguf", required=True)
ap.add_argument("--set", required=True, choices=["dev", "held"])
ap.add_argument("--out", required=True)
ap.add_argument("--port", type=int, default=8281)
ap.add_argument("--slots", type=int, default=8)
ap.add_argument("--cap", type=int, default=C.CAP)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--ngl", type=int, default=99)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--tag", default="")
a = ap.parse_args()

probs = C.problems(a.set)
if a.limit:
    probs = probs[:a.limit]
ctx = a.slots * (a.cap + 1200)
log = open(a.out + ".server.log", "w")
cmd = [SRV, "-m", a.gguf, "--port", str(a.port), "--host", "127.0.0.1", "-ngl", str(a.ngl), "-c", str(ctx), "-np", str(a.slots), "--jinja", "--threads", "4", "--no-webui"]
t0 = time.time()
p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
ok = False
for _ in range(240):
    if p.poll() is not None:
        break
    try:
        if b"ok" in urllib.request.urlopen("http://127.0.0.1:%d/health" % a.port, timeout=2).read().lower():
            ok = True
            break
    except Exception:
        pass
    time.sleep(1)
if not ok:
    print("WD_GEN FAIL: server did not come up (see %s)" % (a.out + ".server.log"))
    try:
        p.kill()
    except Exception:
        pass
    sys.exit(1)
load_s = time.time() - t0
t1 = time.time()
try:
    rows = C.run_problems(a.port, probs, a.slots, cap=a.cap, seed=a.seed)
finally:
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=30)
    except Exception:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            pass
s = C.summarise(rows)
s.update(set=a.set, tag=a.tag, gguf=os.path.basename(a.gguf), cap=a.cap, load_s=round(load_s, 1), gen_s=round(time.time() - t1, 1), seed=a.seed)
json.dump(dict(summary=s, rows=rows), open(a.out, "w"))
print("WD_GEN OK " + json.dumps(s))
