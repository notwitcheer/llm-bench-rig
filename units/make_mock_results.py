#!/usr/bin/env python3
"""build a MOCK results dir (scripted answers via mock_units.py, never a model) under the REAL model keys, so the renderer and the readout can be proven on CPU.
usage: make_mock_results.py OUTDIR [PORT]"""
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
out = sys.argv[1]
port = int(sys.argv[2]) if len(sys.argv) > 2 else 8652
mods = json.load(open(os.path.join(HERE, "models.json")))
prof = ["mock-slippy", "mock-clean", "mock-three", "mock-perfect", "mock-shapes"]
for m, p in zip(mods["legs"], prof):
    m["alias"] = p
os.makedirs(out, exist_ok=True)
mp = os.path.join(out, "mock-models.json")
json.dump(mods, open(mp, "w"))
srv = subprocess.Popen([sys.executable, os.path.join(HERE, "mock_units.py"), str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(50):
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=1).read()
            break
        except Exception:
            time.sleep(0.1)
    res = os.path.join(out, "results")
    for m in mods["legs"]:
        r = subprocess.run([sys.executable, os.path.join(HERE, "run_units.py"), "--model", m["key"], "--port", str(port), "--out", res, "--models", mp], capture_output=True, text=True)
        print(m["key"], "<-", m["alias"], "rc", r.returncode)
finally:
    srv.terminate()
print("MOCK_RESULTS", res, mp)
