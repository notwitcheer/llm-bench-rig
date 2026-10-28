#!/usr/bin/env python3
"""fit_probe.py: VRAM of ONE server at depth. Sends one ~14k-token prompt (what Hermes' first call carries) and samples nvidia-smi memory.used while it runs.
stdlib only. usage: fit_probe.py --port P --out FILE --label L [--target-tokens 14000] [--no-gpu]"""
import argparse
import json
import subprocess
import threading
import time
import urllib.error
import urllib.request


def post(url, body, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()[:300]}


def vram():
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.split()[0])
    except Exception:  # noqa: BLE001
        return None


ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--target-tokens", type=int, default=14000)
ap.add_argument("--no-gpu", action="store_true")
a = ap.parse_args()
base = f"http://127.0.0.1:{a.port}"
para = "The quick brown fox jumps over the lazy dog while the build server compiles the release notes for version %d. "
text, n = "", 0
while True:
    text += para % n
    n += 1
    if n % 50 == 0:
        st, tk = post(base + "/tokenize", {"content": text})
        if st == 200 and len(tk.get("tokens", [])) >= a.target_tokens:
            break
    if n > 3000:
        break
samples, stop = [], False


def sampler():
    while not stop:
        v = None if a.no_gpu else vram()
        if v is not None:
            samples.append(v)
        time.sleep(0.5)


before = None if a.no_gpu else vram()
th = threading.Thread(target=sampler, daemon=True)
th.start()
t0 = time.time()
st, resp = post(base + "/v1/chat/completions", {"model": "x", "messages": [{"role": "user", "content": text + "\nReply with the single word ok."}], "max_tokens": 16, "temperature": 0})
dt = time.time() - t0
stop = True
th.join(timeout=3)
after = None if a.no_gpu else vram()
usage = resp.get("usage", {}) if isinstance(resp, dict) else {}
out = {"label": a.label, "http": st, "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"), "request_s": round(dt, 2),
       "prefill_tok_s_approx": round(usage["prompt_tokens"] / dt, 1) if usage.get("prompt_tokens") and dt else None,
       "vram_before_mib": before, "vram_peak_mib": max(samples) if samples else None, "vram_after_mib": after, "samples": len(samples), "error": resp.get("error") if st != 200 else None}
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps(out))
