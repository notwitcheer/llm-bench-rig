#!/usr/bin/env python3
"""Context-ceiling probe for one GGUF on one card (Nanbeige4.3-33B item 77, leg "fit").

For one KV cache type (f16 or q8_0) walks a ladder of context sizes, high enough to find the ceiling:
for each point it starts ITS OWN llama-server (-ngl 99 -fa on --fit off -c N, never touches another pid),
waits for /health, sends one real prefill (~6k filler tokens, 16 generated), samples this pid's VRAM
(nvidia-smi --query-compute-apps) while it loads and answers, stops the server, waits for the card to drain.
The ladder ends at the first failure (no health, a non-200 reply, or an out-of-memory signature in the
log); then up to --refine midpoints (multiples of 1024) between the last pass and the first fail are tried.
ceiling_ctx = the largest context that loaded AND answered. Resumable: points already in --out are kept.

usage: nb_fit.py --server-bin B --model M --kv f16|q8_0 --out fit.json [--points "8192 16384 ..."]
                 [--port 8199] [--deadline-epoch E] [--refine 2] [--cpu] [--log-dir D]
exit 0 = ladder finished (ceiling found or nothing fits), 3 = stopped by the deadline (resumable), 2 = bad args.
"""
import argparse
import json
import os
import subprocess
import threading
import time
import urllib.request

DEFAULT_POINTS = "8192 12288 16384 20480 24576 28672 32768 40960 49152 57344 65536 81920 98304 114688 131072 196608 262144"
OOM_SIGNS = ("out of memory", "failed to allocate", "cudamalloc failed", "unable to allocate",
             "ggml_backend_cuda_buffer_type_alloc_buffer", "cuda error", "failed to create context")


def smi(*args):
    try:
        return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return ""


def card_used_mib():
    v = smi("--query-gpu=memory.used", "--format=csv,noheader,nounits").strip().splitlines()
    return int(v[0]) if v and v[0].strip().isdigit() else -1


def pid_used_mib(pid):
    for line in smi("--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits").splitlines():
        a = [x.strip() for x in line.split(",")]
        if len(a) == 2 and a[0] == str(pid) and a[1].isdigit():
            return int(a[1])
    return None


class Sampler(threading.Thread):
    def __init__(self, pid, cpu):
        super().__init__(daemon=True)
        self.pid, self.cpu, self.peak, self.stop_ev, self.card_peak = pid, cpu, 0, threading.Event(), 0

    def run(self):
        while not self.stop_ev.is_set():
            if not self.cpu:
                v = pid_used_mib(self.pid)
                if v:
                    self.peak = max(self.peak, v)
                self.card_peak = max(self.card_peak, card_used_mib())
            self.stop_ev.wait(0.5)


def http(url, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read() or b"{}")


def try_point(a, ctx, logf):
    cmd = [a.server_bin, "-m", a.model, "-c", str(ctx), "-fa", "on", "--fit", "off", "--host", "127.0.0.1", "--port", str(a.port)]
    env = dict(os.environ)
    if a.cpu:
        cmd += ["-ngl", "0", "--threads", "6"]
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        cmd += ["-ngl", "99"]
    if a.kv == "q8_0":
        cmd += ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
    rec = {"ctx": ctx, "ok": False, "reason": "", "cmd": " ".join(cmd[1:])}
    t0 = time.time()
    lf = open(logf, "a")
    lf.write(f"\n===== ctx {ctx} kv {a.kv} {time.strftime('%F %T')}\n")
    lf.flush()
    p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
    smp = Sampler(p.pid, a.cpu)
    smp.start()
    try:
        up = False
        while time.time() - t0 < a.load_wait:
            if p.poll() is not None:
                break
            try:
                st, j = http(f"http://127.0.0.1:{a.port}/health", timeout=3)
                if st == 200 and str(j.get("status", "")).lower() == "ok":
                    up = True
                    break
            except Exception:
                pass
            time.sleep(1.5)
        rec["load_s"] = round(time.time() - t0, 1)
        if not up:
            tail = open(logf, errors="replace").read()[-6000:].lower()
            oom = any(s in tail for s in OOM_SIGNS)
            rec["reason"] = "load failed with an out-of-memory signature" if oom else ("server exited before health" if p.poll() is not None else "no health within the wait")
            rec["oom_signature"] = oom
            return rec
        smp.stop_ev.wait(1.0)
        rec["vram_loaded_mib"] = pid_used_mib(p.pid) if not a.cpu else None
        filler = ("The quick brown fox jumps over the lazy dog while the cat watches from the old stone wall. " * 600)
        n_fill = max(256, min(6000, ctx - 512))
        try:
            st, tk = http(f"http://127.0.0.1:{a.port}/tokenize", {"content": filler, "add_special": False}, 60)
            ids = tk["tokens"][:n_fill]
            st, r = http(f"http://127.0.0.1:{a.port}/completion", {"prompt": ids, "n_predict": 16, "temperature": 0, "cache_prompt": False}, 900)
            rec["http"] = st
            rec["prompt_tokens"] = r.get("tokens_evaluated")
            rec["decoded"] = r.get("tokens_predicted")
            rec["ok"] = st == 200 and (r.get("tokens_predicted") or 0) > 0
            rec["reason"] = "ok" if rec["ok"] else "reply without tokens"
        except Exception as ex:
            rec["reason"] = f"request failed: {repr(ex)[:160]}"
            tail = open(logf, errors="replace").read()[-6000:].lower()
            rec["oom_signature"] = any(s in tail for s in OOM_SIGNS)
        return rec
    finally:
        smp.stop_ev.set()
        smp.join(timeout=3)
        rec["vram_peak_mib"] = smp.peak if not a.cpu else None
        rec["card_peak_mib"] = smp.card_peak if not a.cpu else None
        try:
            p.terminate()
            p.wait(timeout=25)
        except Exception:
            p.kill()
            p.wait(timeout=15)
        lf.close()
        if not a.cpu:
            for _ in range(60):
                if 0 <= card_used_mib() < 1500:
                    break
                time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--kv", required=True, choices=["f16", "q8_0"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--points", default=DEFAULT_POINTS)
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--deadline-epoch", type=int, default=0)
    ap.add_argument("--refine", type=int, default=2)
    ap.add_argument("--load-wait", type=int, default=420)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--log-dir", default="")
    a = ap.parse_args()
    pts = [int(x) for x in a.points.split()]
    out = a.out
    logf = os.path.join(a.log_dir or os.path.dirname(out) or ".", f"fit-{a.kv}.server.log")
    res = json.load(open(out)) if os.path.exists(out) else {}
    res.update({"label": a.label, "model": a.model, "kv": a.kv, "server_bin": a.server_bin, "ladder": pts,
                "method": "own llama-server per point, -ngl 99 -fa on --fit off -c N, ~6k-token prefill + 16 decoded, per-pid VRAM peak"})
    res.setdefault("points", {})

    def save():
        ok = [int(k) for k, v in res["points"].items() if v.get("ok")]
        bad = [int(k) for k, v in res["points"].items() if not v.get("ok")]
        res["ceiling_ctx"] = max(ok) if ok else None
        res["first_fail_ctx"] = min([b for b in bad if not ok or b > max(ok)] or [0]) or None
        json.dump(res, open(out, "w"), indent=1)

    def left():
        return (a.deadline_epoch - time.time()) if a.deadline_epoch else 1e9

    def run_point(c):
        if str(c) in res["points"]:
            return res["points"][str(c)]["ok"]
        if left() < 300:   # a point needs minutes, not its worst-case load wait; the caller's timeout is the hard cap
            return None
        r = try_point(a, c, logf)
        res["points"][str(c)] = r
        save()
        print(f"[fit {a.kv}] ctx {c}: {'OK' if r['ok'] else 'FAIL'} {r['reason']} load {r.get('load_s')}s vram loaded {r.get('vram_loaded_mib')} peak {r.get('vram_peak_mib')} MiB prompt {r.get('prompt_tokens')}", flush=True)
        return r["ok"]

    last_ok, first_fail, stopped = None, None, False
    for c in pts:
        st = res["points"].get(str(c))
        if st is not None and not st["ok"]:
            first_fail = c
            break
        r = run_point(c)
        if r is None:
            stopped = True
            break
        if r:
            last_ok = c
        else:
            first_fail = c
            break
    if not stopped and first_fail and last_ok and a.refine > 0:
        lo, hi = last_ok, first_fail
        for _ in range(a.refine):
            mid = (lo + hi) // 2 // 1024 * 1024
            if mid <= lo or mid >= hi:
                break
            r = run_point(mid)
            if r is None:
                stopped = True
                break
            lo, hi = (mid, hi) if r else (lo, mid)
    if not stopped and last_ok is None and first_fail is None:
        stopped = True
    save()
    res["complete"] = not stopped
    json.dump(res, open(out, "w"), indent=1)
    print(f"[fit {a.kv}] ceiling_ctx {res.get('ceiling_ctx')} first_fail {res.get('first_fail_ctx')} complete {res['complete']}", flush=True)
    raise SystemExit(3 if stopped else 0)


if __name__ == "__main__":
    main()
