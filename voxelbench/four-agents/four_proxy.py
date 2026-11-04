#!/usr/bin/env python3
"""four_proxy.py: per-agent counting proxy + HUD sampler for item 80 (four Hermes agents, one llama-server, --parallel 4).

usage: four_proxy.py <upstream http://127.0.0.1:PORT> <first-listen-port> <outdir> [--agents abcd] [--hz 1.0]
                     [--pool TOKENS [--reserve-gen 12288] [--reserve-first 26624] [--reserve-summary 4096] [--guard-wait-max 90]]

* listens on first-port .. first-port+3, one port per agent (a b c d). Each agent's Hermes profile points its base_url at its own port, so every
  request is attributed to an agent without touching the request body (llama-server has no notion of a client). The proxy adds `X-Race-Agent`
  and forwards the stream byte for byte; a hold-up here would be a hold-up in the race, so it does no parsing before forwarding.
* counts streamed deltas per agent (content, reasoning_content, tool_calls pieces: one SSE delta is about one token on llama.cpp; the exact
  per-request numbers come from the `timings` object of the last chunk and are logged next to it). llama-server's own
  `llamacpp:tokens_predicted_total` (/metrics) is the aggregate truth; the sampler logs both so the HUD can say which it shows.
* sampler thread, every 1/hz s: upstream /metrics + /slots (+ nvidia-smi memory every 5 s) + the per-agent counters -> <outdir>/hud.jsonl
  (cumulative counters; the renderer differentiates). <outdir>/requests.jsonl = one line per finished request.
* POOL GUARD (80b, off unless --pool is given; item 80 ran without it): the proxy keeps a running estimate of how many KV tokens each agent holds in the shared pool
  (idle agent = its last finished request's cache_n + prompt_n + predicted_n from the server's own timings; running agent = estimated prompt + max(reserve, tokens streamed so far),
  reserve = --reserve-first for an agent's first request, --reserve-gen after, --reserve-summary for a request with no tools array = Hermes' summary call)
  and HOLDS a new request (up to --guard-wait-max s, polling every 0.25 s) while
  the others' use + this request's estimate + its reserve would exceed --pool. It never holds when no other request is running (no deadlock) and never longer than the cap.
  Every request line in requests.jsonl carries guard_wait_s (0 = not held) and ctx_end; /_state carries guard_holds + guard_timeouts. Prompt size is estimated from the body
  bytes with a per-agent bytes-per-token ratio learned from that agent's previous request (3.0 before the first), which over-counts a little on purpose.
* GET /_state on any listen port returns the newest sampler row (the recorder and the item read it to know how many requests are in flight).
Stdlib only, so it runs under any capsule python.
"""
import http.client
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

up_url, first_port, outdir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
agents = "abcd"
hz = 1.0
if "--agents" in sys.argv:
    agents = sys.argv[sys.argv.index("--agents") + 1]
if "--hz" in sys.argv:
    hz = float(sys.argv[sys.argv.index("--hz") + 1])


def _opt(name, default, cast=int):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


POOL = _opt("--pool", 0)
RESERVE_GEN = _opt("--reserve-gen", 12288)
RESERVE_FIRST = _opt("--reserve-first", 26624)
RESERVE_SUMMARY = _opt("--reserve-summary", 4096)       # a request without a tools array is Hermes' own summary / title call: short output
WAIT_MAX = _opt("--guard-wait-max", 90.0, float)
UP = urlparse(up_url)
os.makedirs(outdir, exist_ok=True)
LOCK = threading.Lock()
C = {a: {"chunks": 0, "inflight": 0, "reqs": 0, "done": 0, "err": 0, "exact_pred": 0, "exact_prompt": 0, "last_start": 0.0, "last_end": 0.0, "last_status": 200} for a in agents}
LAST = {}
G = {a: {"state": "idle", "last_end": 0, "ratio": 3.0, "est_prompt": 0, "n": 0} for a in agents}   # pool-guard bookkeeping (under LOCK)
GUARD = {"holds": 0, "timeouts": 0, "wait_s": 0.0}


def _reserve(g):
    return RESERVE_SUMMARY if g.get("no_tools") else (RESERVE_FIRST if g["n"] <= 1 else RESERVE_GEN)


def _size(o):
    """KV tokens agent o holds right now (caller holds LOCK)."""
    g = G[o]
    if g["state"] == "run":
        return g["est_prompt"] + max(_reserve(g), C[o]["chunks"] - g.get("chunks0", 0))
    return g["last_end"]
REQLOG = open(os.path.join(outdir, "requests.jsonl"), "a", buffering=1)
HOP = {"host", "content-length", "connection", "transfer-encoding", "keep-alive", "proxy-connection", "te", "trailer", "upgrade"}


def upstream_conn():
    return http.client.HTTPConnection(UP.hostname, UP.port, timeout=3600)


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"      # close-delimited bodies: no chunk re-framing in the way of the stream

    def log_message(self, *a):
        pass

    def _agent(self):
        return agents[self.server.server_address[1] - first_port]

    def _forward(self, method):
        ag = self._agent()
        if self.path == "/_state":
            b = json.dumps(LAST).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            return
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else None
        hdr = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        hdr["X-Race-Agent"] = ag
        chat = method == "POST" and self.path.rstrip("/").endswith("/chat/completions")
        t0 = time.time()
        rec = {"agent": ag, "t0": round(t0, 3), "path": self.path, "status": 0, "chunks": 0}
        if chat:
            with LOCK:
                C[ag]["inflight"] += 1           # a held request counts as in flight (the recorder must not call the agent finished while it waits)
                C[ag]["reqs"] += 1
                C[ag]["last_start"] = t0
                g = G[ag]
                g["n"] += 1
                g["est_prompt"] = int(len(body or b"") / g["ratio"])
                try:
                    g["no_tools"] = not json.loads(body or b"{}").get("tools")
                except ValueError:
                    g["no_tools"] = False
                if g["no_tools"]:
                    g["n"] -= 1          # a summary/title call is not the agent's "first request"
            if POOL:
                held = False
                while True:
                    with LOCK:
                        others = sum(_size(o) for o in agents if o != ag)
                        running = any(G[o]["state"] == "run" for o in agents if o != ag)
                        need = G[ag]["est_prompt"] + _reserve(G[ag])
                        ok = (not running) or others + need <= POOL
                        if ok or time.time() - t0 >= WAIT_MAX:
                            if not ok:
                                GUARD["timeouts"] += 1
                                rec["guard_timeout"] = True
                            G[ag]["state"] = "run"
                            G[ag]["chunks0"] = C[ag]["chunks"]
                            rec["est_prompt"], rec["others_est"] = G[ag]["est_prompt"], others
                            break
                        if not held:
                            held = True
                            GUARD["holds"] += 1
                    time.sleep(0.25)
                rec["guard_wait_s"] = round(time.time() - t0, 2)
                with LOCK:
                    GUARD["wait_s"] += rec["guard_wait_s"]
            else:
                with LOCK:
                    G[ag]["state"] = "run"
                    G[ag]["chunks0"] = C[ag]["chunks"]
        try:
            cn = upstream_conn()
            cn.request(method, self.path, body=body, headers=hdr)
            r = cn.getresponse()
            rec["status"] = r.status
            self.send_response(r.status)
            for k, v in r.getheaders():
                if k.lower() not in HOP:
                    self.send_header(k, v)
            clen = r.getheader("Content-Length")
            if clen is not None:
                self.send_header("Content-Length", clen)
            self.send_header("Connection", "close")
            self.end_headers()
            buf = b""
            while True:
                piece = r.read1(65536) if hasattr(r, "read1") else r.read(8192)
                if not piece:
                    break
                try:
                    self.wfile.write(piece)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break
                if chat:
                    buf += piece
                    while b"\n\n" in buf:
                        ev, buf = buf.split(b"\n\n", 1)
                        self._count(ag, ev, rec)
            if chat and buf:
                self._count(ag, buf, rec)
            cn.close()
        except Exception as e:      # upstream down / reset: the agent sees a connection error, the race log keeps it
            rec["error"] = str(e)[:160]
            try:
                self.send_error(502, "upstream: %s" % str(e)[:80])
            except Exception:
                pass
        finally:
            if chat:
                t1 = time.time()
                rec["t1"] = round(t1, 3)
                with LOCK:
                    C[ag]["inflight"] -= 1
                    C[ag]["done"] += 1
                    C[ag]["last_end"] = t1
                    C[ag]["last_status"] = rec["status"] if not rec.get("error") else 0
                    g = G[ag]
                    g["state"] = "idle"
                    if rec["status"] != 200 or rec.get("error"):
                        C[ag]["err"] += 1
                        g["last_end"] = 0          # a rejected request leaves nothing of its own in the pool
                    elif rec.get("prompt_n") is not None:
                        tot = (rec.get("cache_n") or 0) + (rec.get("prompt_n") or 0)
                        rec["ctx_end"] = tot + (rec.get("predicted_n") or 0)
                        g["last_end"] = rec["ctx_end"]
                        if tot > 0 and body:
                            g["ratio"] = max(0.2, min(8.0, len(body) / tot))
                REQLOG.write(json.dumps(rec) + "\n")

    def _count(self, ag, ev, rec):
        for line in ev.split(b"\n"):
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == b"[DONE]":
                continue
            try:
                j = json.loads(payload)
            except ValueError:
                continue
            if isinstance(j.get("error"), dict):           # an error event inside a 200 stream (a slot dropped mid-decode)
                rec["sse_error"] = str(j["error"].get("message"))[:100]
                rec["error"] = rec.get("error") or "sse_error"
                continue
            ch = (j.get("choices") or [{}])[0].get("delta") or {}
            if ch.get("content") or ch.get("reasoning_content") or ch.get("reasoning") or ch.get("tool_calls"):
                with LOCK:
                    C[ag]["chunks"] += 1
                rec["chunks"] += 1
            tm = j.get("timings")
            if tm:
                with LOCK:
                    C[ag]["exact_pred"] += int(tm.get("predicted_n") or 0)
                    C[ag]["exact_prompt"] += int(tm.get("prompt_n") or 0)
                rec.update(prompt_n=tm.get("prompt_n"), predicted_n=tm.get("predicted_n"), pred_ps=tm.get("predicted_per_second"), prompt_ps=tm.get("prompt_per_second"),
                           cache_n=tm.get("cache_n"))

    def do_GET(self):
        self._forward("GET")

    def do_POST(self):
        self._forward("POST")


def get(path, timeout=2):
    cn = http.client.HTTPConnection(UP.hostname, UP.port, timeout=timeout)
    cn.request("GET", path)
    r = cn.getresponse()
    d = r.read()
    cn.close()
    return r.status, d


def prom(txt, name):
    m = re.search(r"^" + re.escape(name) + r"(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", txt, re.M)
    return float(m.group(1)) if m else None


def sampler():
    f = open(os.path.join(outdir, "hud.jsonl"), "a", buffering=1)
    vram = None
    last_v = 0.0
    while True:
        t = time.time()
        row = {"t": round(t, 3)}
        try:
            st, d = get("/metrics")
            txt = d.decode("utf8", "replace")
            for k, nm in (("pred_total", "llamacpp:tokens_predicted_total"), ("prompt_total", "llamacpp:prompt_tokens_total"), ("pred_seconds", "llamacpp:tokens_predicted_seconds_total"),
                          ("processing", "llamacpp:requests_processing"), ("deferred", "llamacpp:requests_deferred"), ("cached_total", "llamacpp:prompt_tokens_cached_total"),
                          ("n_tokens_max", "llamacpp:n_tokens_max"), ("spec_draft", "llamacpp:spec_decode_num_draft_tokens_total"),
                          ("spec_accept", "llamacpp:spec_decode_num_accepted_tokens_total"), ("busy_per_decode", "llamacpp:n_busy_slots_per_decode")):
                row[k] = prom(txt, nm)
        except Exception as e:
            row["metrics_err"] = str(e)[:80]
        try:
            st, d = get("/slots")
            sl = json.loads(d)
            row["slots_busy"] = sum(1 for s in sl if s.get("is_processing"))
            row["slots_n"] = len(sl)
        except Exception as e:
            row["slots_err"] = str(e)[:80]
        if t - last_v > 5:
            last_v = t
            try:
                vram = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=4).stdout.split()[0])
            except Exception:
                vram = None
        row["vram_mib"] = vram
        with LOCK:
            row["agents"] = {a: dict(v) for a, v in C.items()}
            if POOL:
                row["pool_est"] = {a: _size(a) for a in agents}
                row["guard"] = dict(GUARD)
        LAST.clear()
        LAST.update(row)
        f.write(json.dumps(row) + "\n")
        time.sleep(max(0.05, 1.0 / hz - (time.time() - t)))


if __name__ == "__main__":
    threading.Thread(target=sampler, daemon=True).start()
    servers = []
    for i, a in enumerate(agents):
        s = ThreadingHTTPServer(("127.0.0.1", first_port + i), H)
        s.daemon_threads = True
        servers.append(s)
        threading.Thread(target=s.serve_forever, daemon=True).start()
    print("four_proxy up: agents %s on %d..%d -> %s, out %s" % (agents, first_port, first_port + len(agents) - 1, up_url, outdir), flush=True)
    while True:
        time.sleep(3600)
