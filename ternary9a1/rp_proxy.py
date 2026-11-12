#!/usr/bin/env python3
"""rp_proxy.py: counting proxy + HUD sampler for item 78z (one Hermes agent per take; vLLM or llama-server upstream). Derived from four_proxy.py (item 80) with one change:
the sampler also writes `srv_tokens`, a server-side generated-token count that is live (not only at request end) on BOTH engines, each counted from its own stream: llama-server = exact predicted_n of finished requests (proxy, from `timings`) + the SSE deltas of the request in flight (v2: the item-85 version added /slots n_decoded and double-counted the last request), vLLM = the vllm:generation_tokens_total counter on /metrics. The renderer uses it for the HUD so the two panes are read the same way;
the proxy stream-delta count (`chunks`) stays as the cross-check (vLLM may emit a tool call in a few big deltas, so deltas are NOT tokens there).
Launch with `--agents a` (one listen port).

Original docstring (item 80):
four_proxy.py: per-agent counting proxy + HUD sampler for item 80 (four Hermes agents, one llama-server, --parallel 4).

usage: four_proxy.py <upstream http://127.0.0.1:PORT> <first-listen-port> <outdir> [--agents abcd] [--hz 1.0]

* listens on first-port .. first-port+3, one port per agent (a b c d). Each agent's Hermes profile points its base_url at its own port, so every
  request is attributed to an agent without touching the request body (llama-server has no notion of a client). The proxy adds `X-Race-Agent`
  and forwards the stream byte for byte; a hold-up here would be a hold-up in the race, so it does no parsing before forwarding.
* counts streamed deltas per agent (content, reasoning_content, tool_calls pieces: one SSE delta is about one token on llama.cpp; the exact
  per-request numbers come from the `timings` object of the last chunk and are logged next to it). llama-server's own
  `llamacpp:tokens_predicted_total` (/metrics) is the aggregate truth; the sampler logs both so the HUD can say which it shows.
* sampler thread, every 1/hz s: upstream /metrics + /slots (+ nvidia-smi memory every 5 s) + the per-agent counters -> <outdir>/hud.jsonl
  (cumulative counters; the renderer differentiates). <outdir>/requests.jsonl = one line per finished request.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rp_count import llama_srv_tokens      # noqa: E402

up_url, first_port, outdir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
agents = "abcd"
hz = 1.0
if "--agents" in sys.argv:
    agents = sys.argv[sys.argv.index("--agents") + 1]
if "--hz" in sys.argv:
    hz = float(sys.argv[sys.argv.index("--hz") + 1])
UP = urlparse(up_url)
os.makedirs(outdir, exist_ok=True)
LOCK = threading.Lock()
C = {a: {"chunks": 0, "inflight": 0, "reqs": 0, "done": 0, "err": 0, "exact_pred": 0, "exact_prompt": 0, "last_start": 0.0, "last_end": 0.0,
         "srv_done": 0, "done_chunks": 0, "final_seen": False, "aborted_est": 0} for a in agents}
LAST = {}
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
        if chat:
            with LOCK:
                C[ag]["inflight"] += 1
                C[ag]["reqs"] += 1
                C[ag]["last_start"] = t0
                C[ag]["final_seen"] = False
        rec = {"agent": ag, "t0": round(t0, 3), "path": self.path, "status": 0, "chunks": 0}
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
                    if not C[ag]["final_seen"]:       # stream cut before its last chunk (wall, reset): the deltas so far are the best count of what it generated
                        est = C[ag]["chunks"] - C[ag]["done_chunks"]
                        C[ag]["srv_done"] += est
                        C[ag]["aborted_est"] += est
                        C[ag]["done_chunks"] = C[ag]["chunks"]
                    if rec["status"] != 200 or rec.get("error"):
                        C[ag]["err"] += 1
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
            ch = (j.get("choices") or [{}])[0].get("delta") or {}
            if ch.get("content") or ch.get("reasoning_content") or ch.get("reasoning") or ch.get("tool_calls"):
                with LOCK:
                    C[ag]["chunks"] += 1
                rec["chunks"] += 1
            tm = j.get("timings")
            us = j.get("usage")
            if not tm and isinstance(us, dict) and us.get("completion_tokens") is not None:      # vLLM (stream_options.include_usage): same fields, no per-second rate
                tm = {"predicted_n": us.get("completion_tokens"), "prompt_n": us.get("prompt_tokens")}
            if tm:
                with LOCK:
                    C[ag]["exact_pred"] += int(tm.get("predicted_n") or 0)
                    C[ag]["exact_prompt"] += int(tm.get("prompt_n") or 0)
                    # this request is now counted EXACTLY (srv_done) and its deltas are retired (done_chunks) in one step: see llama_srv_tokens()
                    C[ag]["srv_done"] += int(tm.get("predicted_n") or 0)
                    C[ag]["done_chunks"] = C[ag]["chunks"]
                    C[ag]["final_seen"] = True
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
    cum = 0.0                 # server-side token count (see module docstring)
    vllm_prev = None          # vLLM: last generation_tokens_total
    engine = None
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
            vg = prom(txt, "vllm:generation_tokens_total")
            if vg is not None:
                engine = "vllm"
                row["vllm_gen_total"] = vg
                row["vllm_running"] = prom(txt, "vllm:num_requests_running")
                row["vllm_prompt_total"] = prom(txt, "vllm:prompt_tokens_total")
                row["vllm_kv_usage"] = prom(txt, "vllm:kv_cache_usage_perc")
                if vllm_prev is not None:
                    cum += max(0.0, vg - vllm_prev) if vg >= vllm_prev else vg
                vllm_prev = vg
        except Exception as e:
            row["metrics_err"] = str(e)[:80]
        if engine != "vllm":
            try:
                st, d = get("/slots")
                sl = json.loads(d)
                row["slots_busy"] = sum(1 for s in sl if s.get("is_processing"))
                row["slots_n"] = len(sl)
                engine = "llama"
                # llama-server: its own stream. The count of a FINISHED request is its exact `timings.predicted_n` (proxy, last chunk); the request in flight by the slot's n_decoded (cross-checked against the SSE deltas,
                # 99.3% of tokens on the 2026-11-03 take). The item-85 version summed n_decoded with the finished-request count for EVERY processing slot: the slot still reads is_processing with its full n_decoded for a moment after
                # the last chunk, so the finished request was counted twice, and a max() ratchet froze the error in (66,772 shown for 36,356 generated). Now: a request whose `timings` have arrived is never read from the slot, and no ratchet.
                row["slots_n_decoded"] = sum(int(((s_.get("next_token") or [{}])[0]).get("n_decoded") or 0) for s_ in sl if s_.get("is_processing"))
                with LOCK:
                    cum = llama_srv_tokens(C, row["slots_n_decoded"])
            except Exception as e:
                row["slots_err"] = str(e)[:80]
        row["engine"] = engine
        row["srv_tokens"] = round(cum, 1) if engine else None
        if t - last_v > 5:
            last_v = t
            try:
                vram = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=4).stdout.split()[0])
            except Exception:
                vram = None
        row["vram_mib"] = vram
        with LOCK:
            row["agents"] = {a: dict(v) for a, v in C.items()}
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
