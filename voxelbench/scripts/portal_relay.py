#!/usr/bin/env python3
"""portal_relay.py: local OpenAI-compatible relay for HOSTED voxel runs (2026-10-24, mercury, GPU thread).

why a relay instead of pointing the bench profile at the Portal directly:
  1. run_agent.sh holds the model server with SIGSTOP while it queues /goal + /goal gate add (HARDENING: gate setup must
     not depend on model speed). a hosted model has no local pid; the relay is that pid, so the same hold works.
  2. one place that logs every request's usage and cost (relay-log.jsonl) and enforces a hard spend ceiling: past the
     ceiling every new chat request gets HTTP 402 and nothing more is billed.
  3. the credential stays in the relay process: the bench profile config carries a dummy key, the real bearer is never
     written into the run dir, the profile or any log.
  4. request shaping per the maker's API contract (e.g. Kimi K3.1 Pro: sampling fixed server-side, reasoning_effort
     low|high|max with default max; Hermes' custom route sends reasoning_effort "medium", which that API does not list).

usage: portal_relay.py --port 8191 --upstream https://inference-api.nousresearch.com/v1 \
         --key-source hermes_home:/home/witcheer/.hermes-portal-bench | file:/path/keyfile \
         --rates 2.0,0.2,10.0 --ceiling-usd 15 --log <dir>/relay-log.jsonl --state <dir>/relay-state.json \
         [--reasoning-effort omit|low|high|max|passthrough] [--include-usage]
endpoints: GET /health (local, no upstream call); everything else is relayed to <upstream>/<path minus a leading /v1>.
stdlib only. the key is never printed.
"""
import argparse, http.client, json, os, subprocess, sys, threading, time, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

A = None
LOCK = threading.Lock()
KEY = {"k": None, "exp": 0.0}
TOT = {"requests": 0, "ok": 0, "errors": 0, "blocked": 0, "prompt": 0, "cached": 0, "completion": 0,
       "reasoning": 0, "cost_usd": 0.0, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
HERMES_PY = "/home/witcheer/.hermes/hermes-agent/venv/bin/python"


def get_key(force=False):
    with LOCK:
        if not force and KEY["k"] and KEY["exp"] - time.time() > 600:
            return KEY["k"]
        kind, _, val = A.key_source.partition(":")
        if kind == "file":
            KEY["k"] = open(val).read().strip(); KEY["exp"] = time.time() + 10 ** 9
        elif kind == "hermes_home":
            code = ("import json; from hermes_cli.auth_nous import resolve_nous_runtime_credentials as r; "
                    f"d=r(force_refresh={bool(force)}); "
                    "print(json.dumps({'k': d['api_key'], 'e': d.get('expires_in') or 0}))")
            # the venv python alone misses the checkout root on sys.path (`pm` package): run from the source tree
            src = os.path.dirname(os.path.dirname(os.path.dirname(A.hermes_python)))
            env = dict(os.environ, HERMES_HOME=val, PYTHONPATH=src)
            out = subprocess.run([A.hermes_python, "-c", code], env=env, cwd=src, capture_output=True, text=True, timeout=60)
            if out.returncode != 0:
                raise RuntimeError("credential resolve failed (rc %d): %s" % (out.returncode, out.stderr.strip()[-300:]))
            d = json.loads(out.stdout.strip().splitlines()[-1])
            KEY["k"] = d["k"]; KEY["exp"] = time.time() + float(d["e"] or 0)
        else:
            raise RuntimeError("unknown --key-source kind %r" % kind)
        return KEY["k"]


def usage_cost(u):
    p = int(u.get("prompt_tokens") or 0); c = int(u.get("completion_tokens") or 0)
    det = u.get("prompt_tokens_details") or {}
    cached = int(det.get("cached_tokens") or u.get("cache_read_input_tokens") or u.get("cached_tokens") or 0)
    rdet = u.get("completion_tokens_details") or {}
    reas = int(rdet.get("reasoning_tokens") or 0)
    rin, rcache, rout = A.rate_tuple
    cost = (max(p - cached, 0) * rin + cached * rcache + c * rout) / 1e6
    # Nous Portal returns its own billed estimate in usage.cost (matched our catalogue maths to 4 decimals in the
    # 2026-10-24 probe); prefer it for the ceiling, keep ours as a cross-check
    if isinstance(u.get("cost"), (int, float)):
        cost = max(cost, float(u["cost"]))
    return p, cached, c, reas, cost


def write_state():
    tmp = A.state + ".tmp"
    with open(tmp, "w") as f:
        json.dump(dict(TOT, ceiling_usd=A.ceiling_usd, rates_per_m=A.rates, updated=time.strftime("%H:%M:%S")), f)
    os.replace(tmp, A.state)


def log(rec):
    with open(A.log, "a") as f:
        f.write(json.dumps(rec) + "\n")


def find_usage(tail_bytes):
    """last non-null usage object in an SSE tail or a JSON body."""
    txt = tail_bytes.decode("utf-8", "replace")
    best = None
    for line in txt.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line.startswith("{") or '"usage"' not in line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        if o.get("usage"):
            best = o["usage"]
    if best is None:
        try:
            o = json.loads(txt)
            best = o.get("usage")
        except Exception:
            pass
    return best


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", "/v1/health"):
            return self._json(200, {"status": "ok", "relay": True, "spent_usd": round(TOT["cost_usd"], 4)})
        self._relay(None)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        self._relay(body)

    def _relay(self, body):
        t0 = time.time()
        path = self.path[3:] if self.path.startswith("/v1/") else self.path
        is_chat = path.startswith("/chat/completions")
        model = None; shaped = []
        if body and is_chat:
            if TOT["cost_usd"] >= A.ceiling_usd:
                TOT["blocked"] += 1; write_state()
                log({"t": time.strftime("%H:%M:%S"), "blocked": True, "spent": round(TOT["cost_usd"], 4)})
                return self._json(402, {"error": {"message": "relay spend ceiling reached (%.2f USD); no further requests are sent upstream" % A.ceiling_usd, "type": "spend_ceiling"}})
            try:
                req = json.loads(body)
                model = req.get("model")
                if A.reasoning_effort == "omit" and "reasoning_effort" in req:
                    shaped.append("reasoning_effort %r dropped" % req.pop("reasoning_effort"))
                elif A.reasoning_effort in ("low", "high", "max"):
                    if req.get("reasoning_effort") != A.reasoning_effort:
                        shaped.append("reasoning_effort %r -> %r" % (req.get("reasoning_effort"), A.reasoning_effort))
                    req["reasoning_effort"] = A.reasoning_effort
                for k in A.drop_fields:
                    if k in req:
                        shaped.append("%s dropped" % k); req.pop(k)
                if A.include_usage and req.get("stream") and not (req.get("stream_options") or {}).get("include_usage"):
                    req["stream_options"] = dict(req.get("stream_options") or {}, include_usage=True); shaped.append("include_usage on")
                body = json.dumps(req).encode()
            except Exception as e:
                shaped.append("body not json (%s), relayed as is" % e)
        u = urllib.parse.urlparse(A.upstream)
        upath = u.path.rstrip("/") + path
        for attempt in (0, 1):
            try:
                key = get_key(force=bool(attempt))
            except Exception as e:
                TOT["errors"] += 1; write_state()
                log({"t": time.strftime("%H:%M:%S"), "error": "key: %s" % e})
                return self._json(502, {"error": {"message": "relay could not resolve a credential: %s" % e}})
            conn = (http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection)(
                u.hostname, u.port or (443 if u.scheme == "https" else 80), timeout=A.timeout)
            hdr = {"Authorization": "Bearer " + key, "Content-Type": self.headers.get("Content-Type", "application/json"),
                   "Accept": self.headers.get("Accept", "*/*"), "User-Agent": "voxelbench-relay/1"}
            try:
                conn.request(self.command, upath, body=body, headers=hdr)
                r = conn.getresponse()
            except Exception as e:
                TOT["errors"] += 1; write_state()
                log({"t": time.strftime("%H:%M:%S"), "error": "upstream: %s" % e, "path": path})
                return self._json(502, {"error": {"message": "relay upstream error: %s" % e}})
            if r.status == 401 and attempt == 0:
                r.read(); conn.close(); continue
            break
        self.send_response(r.status)
        for k in ("Content-Type", "Cache-Control"):
            v = r.getheader(k)
            if v:
                self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
        tail = bytearray(); total = 0
        try:
            while True:
                chunk = r.read1(65536)
                if not chunk:
                    break
                total += len(chunk)
                self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n"); self.wfile.flush()
                tail += chunk
                if len(tail) > 262144:
                    del tail[:-131072]
            self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            conn.close()
        rec = {"t": time.strftime("%H:%M:%S"), "path": path, "status": r.status, "model": model,
               "bytes": total, "dur_s": round(time.time() - t0, 1)}
        if shaped:
            rec["shaped"] = shaped
        if is_chat:
            TOT["requests"] += 1
            if r.status == 200:
                TOT["ok"] += 1
            else:
                TOT["errors"] += 1; rec["err_tail"] = bytes(tail[-600:]).decode("utf-8", "replace")
            us = find_usage(bytes(tail))
            if us:
                p, cached, c, reas, cost = usage_cost(us)
                TOT["prompt"] += p; TOT["cached"] += cached; TOT["completion"] += c; TOT["reasoning"] += reas
                TOT["cost_usd"] += cost
                rec.update(prompt=p, cached=cached, completion=c, reasoning=reas, cost=round(cost, 5))
            elif r.status == 200:
                rec["usage"] = "missing"
            rec["spent"] = round(TOT["cost_usd"], 4)
            write_state()
        log(rec)


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8191)
    ap.add_argument("--upstream", default="https://inference-api.nousresearch.com/v1")
    ap.add_argument("--key-source", required=True)
    ap.add_argument("--hermes-python", default=HERMES_PY)
    ap.add_argument("--rates", required=True, help="USD per M tokens: input,cache_read,output")
    ap.add_argument("--ceiling-usd", type=float, required=True)
    ap.add_argument("--reasoning-effort", default="passthrough", choices=["passthrough", "omit", "low", "high", "max"])
    ap.add_argument("--drop-fields", default="", help="comma list of request fields to drop (e.g. temperature,top_p)")
    ap.add_argument("--include-usage", action="store_true")
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--log", required=True); ap.add_argument("--state", required=True)
    A = ap.parse_args()
    A.rate_tuple = tuple(float(x) for x in A.rates.split(","))
    A.drop_fields = [x for x in A.drop_fields.split(",") if x]
    get_key()  # fail fast before Hermes starts
    write_state()
    srv = ThreadingHTTPServer(("127.0.0.1", A.port), H)
    srv.daemon_threads = True
    print("relay up on 127.0.0.1:%d -> %s (ceiling %.2f USD, effort %s)" % (A.port, A.upstream, A.ceiling_usd, A.reasoning_effort), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
