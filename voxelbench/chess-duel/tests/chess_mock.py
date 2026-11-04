#!/usr/bin/env python3
"""chess_mock.py: llama-server-shaped MOCK MODEL for the chess-duel PASS path (CPU only, no model, test aid only, never part of a run).

Speaks what the harness needs from llama-server: /health, /v1/models, /props, /metrics, /slots and streaming /v1/chat/completions (reasoning deltas,
tool-call argument chunks). It plays a fixed script keyed on the tool results since the last user message, and answers the Hermes /goal judge call.

  first goal turn      reasoning, write_file(<cwd>/engine.py, MOCK_FIRST variant of the stub engines), terminal(pipe check), terminal(perft 3), final text
  after a gate failure (the user message carries the gate output)   write_file(<cwd>/engine.py, good.py), terminal(pipe check), final text
  judge call (no tools, system prompt mentions the judge)           {"verdict": "done", "reason": ...}  [MOCK_JUDGE=continue answers "continue" once]

usage: MOCK_DIR=<dir with good.py bad_*.py> MOCK_FIRST=good|bad_import|none MOCK_SCALE=1.0 chess_mock.py PORT
  MOCK_FIRST=none: never writes a file (the "agent produced nothing" red path: the goal ends with no engine.py)
"""
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SCALE = float(os.environ.get("MOCK_SCALE", "1.0"))
MOCK_DIR = os.environ.get("MOCK_DIR", ".")
FIRST = os.environ.get("MOCK_FIRST", "good")
RATE = float(os.environ.get("MOCK_RATE", "70"))
LOCK = threading.Lock()
STATE = {"predicted": 0, "prompt": 0, "pred_seconds": 0.0, "busy": False, "decoded": 0, "requests": 0, "judge_calls": 0}


def engine_text(name):
    return open(os.path.join(MOCK_DIR, name + ".py")).read()


def msg_text(m):
    c = m.get("content")
    if isinstance(c, list):
        return " ".join(x.get("text", "") for x in c if isinstance(x, dict))
    return c if isinstance(c, str) else ""


def script(body):
    msgs = body.get("messages", [])
    if not body.get("tools"):
        sysm = " ".join(msg_text(m) for m in msgs if m.get("role") == "system").lower()
        if "judge" in sysm or "verdict" in sysm:
            with LOCK:
                STATE["judge_calls"] += 1
                n = STATE["judge_calls"]
            if os.environ.get("MOCK_JUDGE") == "continue" and n == 1:
                return {"text": json.dumps({"verdict": "continue", "reason": "mock judge: not yet"})}
            return {"text": json.dumps({"verdict": "done", "reason": "mock judge: the engine was written and checked"})}
        return {"text": "mock title"}
    cwd = None
    for m in msgs:
        mm = re.search(r"exactly (\S+?)/engine\.py", msg_text(m))
        if mm:
            cwd = mm.group(1)
            break
    if cwd is None:   # the warm-up turn the driver sends before /goal
        return {"text": "ready", "think": 3}
    last_user = max((i for i, m in enumerate(msgs) if m.get("role") == "user"), default=-1)
    k = sum(1 for m in msgs[last_user + 1:] if m.get("role") == "tool")
    ut = msg_text(msgs[last_user]) if last_user >= 0 else ""
    fix = "gate failed" in ut.lower()
    which = "good" if fix else FIRST
    if FIRST == "none":    # the stubborn agent: never writes a file, even when the gate tells it to
        return {"text": "mock: nothing written", "think": 20}
    if fix:   # Hermes refuses to overwrite a file the task has not read in full: read it, then write
        if k == 0:
            return {"tool": "read_file", "args": {"path": cwd + "/engine.py"}, "think": 40}
        if k == 1:
            return {"tool": "write_file", "args": {"path": cwd + "/engine.py", "content": engine_text("good")}, "think": 60}
        if k == 2:
            return {"tool": "terminal", "args": {"command": "cd %s && printf 'uci\\nisready\\nposition startpos\\ngo movetime 300\\nquit\\n' | python3 engine.py | tail -5" % cwd}, "think": 20}
        return {"text": "fixed: engine.py rewritten without the forbidden import; handshake checked.", "think": 20}
    if k == 0:
        return {"tool": "write_file", "args": {"path": cwd + "/engine.py", "content": engine_text(which)}, "think": 160}
    if k == 1:
        return {"tool": "terminal", "args": {"command": "cd %s && printf 'uci\\nisready\\nposition startpos\\ngo movetime 300\\nquit\\n' | python3 engine.py | tail -5" % cwd}, "think": 30}
    if k == 2 and not fix:
        return {"tool": "terminal", "args": {"command": "cd %s && printf 'position startpos\\ngo perft 3\\nquit\\n' | python3 engine.py | tail -3" % cwd}, "think": 30}
    return {"text": "engine.py is written: legal move generator, alpha-beta search, UCI shell, go perft. checked the handshake and perft 3 from a pipe.", "think": 25}


def chunks_for(s):
    out = []
    for i in range(int(s.get("think", 0) * SCALE)):
        out.append(("r", "hmm " if i % 7 else "so the board. "))
    if "tool" in s:
        a = json.dumps(s["args"])
        step = max(6, len(a) // max(60, int(300 * SCALE)))
        out += [("t", a[i:i + step]) for i in range(0, len(a), step)]
    else:
        t = s["text"]
        out += [("c", t[i:i + 5]) for i in range(0, len(t), 5)]
    return out


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, code, obj, ctype="application/json"):
        b = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p.endswith("/health"):
            return self._json(200, {"status": "ok"})
        if p.endswith("/models"):
            return self._json(200, {"object": "list", "data": [{"id": "mock-chess", "object": "model"}]})
        if p == "/props":
            return self._json(200, {"default_generation_settings": {"params": {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.05}, "n_ctx": 131072}, "total_slots": 1})
        if p == "/metrics":
            with LOCK:
                txt = ("llamacpp:tokens_predicted_total %d\nllamacpp:prompt_tokens_total %d\nllamacpp:tokens_predicted_seconds_total %.3f\nllamacpp:requests_processing %d\n"
                       % (STATE["predicted"], STATE["prompt"], STATE["pred_seconds"], 1 if STATE["busy"] else 0))
            return self._json(200, txt.encode(), "text/plain")
        if p == "/slots":
            with LOCK:
                return self._json(200, [{"id": 0, "n_ctx": 131072, "is_processing": STATE["busy"], "next_token": [{"n_decoded": STATE["decoded"]}]}])
        return self._json(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        s = script(body)
        ch = chunks_for(s)
        cid = "chatcmpl-mock%d" % int(time.time() * 1000)
        if not body.get("stream"):
            return self._json(200, {"id": cid, "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": s.get("text", "mock")}, "finish_reason": "stop"}],
                                    "usage": {"prompt_tokens": 40, "completion_tokens": 20}})
        with LOCK:
            STATE["busy"], STATE["decoded"], STATE["requests"], STATE["prompt"] = True, 0, STATE["requests"] + 1, STATE["prompt"] + 3000
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def send(obj):
            raw = ("data: " + json.dumps(obj) + "\n\n").encode()
            self.wfile.write(("%x\r\n" % len(raw)).encode() + raw + b"\r\n")
            self.wfile.flush()
        tcid, first_t, t0 = "call_%d" % int(time.time() * 1000), True, time.time()
        try:
            send({"id": cid, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})
            for kind, txt in ch:
                time.sleep(1.0 / RATE)
                if kind == "r":
                    d = {"reasoning_content": txt}
                elif kind == "c":
                    d = {"content": txt}
                else:
                    tc = {"index": 0, "function": {"arguments": txt}}
                    if first_t:
                        tc.update({"id": tcid, "type": "function"})
                        tc["function"]["name"] = s["tool"]
                        first_t = False
                    d = {"tool_calls": [tc]}
                send({"id": cid, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": d, "finish_reason": None}]})
                with LOCK:
                    STATE["predicted"] += 1
                    STATE["decoded"] += 1
            el = max(time.time() - t0, 1e-3)
            send({"id": cid, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if "tool" in s else "stop"}],
                  "usage": {"prompt_tokens": 3000, "completion_tokens": len(ch), "total_tokens": 3000 + len(ch)},
                  "timings": {"prompt_n": 3000, "predicted_n": len(ch), "predicted_per_second": round(len(ch) / el, 2)}})
            raw = b"data: [DONE]\n\n"
            self.wfile.write(("%x\r\n" % len(raw)).encode() + raw + b"\r\n0\r\n\r\n")
            self.wfile.flush()
            with LOCK:
                STATE["pred_seconds"] += el
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with LOCK:
                STATE["busy"] = False


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
