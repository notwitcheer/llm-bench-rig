#!/usr/bin/env python3
"""desk_mock.py PORT SCRIPT.json : scripted OpenAI-compatible model for the desktop arena (CPU proof of the computer_use path, no model noise).
speaks /health, /v1/models, /v1/chat/completions (stream + non-stream). SCRIPT.json = list of steps, one per tool-carrying request, played in order
(keyed on the number of tool results already in the conversation). A step is {"args": {...computer_use args...}} or {"text": "final answer"}.
A string arg value of the form "@el:<regex>" is resolved against the LAST tool result text: the SOM/AX line matching the regex yields its element index
(the first integer in a leading '#N' / '[N]' / 'N:' token of that line); no match -> the request answers text 'MOCKFAIL no element for <regex>' so the
run shows the miss instead of clicking blind. Requests without tools (title generation) get a short text. Every request body is appended to
<script>.requests.jsonl (message roles + tool names + image part count only, never image bytes). test aid only, never part of a run.
"""
import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1])
SCRIPT = json.load(open(sys.argv[2]))
LOG = sys.argv[2] + ".requests.jsonl"


def last_tool_text(msgs):
    for m in reversed(msgs):
        if m.get("role") == "tool":
            c = m.get("content")
            if isinstance(c, list):
                c = "\n".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
            c = c or ""
            try:  # computer_use results are JSON; the numbered element lines live in "summary"
                d = json.loads(c)
                if isinstance(d, dict) and d.get("summary"):
                    return d["summary"]
            except ValueError:
                pass
            return c
    return ""


def resolve(args, text):
    out = {}
    for k, v in args.items():
        if isinstance(v, str) and v.startswith("@el:"):
            pat = re.compile(v[4:], re.I)
            hit = None
            for line in text.splitlines():
                if pat.search(line):
                    mm = re.match(r"\s*(?:#|\[)?(\d+)(?:\]|:|\s)", line)
                    if mm:
                        hit = int(mm.group(1))
                        break
            if hit is None:
                return None, v[4:]
            out[k] = hit
        else:
            out[k] = v
    return out, None


def step(body):
    msgs = body.get("messages", [])
    n = sum(1 for m in msgs if m.get("role") == "tool")
    if not body.get("tools"):
        return {"text": "mock title"}
    if n >= len(SCRIPT):
        return {"text": "mock: script finished"}
    s = SCRIPT[n]
    if "text" in s:
        return s
    if "tool" in s:
        return s
    args, miss = resolve(s["args"], last_tool_text(msgs))
    if args is None:
        return {"text": "MOCKFAIL no element for " + miss}
    return {"tool": "computer_use", "args": args}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/health"):
            return self._send(200, {"status": "ok"})
        if self.path.startswith("/v1/models"):
            return self._send(200, {"object": "list", "data": [{"id": "mock", "object": "model"}]})
        return self._send(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        s = step(body)
        msgs = body.get("messages", [])
        imgs = 0
        for m in msgs:
            if isinstance(m.get("content"), list):
                imgs += sum(1 for p in m["content"] if isinstance(p, dict) and p.get("type") == "image_url")
        with open(LOG, "a") as f:
            f.write(json.dumps({"t": int(time.time()), "n_msgs": len(msgs), "tools": [t["function"]["name"] for t in body.get("tools", [])],
                                "tool_results": sum(1 for m in msgs if m.get("role") == "tool"), "image_parts": imgs, "reply": s.get("args") or s.get("text")}) + "\n")
        cid = "chatcmpl-mock%d" % int(time.time() * 1000)
        if "tool" in s:
            tc = {"id": "call_%d" % int(time.time() * 1000), "type": "function", "function": {"name": s["tool"], "arguments": json.dumps(s["args"])}}
            delta = {"role": "assistant", "content": None, "tool_calls": [dict(tc, index=0)]}
            fin = "tool_calls"
            msg = {"role": "assistant", "content": None, "tool_calls": [tc]}
        else:
            delta = {"role": "assistant", "content": s["text"]}
            fin = "stop"
            msg = {"role": "assistant", "content": s["text"]}
        usage = {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for ch in ({"id": cid, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                       {"id": cid, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": fin}], "usage": usage}):
                self.wfile.write(("data: " + json.dumps(ch) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            self._send(200, {"id": cid, "object": "chat.completion", "choices": [{"index": 0, "message": msg, "finish_reason": fin}], "usage": usage})


ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
