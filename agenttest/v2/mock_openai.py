#!/usr/bin/env python3
"""mock_openai.py: a scripted OpenAI-compatible server to prove the runner's PASS path and the tool-error / approval-block counting WITHOUT a model.
speaks /health, /v1/models, /v1/chat/completions (stream and non-stream). For a request that carries tools it plays a fixed script keyed on the number of tool results already in the conversation:
  0 -> terminal(python3 -c ...)  [Hermes' default approvals block this headless: counted as an approval block]
  1 -> terminal(cat /nonexistent)  [a plain tool error, counted separately]
  2 -> write_file(<path from the prompt>/out/notes.txt, the exact t01 content)
  3+ -> final text
requests without tools (title generation) get a short text. test aid only, never part of a run.
usage: [MOCK_HANG=seconds] mock_openai.py PORT   (MOCK_HANG makes the first tool-carrying request hang: the harness timeout / deadline path)"""
import json
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CONTENT = "release: v2.4.1\n  indented by two spaces\nquotes 'single' and \"double\" stay as typed\n"


def script(body):
    msgs = body.get("messages", [])
    n = sum(1 for m in msgs if m.get("role") == "tool")
    if not body.get("tools"):
        return {"text": "mock title"}
    prompt = next((m["content"] for m in msgs if m.get("role") == "user" and isinstance(m.get("content"), str) and "notes.txt" in m["content"]), "")
    m = re.search(r"(/\S+?)/out and", prompt)
    base = m.group(1) if m else "/tmp"
    if n == 0:
        if os.environ.get("MOCK_HANG"):  # deadline-path test: the model never answers
            time.sleep(int(os.environ["MOCK_HANG"]))
        return {"tool": "terminal", "args": {"command": "python3 -c \"print(1)\""}}
    if n == 1:
        return {"tool": "terminal", "args": {"command": "cat /nonexistent/file.txt"}}
    if n == 2:
        return {"tool": "write_file", "args": {"path": base + "/out/notes.txt", "content": CONTENT}}
    return {"text": "done, the file is written and verified."}


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
        s = script(body)
        cid = "chatcmpl-mock%d" % int(time.time() * 1000)
        if "tool" in s:
            delta = {"role": "assistant", "content": None, "tool_calls": [{"index": 0, "id": "call_%d" % int(time.time() * 1000), "type": "function", "function": {"name": s["tool"], "arguments": json.dumps(s["args"])}}]}
            fin = "tool_calls"
            msg = {"role": "assistant", "content": None, "tool_calls": [{"id": delta["tool_calls"][0]["id"], "type": "function", "function": delta["tool_calls"][0]["function"]}]}
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


ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
