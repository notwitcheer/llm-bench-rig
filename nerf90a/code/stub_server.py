#!/usr/bin/env python3
"""stub_server.py <port> : a llama-server-shaped stub for the CPU proof of run_legs.py (never a result). /health, /v1/chat/completions.
Answers every multiple-choice prompt with 'A' and every code prompt with a tiny function body; usage.completion_tokens = 2. Stdlib only."""
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send({"status": "ok"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        sysmsg = (body.get("messages") or [{}])[0].get("content", "")
        text = "    return None" if "Python function" in sysmsg else "A"
        self._send({"choices": [{"message": {"role": "assistant", "content": text, "reasoning_content": ""}, "finish_reason": "stop"}], "usage": {"completion_tokens": 2, "prompt_tokens": 10}})


HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
