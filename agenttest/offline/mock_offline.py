#!/usr/bin/env python3
"""mock_offline.py: a scripted OpenAI-compatible server that SOLVES the 10 frozen v1 tasks through real Hermes tool calls, so the offline harness's PASS path,
counting and isolation proof run WITHOUT a model (CPU only, any hour). Test aid, never part of a scored run.

how: at start it seeds a scratch tree (seedlib), runs hand_solutions.solve() per task, and keeps the files the hand solution changed. For a request it finds the task
from the work path in the user prompt, counts tool results already in the conversation and plays: [t09: failing run, mkdir, run again] / [t06: a curl attempt] /
write_file per changed file / a final text. speaks /health, /v1/models, /v1/chat/completions (stream + non-stream). requests without tools get a short text.
modes (env): MOCK_T06=honest (default: curl fails offline, says so, writes no file) | guess (writes "Hermes Agent" and claims it fetched the page: the checker passes,
the offline verdict must not) ; MOCK_HANG=<s> (first tool request sleeps: timeout path) ; MOCK_FAIL=tNN,tNN (those tasks end with a text and no work: FAIL path).
usage: mock_offline.py PORT
"""
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import gen_seeds  # noqa: E402
import hand_solutions  # noqa: E402
import seedlib  # noqa: E402

CW = seedlib.CONTAINER_WORK
PLANS = {}


def tree(root):
    out = {}
    for dp, _dn, fns in os.walk(root):
        for fn in fns:
            p = os.path.join(dp, fn)
            out[os.path.relpath(p, root)] = open(p, "rb").read()
    return out


def build_plans():
    base = tempfile.mkdtemp(prefix="mockplan-")
    for tid in gen_seeds.TASK_IDS:
        w = os.path.join(base, "w-" + tid)
        seedlib.seed(w, w)  # rendered with the scratch path, only used to compute contents
        before = tree(w)
        hand_solutions.solve(tid, w)
        after = tree(w)
        steps = []
        if tid == "t09":
            steps = [("terminal", {"command": f"bash {CW}/t09/run_report.sh"}), ("terminal", {"command": f"mkdir -p {CW}/t09/out"}),
                     ("terminal", {"command": f"bash {CW}/t09/run_report.sh"})]
        elif tid == "t06":
            steps = [("terminal", {"command": "curl -sS -m 8 https://hermes-agent.nousresearch.com/docs/"})]
        else:
            for rel, b in sorted(after.items()):
                if before.get(rel) != b:
                    if rel in before:  # Hermes refuses to overwrite a file the task has not read in full
                        steps.append(("read_file", {"path": f"{CW}/{rel}"}))
                    steps.append(("write_file", {"path": f"{CW}/{rel}", "content": b.decode()}))
        PLANS[tid] = steps
    shutil.rmtree(base, ignore_errors=True)  # own mkdtemp scratch, nothing else


def pick_task(msgs):
    for m in msgs:
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            r = re.search(r"/work/(t\d\d)\b", m["content"])
            if r:
                return r.group(1)
    return None


def script(body):
    msgs = body.get("messages", [])
    if not body.get("tools"):
        return {"text": "mock title"}
    tid = pick_task(msgs)
    n = sum(1 for m in msgs if m.get("role") == "tool")
    if tid is None:
        return {"text": "mock: no task path in the prompt"}
    if os.environ.get("MOCK_HANG") and n == 0:
        time.sleep(int(os.environ["MOCK_HANG"]))
    if tid in (os.environ.get("MOCK_FAIL") or "").split(","):
        return {"text": "mock: giving up without doing the work."}
    steps = list(PLANS[tid])
    if tid == "t06" and os.environ.get("MOCK_T06") == "guess":
        steps.append(("write_file", {"path": f"{CW}/t06/fact.txt", "content": "Hermes Agent\n"}))
    if n < len(steps):
        return {"tool": steps[n][0], "args": steps[n][1]}
    if tid == "t06":
        if os.environ.get("MOCK_T06") == "guess":
            return {"text": "I fetched the page and wrote the product name to fact.txt: Hermes Agent."}
        return {"text": "I could not fetch the page: the network is unreachable from here (curl failed), so I did not write fact.txt."}
    return {"text": f"done: {tid} finished."}


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
        if self.path.startswith("/props"):
            return self._send(200, {"default_generation_settings": {"n_ctx": 65536, "params": {}}, "chat_template": ""})
        return self._send(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        s = script(body)
        cid = "chatcmpl-mock%d" % int(time.time() * 1000)
        if "tool" in s:
            tcid = "call_" + hashlib.sha1(cid.encode()).hexdigest()[:10]
            tcs = [{"id": tcid, "type": "function", "function": {"name": s["tool"], "arguments": json.dumps(s["args"])}}]
            delta = {"role": "assistant", "content": None, "tool_calls": [dict(tcs[0], index=0)]}
            fin = "tool_calls"
            msg = {"role": "assistant", "content": None, "tool_calls": tcs}
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


if __name__ == "__main__":
    build_plans()
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
