#!/usr/bin/env python3
"""rc_mockserver.py <port>: a MOCK OpenAI-compatible source/builder for the CPU proof (never counted). It answers the house-model role only:
  - 'Solve this problem' prompts: a canned worked solution that ends with the golden label (deliberately WRONG on every 7th question, except the build instances,
    so the scorer sees both cases)
  - builder prompts: a hand-written solve() + schema + params for three templates (365, 269, 636); template 269's first reply is a deliberate bad function so the
    validate-and-rewrite loop runs.
The 1B extractor and the embedder in the proof are the REAL servers."""
import json, os, re, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rc_common as C

ROWS, BUILDQ = {}, set()
for sp in ("main", "p1"):
    for r in C.load_rows(sp):
        ROWS[r["question"]] = r
        if sp == "main" and r["instance"] == 0:
            BUILDQ.add(r["question"])
COUNT = {"n": 0}

ENT = {
    365: {"description": "the number of kids in a group when one count is given and the other is a stated ratio of it", "schema": [
        {"name": "girls", "description": "how many girls", "example": 18}, {"name": "ratio_num", "description": "numerator of the stated ratio (a 'times' word is the multiplier over 1)", "example": 2},
        {"name": "ratio_den", "description": "denominator of the stated ratio (1 for a plain 'times')", "example": 3}], "params": {"girls": 18, "ratio_num": 2, "ratio_den": 3},
        "code": "def solve(params):\n    g = params['girls']\n    boys = g * params['ratio_num'] / params['ratio_den']\n    return g + boys\n"},
    269: {"description": "how many items of one kind are left after taking two successive fractions of a total", "schema": [
        {"name": "total", "description": "the total number of items", "example": 640}, {"name": "a_num", "description": "numerator of the first fraction", "example": 1},
        {"name": "a_den", "description": "denominator of the first fraction", "example": 8}, {"name": "b_num", "description": "numerator of the second fraction", "example": 1},
        {"name": "b_den", "description": "denominator of the second fraction", "example": 10}], "params": {"total": 640, "a_num": 1, "a_den": 8, "b_num": 1, "b_den": 10},
        "code": "def solve(params):\n    return params['total'] * params['a_num'] / params['a_den'] * params['b_num'] / params['b_den']\n",
        "bad": "def solve(params):\n    return params['total'] * params['a_num'] / params['a_den'] + params['b_num'] / params['b_den']\n"},
    636: {"description": "the weekly total of work done by two people when one works a stated number of times as fast as the other", "schema": [
        {"name": "factor", "description": "how many times as fast the faster person is", "example": 3}, {"name": "daily", "description": "daily output of the slower person", "example": 55}],
        "params": {"factor": 3, "daily": 55}, "code": "def solve(params):\n    return (params['daily'] + params['daily'] * params['factor']) * 7\n"},
}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        self._send({"status": "ok"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        msgs = body["messages"]; first = msgs[0]["content"]
        if first.startswith("Solve this problem"):
            q = first.split("`#### <number>` (digits only, no units, no commas).\n\n", 1)[1]
            r = ROWS.get(q)
            COUNT["n"] += 1
            gold = C.golden_of(r) if r else 0
            ans = gold if (q in BUILDQ or COUNT["n"] % 7) else gold + 1
            reasoning = "Let me read the problem again. %s I work it through step by step and check the arithmetic twice." % q[:60]
            content = "Working it out.\n#### %d" % ans if float(ans).is_integer() else "#### %s" % ans
            msg = {"role": "assistant", "content": content, "reasoning_content": reasoning}
            u = {"completion_tokens": 400 + (COUNT["n"] * 37) % 900, "prompt_tokens": 80}
        else:
            q = re.search(r"QUESTION:\n(.*?)\n\nTRANSCRIPT:", first, re.S).group(1)
            r = ROWS[q]; tid = r["original_id"]; e = ENT[tid]
            retry = len(msgs) > 1 and "Your function returned" in msgs[-1]["content"]
            code = e["code"] if (retry or "bad" not in e) else e["bad"]
            j = {"description": e["description"], "schema": e["schema"], "params": e["params"]}
            msg = {"role": "assistant", "content": "```json\n%s\n```\n```python\n%s```" % (json.dumps(j), code), "reasoning_content": "mock"}
            u = {"completion_tokens": 900, "prompt_tokens": 1500}
        self._send({"choices": [{"message": msg, "finish_reason": "stop"}], "usage": u, "timings": {"predicted_per_second": 40.0}})


ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
