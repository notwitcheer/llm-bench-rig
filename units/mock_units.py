#!/usr/bin/env python3
"""mock OpenAI-compatible chat server for the units experiment (CPU, no model). SCRIPTED answers, never model output.
GET /health, GET /props, POST /v1/chat/completions. The "model" field picks a behaviour profile (mock-*), the user prompt is looked up by sha256 in tasks.json.
profiles:
  mock-perfect  every call exact
  mock-slippy   exact except scripted power-of-ten slips (x10, x100, x1000, /10, /1000), one other_value, one malformed JSON, one no_call, one wrong_tool
  mock-clean    exact except 2 slips (the CLEAN boundary) and one other_value
  mock-three    exact except exactly 3 slips (the SLIPS FOUND boundary)
  mock-broken   30% no_call (text answer, no tool call)
  mock-copycat  copies the stated number for non-canonical forms (the classic failure: 90000 for 90 s)
  mock-shapes   exact values but sent in the other wire shapes: arguments as a dict, a numeric string, floats for integers (e.g. 90.0), reasoning_content present
usage: python mock_units.py PORT"""
import hashlib
import json
import os
import sys
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = json.load(open(os.path.join(HERE, "tasks.json")))
BY_SHA = {}
for t in SPEC["tasks"]:
    for f in t["forms"]:
        BY_SHA[hashlib.sha256(f["prompt_user"].encode()).hexdigest()] = (t, f)


BRIDGE = {}
if os.environ.get("UNITS_BRIDGE_VIEWS"):
    for _l in open(os.environ["UNITS_BRIDGE_VIEWS"]):
        if _l.strip():
            _v = json.loads(_l)
            BRIDGE[hashlib.sha256(_v["prompt"].encode()).hexdigest()] = _v


def h(*parts):
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest(), 16)


def fmt(v, argtype):
    v = Fraction(v)
    if argtype == "integer" and v.denominator == 1:
        return int(v)
    return float(v) if v.denominator != 1 else int(v)


SLIPPY = {("t01", "sub"): 3, ("t03", "words"): 1, ("p02", "larger"): -1, ("c03", "sub"): 2, ("e03", "words"): 3, ("d01", "words"): -2, ("m03", "sub"): -3, ("y02", "larger"): 3}
THREE = {("t02", "sub"): 3, ("m02", "larger"): -3, ("g03", "sub"): 3}
TWO = {("t05", "larger"): 1, ("y04", "words"): 3}


def answer(profile, t, f):
    exp = Fraction(t["expected"])
    key = (t["id"], f["role"])
    mode, val = "call", exp
    name = t["tool"]
    args_extra = dict(t["must"])
    if profile == "mock-slippy":
        if key in SLIPPY:
            val = exp * Fraction(10) ** SLIPPY[key]
        elif key == ("u02", "sub"):
            val = exp + 1                    # other_value
        elif key == ("g01", "larger"):
            mode = "malformed"
        elif key == ("t07", "canonical"):
            mode = "nocall"
        elif key == ("p04", "words"):
            name = "terminal"                # wrong_tool
    elif profile == "mock-clean":
        if key in TWO:
            val = exp * Fraction(10) ** TWO[key]
        elif key == ("u02", "sub"):
            val = exp + 1
    elif profile == "mock-three":
        if key in THREE:
            val = exp * Fraction(10) ** THREE[key]
    elif profile == "mock-broken":
        if h(t["id"], f["role"]) % 10 < 3:
            mode = "nocall"
    elif profile == "mock-copycat":
        if f["role"] != "canonical":
            val = Fraction(f["num"])
    return mode, name, val, args_extra


def make_body(profile, payload):
    user = [m for m in payload["messages"] if m["role"] == "user"][-1]["content"]
    hit = BY_SHA.get(hashlib.sha256(user.encode()).hexdigest())
    bv = BRIDGE.get(hashlib.sha256(user.encode()).hexdigest())
    if bv and not hit:
        # bridge prompts: scripted FINAL line. every 7th item answers 10x too big (a scripted slip), the rest exact
        unit = {"L": "liters", "m": "meters", "km": "kilometers", "km/h": "km/h", "": ""}.get(bv["expected_unit"], bv["expected_unit"])
        val = Fraction(bv["expected_number"]) * (10 if int(hashlib.sha256(user.encode()).hexdigest(), 16) % 7 == 0 else 1)
        txt = "FINAL: %s %s" % (val, unit)
        return {"id": "mock", "object": "chat.completion", "model": payload.get("model"), "choices": [{"index": 0, "message": {"role": "assistant", "content": txt}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 80, "completion_tokens": 12, "total_tokens": 92}}, 200
    if not hit:
        return {"error": "unknown prompt"}, 400
    t, f = hit
    mode, name, val, extra = answer(profile, t, f)
    msg = {"role": "assistant", "content": ""}
    finish = "tool_calls"
    if mode == "nocall":
        msg["content"] = "I will do that for you right away."
        finish = "stop"
    else:
        ty = t["arg_type"]
        args = dict(extra)
        for k, p in SPEC["tools"][t["tool"]]["schema"]["parameters"]["properties"].items():
            if k in t["must"] or k == t["arg"]:
                continue
            if k in SPEC["tools"][t["tool"]]["schema"]["parameters"].get("required", []):
                args[k] = "x"
        args[t["arg"]] = fmt(val, ty)
        if mode == "malformed":
            raw = '{"%s": %s' % (t["arg"], fmt(val, ty))                        # truncated JSON
        elif profile == "mock-shapes":
            n = h(t["id"], f["role"]) % 3
            if n == 0:
                args[t["arg"]] = float(Fraction(val))                              # 90.0 for an integer argument
            elif n == 1:
                args[t["arg"]] = str(fmt(val, ty))                                 # numeric string
            raw = args if n == 2 else json.dumps(args)
            msg["reasoning_content"] = "converting units first"
        else:
            raw = json.dumps(args)
        msg["tool_calls"] = [{"id": "call_1", "type": "function", "function": {"name": name, "arguments": raw}}]
    return {"id": "mock", "object": "chat.completion", "model": payload.get("model"), "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": 400, "completion_tokens": 30, "total_tokens": 430}}, 200


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
        if self.path.startswith("/health"):
            self._send({"status": "ok"})
        elif self.path.startswith("/props"):
            self._send({"total_slots": 1, "build_info": "mock", "model_path": "mock", "default_generation_settings": {"n_ctx": 8192}})
        else:
            self._send({"error": "nope"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n).decode())
        if not self.path.startswith("/v1/chat/completions"):
            return self._send({"error": "nope"}, 404)
        body, code = make_body(payload.get("model", "mock-perfect"), payload)
        self._send(body, code)


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H)
    print("[mock-units] serving on", sys.argv[1], flush=True)
    srv.serve_forever()
