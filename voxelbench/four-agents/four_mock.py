#!/usr/bin/env python3
"""four_mock.py: llama-server-shaped mock for the item-80 PASS path (CPU only, no model, test aid only, never part of a run).

Speaks what the four-agents harness needs from llama-server: /health, /v1/models, /v1/chat/completions (stream), /metrics (the llamacpp:* counters
the sampler reads) and /slots (4 slots, is_processing, n_decoded). It plays a fixed script per conversation, keyed on the number of tool results
already in it, and tells the four agents apart by the X-Race-Agent header the proxy adds (a b c d): each gets its own pace and its own small
animated canvas page, so the race has a spread and the four result pages differ.

  turn 0  reasoning deltas (thinking, shown live by Hermes), then write_file(<cwd>/index.html, a canvas page) streamed as tool-call argument chunks
  turn 1  terminal(ls -la)
  turn 2  terminal(node --check on the page's script)   [plain tool call]
  turn 3+ final text

usage: MOCK_SCALE=1.0 four_mock.py PORT      (MOCK_SCALE < 1 shortens every stream; MOCK_FAIL_AGENT=c makes agent c answer one 500 then recover)
"""
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SCALE = float(os.environ.get("MOCK_SCALE", "1.0"))
PACE = {"a": 1.0, "b": 0.82, "c": 1.18, "d": 0.64}      # relative tokens/s per agent (the spread the race should show)
BASE_TPS = 55.0                                         # chunks/s at pace 1.0 when alone
HUES = {"a": 200, "b": 28, "c": 130, "d": 300}
LOCK = threading.Lock()
STATE = {"predicted": 0, "prompt": 0, "pred_seconds": 0.0, "busy": {}, "decoded": {}, "fail_done": False, "requests": 0}


def page(agent):
    h = HUES.get(agent, 200)
    return """<!doctype html><html><head><meta charset="utf-8"><title>lighthouse %(a)s</title><style>html,body{margin:0;background:#0b1020}canvas{display:block;width:100vw;height:100vh}</style></head>
<body><canvas id="c" width="960" height="540"></canvas><script>
const cv=document.getElementById('c'),x=cv.getContext('2d');let t=0;
function iso(i,j,k){return [480+(i-j)*18,330+(i+j)*9-k*16];}
function cube(i,j,k,c){const [px,py]=iso(i,j,k);x.fillStyle=c[0];x.beginPath();x.moveTo(px,py-16);x.lineTo(px+18,py-7);x.lineTo(px,py+2);x.lineTo(px-18,py-7);x.fill();
x.fillStyle=c[1];x.beginPath();x.moveTo(px-18,py-7);x.lineTo(px,py+2);x.lineTo(px,py+18);x.lineTo(px-18,py+9);x.fill();
x.fillStyle=c[2];x.beginPath();x.moveTo(px+18,py-7);x.lineTo(px,py+2);x.lineTo(px,py+18);x.lineTo(px+18,py+9);x.fill();}
function frame(){t+=1/60;const g=x.createLinearGradient(0,0,0,540);g.addColorStop(0,'hsl(%(h)d,55%%,12%%)');g.addColorStop(1,'hsl(%(h)d,60%%,30%%)');x.fillStyle=g;x.fillRect(0,0,960,540);
for(let s=0;s<60;s++){x.fillStyle='rgba(255,255,255,'+(0.3+0.5*Math.sin(t*2+s))+')';x.fillRect((s*97)%%960,(s*53)%%200,2,2);}
for(let i=-9;i<=9;i++)for(let j=-9;j<=9;j++){const d=Math.hypot(i,j);const w=Math.sin(t*2+i*.6+j*.4)*.35;if(d>6.2)cube(i,j,-1+w,['hsl(%(h)d,70%%,45%%)','hsl(%(h)d,70%%,32%%)','hsl(%(h)d,70%%,38%%)']);else if(d>5)cube(i,j,0,['#e3d18a','#b9a565','#cdb978']);else cube(i,j,0,['#4f9d4a','#357a34','#3f8b3d']);}
for(let k=1;k<=9;k++){const s=k%%2?['#f4f4f4','#cfcfcf','#e0e0e0']:['#c8372d','#9e2a22','#b3312a'];cube(0,0,k,s);}
const a=t*1.6,bx=Math.cos(a)*400,by=Math.sin(a)*60;x.fillStyle='rgba(255,236,140,.35)';x.beginPath();x.moveTo(480,170);x.lineTo(480+bx*1.2-40,170+by);x.lineTo(480+bx*1.2+40,170+by+30);x.fill();
cube(0,0,10,['#ffe27a','#d9b84f','#f0cc62']);requestAnimationFrame(frame);}
frame();
</script></body></html>
""" % {"a": agent, "h": h}


def script(body, agent):
    msgs = body.get("messages", [])
    n = sum(1 for m in msgs if m.get("role") == "tool")
    if not body.get("tools"):
        return {"text": "mock title"}
    cwd = "/tmp"
    for m in msgs:
        c = m.get("content")
        if isinstance(c, str):
            f = re.search(r"exactly (\S+?)/index\.html", c)
            if f:
                cwd = f.group(1)
    if n == 0:
        return {"tool": "write_file", "args": {"path": cwd + "/index.html", "content": page(agent)}, "think": 260}
    if n == 1:
        return {"tool": "terminal", "args": {"command": "ls -la"}, "think": 40}
    if n == 2:
        return {"tool": "terminal", "args": {"command": "python3 -c \"import re,sys;s=open('index.html').read();print(len(s),'bytes',s.count('requestAnimationFrame'),'raf')\""}, "think": 40}
    return {"text": "the lighthouse island is written to index.html: iso voxels, a sweeping beam, waves and stars. checked the file exists and animates.", "think": 30}


def chunks_for(s):
    """list of (kind, text): reasoning deltas first, then the answer or the tool arguments cut into small pieces"""
    out = []
    think = int(s.get("think", 0) * SCALE)
    for i in range(think):
        out.append(("r", "hmm " if i % 7 else "so the beam. "))
    if "tool" in s:
        a = json.dumps(s["args"])
        step = max(4, int(len(a) / max(60, int(220 * SCALE))))
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
            return self._json(200, {"object": "list", "data": [{"id": "mock-qwen", "object": "model"}]})
        if p == "/props":
            return self._json(200, {"default_generation_settings": {"params": {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.05}, "n_ctx": 196608}, "total_slots": 4})
        if p == "/metrics":
            with LOCK:
                busy = sum(1 for v in STATE["busy"].values() if v)
                txt = ("# TYPE llamacpp:tokens_predicted_total counter\nllamacpp:tokens_predicted_total %d\n"
                       "llamacpp:prompt_tokens_total %d\nllamacpp:tokens_predicted_seconds_total %.3f\n"
                       "llamacpp:requests_processing %d\nllamacpp:requests_deferred 0\n") % (STATE["predicted"], STATE["prompt"], STATE["pred_seconds"], busy)
            return self._json(200, txt.encode(), "text/plain")
        if p == "/slots":
            with LOCK:
                rows = []
                for i, ag in enumerate("abcd"):
                    rows.append({"id": i, "n_ctx": 49152, "is_processing": bool(STATE["busy"].get(ag)),
                                 "id_task": STATE["requests"] if STATE["busy"].get(ag) else 0,
                                 "next_token": [{"n_decoded": STATE["decoded"].get(ag, 0)}]})
            return self._json(200, rows)
        return self._json(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        agent = (self.headers.get("X-Race-Agent") or "a")[:1]
        if os.environ.get("MOCK_FAIL_AGENT") == agent and not STATE["fail_done"] and body.get("tools"):
            STATE["fail_done"] = True
            return self._json(500, {"error": {"message": "mock one-off 500", "type": "server_error"}})
        s = script(body, agent)
        ch = chunks_for(s)
        cid = "chatcmpl-mock%d" % int(time.time() * 1000)
        if not body.get("stream"):             # the speed probe: a plain JSON answer with llama-server style timings
            n_tok = min(len(ch), int(body.get("max_tokens") or 320))
            with LOCK:
                STATE["busy"][agent] = True
                busy = max(1, sum(1 for v in STATE["busy"].values() if v))
            rate = BASE_TPS * PACE.get(agent, 1.0) / (1 + 0.25 * (busy - 1))
            time.sleep(n_tok / rate)
            with LOCK:
                STATE["busy"][agent] = False
                STATE["predicted"] += n_tok
            return self._json(200, {"id": cid, "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": "mock"}, "finish_reason": "length"}],
                                    "usage": {"prompt_tokens": 40, "completion_tokens": n_tok}, "timings": {"prompt_n": 40, "predicted_n": n_tok, "predicted_per_second": round(rate, 2)}})
        with LOCK:
            STATE["busy"][agent] = True
            STATE["decoded"][agent] = 0
            STATE["requests"] += 1
            STATE["prompt"] += 3000
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def send(obj):
            raw = ("data: " + json.dumps(obj) + "\n\n").encode()
            self.wfile.write(("%x\r\n" % len(raw)).encode() + raw + b"\r\n")
            self.wfile.flush()
        tcid = "call_%d" % int(time.time() * 1000)
        first_t = True
        t0 = time.time()
        # shared-card slowdown: the more agents decode at once the slower each one is (4 busy = about 0.55x each), like a real parallel server
        try:
            send({"id": cid, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})
            for kind, txt in ch:
                with LOCK:
                    busy = max(1, sum(1 for v in STATE["busy"].values() if v))
                rate = BASE_TPS * PACE.get(agent, 1.0) * (1.0 / (1 + 0.25 * (busy - 1))) / max(SCALE, 0.05) ** 0   # chunks per second
                time.sleep(1.0 / rate)
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
                    STATE["decoded"][agent] = STATE["decoded"].get(agent, 0) + 1
            el = max(time.time() - t0, 1e-3)
            fin = "tool_calls" if "tool" in s else "stop"
            send({"id": cid, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": fin}],
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
                STATE["busy"][agent] = False


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
