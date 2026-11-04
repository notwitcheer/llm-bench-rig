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

80b additions (all off unless the env is set, so the item-80 stub test behaves as before):
  MOCK_TOKMULT=10   report token counts as if every ~3.2 bytes of conversation (system prompt excluded: a constant 5,000) were 10 tokens, so a small mock conversation reaches the context
                    sizes of a real thinking-heavy session (first turn ~24k tokens like the real 24-30k) and Hermes' compaction trigger sees real-sized numbers in usage.prompt_tokens
  MOCK_TPS=900 MOCK_FIRST_THINK=16000   stream faster than a real card, and make the first turn think 16,000 chunks (the real first turn: 16k thinking + a write_file call) so the first-turn context is ~23k
  MOCK_TURNS=40 MOCK_CAT=3   length of the scripted task: write_file, then (MOCK_TURNS-1) x `cat index.html` x MOCK_CAT (~3.8k tokens of real tool output per step), then the final answer. Counted per agent
                    on the server side, so a compaction (which shrinks the message list) does not restart the script
  MOCK_POOL=163840  simulate llama-server's pooled KV: every agent's context (prompt + tokens decoded so far, or the end of its last request while idle) counts against
                    one pool; a request whose prompt does not fit gets HTTP 500 "Context size has been exceeded" like the real server did at 14:40 on 2026-11-03
  llama-server style `timings` (prompt_n = tokens evaluated, cache_n = reused prefix, predicted_n) are added to the final chunk so the proxy's pool guard sees what it sees on a real server
  a request without tools that mentions "summar" gets a ~1,200 character structured summary back (Hermes' compaction call), anything else keeps "mock title"
"""
import json
import os
import random
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SCALE = float(os.environ.get("MOCK_SCALE", "1.0"))
VOCAB = ("light tower beam island water wave sand grass stone voxel cube iso canvas frame shade lamp cloud star moon dusk glow rotate sweep angle depth palette color hue "
         "draw loop render sin cos offset tile grid height base step plan check fix write read test order face top left right edge gradient sky sea rock path door window "
         "stripe red white amber blue green dark warm cold soft hard fast slow bright dim smooth pixel row column layer block shadow reflect ripple foam shore cliff mast flag "
         "rope boat dock cabin roof stair rail lens prism ray cone arc curve circle square line point vector matrix scale shift mix blend alpha clip fill stroke path begin end").split()
PACE = {"a": 1.0, "b": 0.82, "c": 1.18, "d": 0.64}      # relative tokens/s per agent (the spread the race should show)
BASE_TPS = float(os.environ.get("MOCK_TPS", "55.0"))                                        # chunks/s at pace 1.0 when alone
HUES = {"a": 200, "b": 28, "c": 130, "d": 300}
LOCK = threading.Lock()
STATE = {"predicted": 0, "prompt": 0, "pred_seconds": 0.0, "busy": {}, "decoded": {}, "fail_done": False, "requests": 0,
         "ctx": {}, "turn": {}, "max_ctx": 0, "rejected": 0, "summaries": 0}
TOKMULT = float(os.environ.get("MOCK_TOKMULT", "0") or 0)       # 0 = item-80 behaviour (constant 3000-token prompt)
TURNS = int(os.environ.get("MOCK_TURNS", "0") or 0)             # 0 = item-80 script (write, ls, check, answer)
CATS = int(os.environ.get("MOCK_CAT", "3") or 3)                # grow mode: each step prints the page this many times (a 4 KB page x 3 = ~3.8k tokens of tool output per step)
POOL = int(os.environ.get("MOCK_POOL", "0") or 0)               # 0 = no pooled-KV simulation
SUMMARY = ("## Goal\nBuild one animated isometric voxel lighthouse island as a single canvas page at the path in the task.\n\n## Progress\n- wrote index.html with write_file (canvas, iso cubes, "
           "sweeping beam, waves, stars)\n- re-read the file to check it; the script block parses\n\n## Key decisions\n- plain canvas, no libraries, no network\n- single file, about 1.5 KB of script logic\n\n"
           "## Next steps\n- confirm the page animates (requestAnimationFrame present), then finish with a short answer\n\n## Critical context\n- the file exists on disk; nothing else is needed\n") * 1


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
        txt = json.dumps(msgs).lower()
        if TOKMULT and "summar" in txt:
            with LOCK:
                STATE["summaries"] += 1
            return {"text": SUMMARY, "think": 0}
        return {"text": "mock title"}
    cwd = "/tmp"
    for m in msgs:
        c = m.get("content")
        if isinstance(c, str):
            f = re.search(r"exactly (\S+?)/index\.html", c)
            if f:
                cwd = f.group(1)
    if TURNS:
        with LOCK:
            k = STATE["turn"].get(agent, 0)
            STATE["turn"][agent] = k + 1
        if k == 0:
            return {"tool": "write_file", "args": {"path": cwd + "/index.html", "content": page(agent)}, "think": int(os.environ.get("MOCK_FIRST_THINK", "260"))}
        if k < TURNS:
            return {"tool": "terminal", "args": {"command": "cat " + " ".join(["index.html"] * CATS)}, "think": 40}
        return {"text": "the lighthouse island is written to index.html and re-read %d times: iso voxels, a sweeping beam, waves and stars." % k, "think": 30}
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
    rnd = random.Random(think * 7919 + len(s.get("text", "")) + len(json.dumps(s.get("args", ""))))
    for i in range(think):
        # varied words: Hermes stops a stream whose reasoning repeats itself (it did on a fixed "hmm hmm" text); a plain vocabulary draw reads as thinking to the detector
        out.append(("r", (rnd.choice(VOCAB) + (". " if rnd.random() < 0.08 else " ")) if think > 600 else ("hmm " if i % 7 else "so the beam. ")))
    if "tool" in s:
        a = json.dumps(s["args"])
        step = max(4, int(len(a) / max(60, int(220 * SCALE))))
        out += [("t", a[i:i + step]) for i in range(0, len(a), step)]
    else:
        t = s["text"]
        out += [("c", t[i:i + 5]) for i in range(0, len(t), 5)]
    return out


def prompt_tokens(body):
    """tokens the mock claims this request's prompt has: 3000 as in item 80, or (80b) 5000 for system + tools + 1 token per 3.2 bytes of conversation x MOCK_TOKMULT"""
    if not TOKMULT:
        return 3000
    conv = [x for x in body.get("messages", []) if x.get("role") != "system"]     # Hermes' own system prompt is a constant ~5k tokens, counted in the 5000
    return 5000 + int(len(json.dumps(conv)) / 3.2 * TOKMULT)


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
        if p == "/_mock":
            with LOCK:
                return self._json(200, {k: STATE[k] for k in ("ctx", "turn", "max_ctx", "rejected", "summaries", "requests")})
        if p == "/props":
            return self._json(200, {"default_generation_settings": {"params": {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.05}, "n_ctx": 196608}, "total_slots": 4})
        if p == "/metrics":
            with LOCK:
                busy = sum(1 for v in STATE["busy"].values() if v)
                txt = ("# TYPE llamacpp:tokens_predicted_total counter\nllamacpp:tokens_predicted_total %d\n"
                       "llamacpp:prompt_tokens_total %d\nllamacpp:tokens_predicted_seconds_total %.3f\n"
                       "llamacpp:requests_processing %d\nllamacpp:requests_deferred 0\nllamacpp:n_tokens_max %d\n") % (STATE["predicted"], STATE["prompt"], STATE["pred_seconds"], busy, STATE["max_ctx"])
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
        ptok = prompt_tokens(body)
        mult = TOKMULT or 1.0
        with LOCK:
            prev = STATE["ctx"].get(agent, 0)
            others = sum(v for k, v in STATE["ctx"].items() if k != agent)
            reject = bool(POOL) and others + ptok > POOL
            if reject:
                STATE["rejected"] += 1
                STATE["ctx"][agent] = 0
            else:
                STATE["busy"][agent] = True
                STATE["decoded"][agent] = 0
                STATE["requests"] += 1
                cache_n = min(prev, ptok) if TOKMULT else 0
                STATE["prompt"] += ptok - cache_n
                STATE["ctx"][agent] = ptok
                STATE["max_ctx"] = max(STATE["max_ctx"], ptok)
        if reject:
            return self._json(500, {"error": {"code": 500, "message": "Context size has been exceeded.", "type": "server_error"}})
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
                    STATE["ctx"][agent] = ptok + int(STATE["decoded"][agent] * mult)
                    STATE["max_ctx"] = max(STATE["max_ctx"], STATE["ctx"][agent])
                    over = bool(POOL) and sum(STATE["ctx"].values()) > POOL and STATE["decoded"][agent] % 16 == 0
                    if over:
                        STATE["rejected"] += 1
                        STATE["ctx"][agent] = 0
                if over:       # the pool filled while this stream was decoding: the server drops the slot and reports it in the stream
                    send({"error": {"code": 500, "message": "Context size has been exceeded.", "type": "server_error"}})
                    raise BrokenPipeError("mock pool exhausted mid-decode")
            el = max(time.time() - t0, 1e-3)
            fin = "tool_calls" if "tool" in s else "stop"
            send({"id": cid, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": fin}],
                  "usage": {"prompt_tokens": ptok, "completion_tokens": int(len(ch) * mult), "total_tokens": ptok + int(len(ch) * mult)},
                  "timings": {"prompt_n": ptok - cache_n, "cache_n": cache_n, "predicted_n": int(len(ch) * mult), "predicted_per_second": round(len(ch) / el, 2)}})
            raw = b"data: [DONE]\n\n"
            self.wfile.write(("%x\r\n" % len(raw)).encode() + raw + b"\r\n0\r\n\r\n")
            self.wfile.flush()
            with LOCK:
                STATE["pred_seconds"] += el
                STATE["ctx"][agent] = ptok + int(len(ch) * mult)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with LOCK:
                STATE["busy"][agent] = False


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
