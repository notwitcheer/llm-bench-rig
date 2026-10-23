#!/bin/bash
# flint-samp-smoke.sh: CPU-only check that server-side sampling defaults reach Hermes goal-mode requests (2026-10-22, mercury).
# tiny probe model on CPU (-ngl 0) with the SAME build + sampling flags as 66z, a logging proxy that records the exact
# request bodies Hermes sends, and a throwaway Hermes profile with the run_agent.sh config shape. never touches :8090.
set -uo pipefail
D=/home/witcheer/flint-samp-smoke; mkdir -p $D; cd $D
BIN=/home/witcheer/llama.cpp-flint/build/bin/llama-server
M=/home/witcheer/models/tiny-probe/qwen2.5-0.5b-instruct-q4_k_m.gguf
HERMES=/home/witcheer/.local/bin/hermes
P=vxsmokesamp; H=/home/witcheer/.hermes/profiles/$P
SAMP="--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0"
echo "[smoke] $(date +%T) start"
$BIN -m $M -ngl 0 -t 4 -c 65536 --host 127.0.0.1 --port 8199 --parallel 1 --jinja $SAMP --alias smoke-model --reasoning-budget 32768 \
  --reasoning-budget-message "Thinking budget reached." > $D/server.log 2>&1 &
SPID=$!
cat > $D/proxy.py <<'PY'
import http.server, json, urllib.request, sys
LOG = "/home/witcheer/flint-samp-smoke/requests.jsonl"
class H(http.server.BaseHTTPRequestHandler):
    def _fwd(self, body=None):
        req = urllib.request.Request("http://127.0.0.1:8199" + self.path, data=body, method=self.command)
        for k, v in self.headers.items():
            if k.lower() not in ("host", "content-length", "accept-encoding"):
                req.add_header(k, v)
        try:
            r = urllib.request.urlopen(req, timeout=300)
            code, hdrs, data = r.status, r.headers.items(), r.read()
        except urllib.error.HTTPError as e:
            code, hdrs, data = e.code, e.headers.items(), e.read()
        self.send_response(code)
        for k, v in hdrs:
            if k.lower() not in ("transfer-encoding", "content-length", "connection"):
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_GET(self): self._fwd()
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); body = self.rfile.read(n)
        try:
            j = json.loads(body)
            keep = {k: v for k, v in j.items() if k not in ("messages", "tools")}
            keep["n_messages"] = len(j.get("messages", [])); keep["n_tools"] = len(j.get("tools", []) or [])
            keep["assistant_msgs_with_reasoning_content"] = sum(1 for m in j.get("messages", []) if m.get("role") == "assistant" and m.get("reasoning_content"))
            keep["path"] = self.path
            open(LOG, "a").write(json.dumps(keep) + "\n")
        except Exception as e:
            open(LOG, "a").write(json.dumps({"path": self.path, "parse_error": str(e)}) + "\n")
        self._fwd(body)
    def log_message(self, *a): pass
http.server.ThreadingHTTPServer(("127.0.0.1", 8198), H).serve_forever()
PY
rm -f $D/requests.jsonl
python3 $D/proxy.py > $D/proxy.log 2>&1 &
PPID2=$!
trap 'kill $SPID $PPID2 2>/dev/null' EXIT
for i in $(seq 1 60); do curl -s 127.0.0.1:8199/health | grep -qi ok && break; sleep 1; done
echo "[smoke] server: $(curl -s 127.0.0.1:8199/health)"
echo "[smoke] /props default_generation_settings:"
curl -s 127.0.0.1:8199/props | python3 -c "import json,sys;d=json.load(sys.stdin);g=d.get('default_generation_settings',{});p=g.get('params',g);print({k:p.get(k) for k in ('temperature','top_k','top_p','min_p','n_predict','reasoning_budget')})"
[ -d "$H" ] || $HERMES profile create "$P" --no-alias --description "sampling smoke (mercury, throwaway)" > $D/profile.log 2>&1
# same shape as run_agent.sh, base_url -> logging proxy; HERMES_EXTRA (optional) appended for the override test
cat > "$H/config.yaml" <<EOF
model:
  default: smoke-model
  provider: custom
  base_url: http://127.0.0.1:8198/v1
  api_key: local-key
  context_length: 65536
toolsets:
  - hermes-cli
agent:
  max_turns: 3
  clarify_timeout: 20
approvals:
  mode: "off"
memory:
  memory_enabled: false
  user_profile_enabled: false
display:
  interface: cli
telemetry:
  shared_metrics:
    enabled: false
    send: false
    offer_version: 2
EOF
cd $D && timeout 240 $HERMES -p $P chat -q "Reply with the single word ok." > $D/hermes.out 2>&1; echo "[smoke] hermes rc=$?"
tail -3 $D/hermes.out
echo "[smoke] request bodies seen by the proxy (messages/tools elided):"
cat $D/requests.jsonl
echo "[smoke] GET /slots params after the request:"
curl -s 127.0.0.1:8199/slots | python3 -c "import json,sys;d=json.load(sys.stdin);s=d[0] if isinstance(d,list) else d;p=s.get('params',s);print({k:p.get(k) for k in ('temperature','top_k','top_p','min_p','n_predict')})" 2>&1 | head -3
echo "[smoke] $(date +%T) end"
