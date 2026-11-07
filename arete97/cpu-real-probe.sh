#!/bin/bash
# CPU-only (CUDA hidden) probe of BOTH real GGUFs with the exact server flags item 97-arete-msm passes (port 8199, never :8090, no card).
# checks: flags accepted, /props sampling = temp 1.0 / top-p 0.95 / top-k 20 / min-p 0, no MTP draft context in the log (parity: no drafting for either),
# the chat template renders the same shape, one tiny chat answers. output: ~/voxel-stage/arete97/cpu/real-<tag>/*
set -u
O=$HOME/voxel-stage/arete97/cpu; mkdir -p $O
BIN=$HOME/llama.cpp-donald/build/bin/llama-server
RB_MSG="Thinking budget reached. Stop planning and write the files now with your tools; refine in later turns."
for pair in "qwen:$HOME/models/qwen38-27b/Qwen3.8-27B-Q6_K.gguf" "arete:$HOME/models/arete-27b/NousResearch_Arete-27B-Q6_K.gguf"; do
  tag=${pair%%:*}; M=${pair#*:}; D=$O/real-$tag; rm -rf $D; mkdir -p $D
  echo "[$tag] $(date +%T) loading on CPU"
  CUDA_VISIBLE_DEVICES="" nice -n 19 $BIN --model "$M" --host 127.0.0.1 --port 8199 --n-gpu-layers 0 -t 6 --ctx-size 4096 --parallel 1 --flash-attn on --jinja --no-warmup \
    --cache-type-k q8_0 --cache-type-v q8_0 --alias "Qwen3.8-27B-duel-Q6_K.gguf" --metrics --reasoning-budget 16384 --reasoning-budget-message "$RB_MSG" \
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 > $D/server.log 2>&1 &
  SP=$!
  for i in $(seq 1 300); do h=$(curl -s -m 3 127.0.0.1:8199/health); [[ "${h,,}" == *ok* ]] && break; kill -0 $SP 2>/dev/null || break; sleep 2; done
  echo "[$tag] $(date +%T) health: $h"
  curl -s 127.0.0.1:8199/props > $D/props.json
  cat > $D/apply.json <<'JSON'
{"messages":[{"role":"system","content":"You are an agent."},{"role":"user","content":"make a file"},
 {"role":"assistant","content":"","reasoning_content":"PLAN-ONE","tool_calls":[{"id":"c1","type":"function","function":{"name":"write_file","arguments":"{\"path\":\"a.txt\",\"content\":\"hi\"}"}}]},
 {"role":"tool","tool_call_id":"c1","content":"ok"},
 {"role":"assistant","content":"done","reasoning_content":"PLAN-TWO"},
 {"role":"user","content":"now edit it"}],
 "tools":[{"type":"function","function":{"name":"write_file","description":"write","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}],
 "reasoning_effort":"medium"}
JSON
  curl -s -X POST 127.0.0.1:8199/apply-template -H 'Content-Type: application/json' -d @$D/apply.json > $D/apply.json.out
  # one tiny chat, shaped like Hermes (reasoning_effort medium, no sampling fields, no max_tokens in Hermes; 24 here so a CPU run ends)
  curl -s -m 600 -X POST 127.0.0.1:8199/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"Qwen3.8-27B-duel-Q6_K.gguf","messages":[{"role":"user","content":"Say hi."}],"max_tokens":24,"reasoning_effort":"medium"}' > $D/chat.json
  curl -s -m 5 127.0.0.1:8199/slots > $D/slots.json
  kill $SP 2>/dev/null; wait $SP 2>/dev/null
  python3 - $D <<'PY'
import json, sys, hashlib, re
d = sys.argv[1]
p = json.load(open(d + "/props.json")); g = p.get("default_generation_settings", {}); pr = g.get("params", g)
log = open(d + "/server.log", errors="replace").read()
ap = json.load(open(d + "/apply.json.out")); prompt = ap.get("prompt", "")
ch = json.load(open(d + "/chat.json")); m = (ch.get("choices") or [{}])[0].get("message", {})
out = {
  "sampling": {k: pr.get(k) for k in ("temperature", "top_k", "top_p", "min_p")},
  "n_ctx": g.get("n_ctx"), "total_slots": p.get("total_slots"),
  "mtp_draft_context_in_log": bool(re.search(r"creating MTP draft context|draft-mtp|speculative", log)),
  "spec_lines": [l[:160] for l in log.splitlines() if re.search(r"spec|draft|mtp|nextn", l, re.I)][:6],
  "template_md5": hashlib.md5((p.get("chat_template") or "").encode()).hexdigest()[:12],
  "prompt_len": len(prompt), "prompt_keeps_PLAN_ONE": "PLAN-ONE" in prompt, "prompt_keeps_PLAN_TWO": "PLAN-TWO" in prompt,
  "prompt_tail": prompt[-120:],
  "chat_finish": (ch.get("choices") or [{}])[0].get("finish_reason"), "chat_usage": ch.get("usage"),
  "chat_content_head": (m.get("content") or "")[:80], "chat_reasoning_head": (m.get("reasoning_content") or "")[:80],
}
json.dump(out, open(d + "/summary.json", "w"), indent=1); print(json.dumps(out)[:1500])
PY
  sleep 3
done
echo "PROBE-DONE $(date +%T)"
