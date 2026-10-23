#!/bin/bash
# lfm-cpuload.sh: CPU-only (-ngl 0) load test of an LFM2.5-24B-A2B gguf on a given llama-server build.
# usage: bash lfm-cpuload.sh <server_bin> <gguf> <tag>   (port 8199, no Donald impact)
BIN="$1"; GGUF="$2"; TAG="${3:-nl}"
LOG=/home/witcheer/lfm-cpuload-$TAG.log
echo "== $(date -Is) $TAG: $($BIN --version 2>&1 | grep version)" 
"$BIN" -m "$GGUF" -ngl 0 -c 4096 --jinja --port 8199 --host 127.0.0.1 > "$LOG" 2>&1 &
PID=$!
ok=0
for i in $(seq 1 120); do
  if ! kill -0 $PID 2>/dev/null; then echo "server DIED during load"; break; fi
  if curl -s localhost:8199/health | grep -qi ok; then ok=1; echo "health ok after $((i*2)) s"; break; fi
  sleep 2
done
grep -aiE "pre-tokenizer|pre_tokenizer|tokenizer.ggml.pre|error|GGML_ASSERT|unknown|failed|arch|n_vocab|vocab type|n_expert|model type|file type|CUDA" "$LOG" | head -40
if [ $ok = 1 ]; then
  echo "-- tokenize (latin + non-latin)"
  curl -s localhost:8199/tokenize -H 'Content-Type: application/json' -d '{"content":"Hello world, 2+2=4. Привет мир. 你好世界。 こんにちは"}'; echo
  echo "-- chat (always-thinking, 96 tok)"
  curl -s localhost:8199/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"x","messages":[{"role":"user","content":"What is 17*3? Answer with the number."}],"max_tokens":256,"temperature":0}' \
    | python3 -c 'import json,sys; j=json.load(sys.stdin); m=j["choices"][0]["message"]; print("finish:",j["choices"][0].get("finish_reason"),"usage:",j.get("usage")); print("REASONING:",(m.get("reasoning_content") or "")[:400].replace("\n"," | ")); print("CONTENT:",(m.get("content") or "")[:300].replace("\n"," | ")); print("timings:",{k:v for k,v in (j.get("timings") or {}).items() if "per_second" in k})'
fi
kill $PID 2>/dev/null; sleep 2; kill -9 $PID 2>/dev/null
echo "== done $TAG"
