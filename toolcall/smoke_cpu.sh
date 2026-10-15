#!/bin/bash
# CPU-only plumbing smoke for the toolcall-v1 driver: tiny Qwen2.5-0.5B on the mainline binary, -ngl 0,
# side port 8199. Checks /props, /tokenize, /completion, parse/grade and the multi-turn loop run end to end.
# Tokenizer parity is EXPECTED to fail here (different vocab); scores are meaningless.
set -uo pipefail
T=$HOME/toolcall-v1
M=$HOME/models/tiny-probe/qwen2.5-0.5b-instruct-q4_k_m.gguf
[ -f "$M" ] || { echo "tiny model missing"; exit 2; }
BIN=$HOME/llama.cpp-nl/build/bin/llama-server
$BIN -m "$M" -ngl 0 -c 16384 --port 8199 --host 127.0.0.1 > $T/smoke-server.log 2>&1 &
SP=$!
trap 'kill $SP 2>/dev/null; wait $SP 2>/dev/null' EXIT
for i in $(seq 1 60); do curl -sf http://127.0.0.1:8199/health | grep -qi ok && break; sleep 1; done
cd $T/code
../.venv/bin/python driver.py --server http://127.0.0.1:8199 --label smoke-tiny --ids $T/artifacts/probe_ids.json \
  --manifest $T/frozen/toolcall-v1/manifest.json --tokenizer $T/qwen38-tokenizer \
  --prompts $T/artifacts/prompts/prompts_toolcall-v1.jsonl.gz --out $T/smoke-out --think on --max-tokens 192 --ctx 16384 \
  --budget-s 420 --server-bin "$BIN (CPU smoke)" 2>&1 | grep -v -i warn
echo "smoke rc=$?"
