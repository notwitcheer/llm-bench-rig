#!/bin/bash
# lfm-tokcheck.sh: compare llama.cpp /tokenize (CPU server on :8199) against the HF tokenizer of LiquidAI/LFM2.5-24B-A2B
BIN=/home/witcheer/llama.cpp-nl/build/bin/llama-server
GGUF=/home/witcheer/models/lfm25-24b/LFM2.5-24B-A2B-UD-Q4_K_XL.gguf
D=/home/witcheer/models/lfm25-24b/hf-tok
/home/witcheer/unsloth-env/bin/hf download LiquidAI/LFM2.5-24B-A2B tokenizer.json tokenizer_config.json --local-dir $D > /dev/null 2>&1 || echo "tok download failed"
"$BIN" -m "$GGUF" -ngl 0 -c 2048 --port 8199 --host 127.0.0.1 -v > /home/witcheer/lfm-tokcheck-server.log 2>&1 &
PID=$!
for i in $(seq 1 90); do curl -s localhost:8199/health | grep -qi ok && break; sleep 2; done
grep -aE "tokenizer.ggml.pre|pre-tokenizer|pre_type|vocab type|n_vocab|n_merges|print_info: (arch|vocab|model type|file type|n_expert)" /home/witcheer/lfm-tokcheck-server.log | head -20
/home/witcheer/unsloth-env/bin/python - <<'EOF'
import json, urllib.request
from tokenizers import Tokenizer
tk = Tokenizer.from_file("/home/witcheer/models/lfm25-24b/hf-tok/tokenizer.json")
texts = [
 "Hello world, 2+2=4. The quick brown fox jumps over 13 lazy dogs!",
 "Привет мир. 你好世界。 こんにちは、元気ですか？ مرحبا بالعالم. नमस्ते दुनिया. สวัสดีชาวโลก",
 "def f(x):\n    return [i**2 for i in range(x)]  # squares\n\n\tif x  >=  10:\n",
 "Let $f(x)=\\frac{3x^2-5}{x+1}$. Find $\\lim_{x\\to\\infty} f(x)/x$ and put it in \\boxed{}.",
 "Write a 300+ word essay in ALL CAPITAL LETTERS, wrapped in double quotes, with 3 sections ***.",
]
bad = 0
for t in texts:
    hf = tk.encode(t, add_special_tokens=False).ids
    r = urllib.request.Request("http://127.0.0.1:8199/tokenize", data=json.dumps({"content": t, "add_special": False}).encode(), headers={"Content-Type": "application/json"})
    lc = json.loads(urllib.request.urlopen(r).read())["tokens"]
    same = hf == lc
    bad += not same
    print(f"{'MATCH' if same else 'DIFF '} hf={len(hf)} llama={len(lc)} max_id={max(hf)} :: {t[:40]!r}")
    if not same:
        print("   hf   ", hf[:30]); print("   llama", lc[:30])
print("vocab size (hf):", tk.get_vocab_size())
print("TOKCHECK", "PASS" if bad == 0 else f"FAIL {bad}/{len(texts)}")
EOF
kill $PID; sleep 2; kill -9 $PID 2>/dev/null; true
