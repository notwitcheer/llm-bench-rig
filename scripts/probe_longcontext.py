"""One-off diagnostic: show what the model emits for a single long-context item.
Run on capsule with llama-server up on :8090. Not part of the suite."""
import json, sys
sys.path.insert(0, ".")
from lib.evals.base import LLMClient
from lib.agentic.base import load_system_prompt, extract_code, build_tool_doc
from lib.agentic.longcontext import build_haystack
from lib.agentic.sandbox import run_code_action
from lib.agentic.codeact import check_result
from lib.config import get

item = json.loads(open("data/agentic/longcontext.jsonl").read().splitlines()[0])
sp = load_system_prompt(get("agentic.system_prompt"))
hay = build_haystack(item["needle"], target_tokens=16384)
user = (f"{build_tool_doc()}\n\nContext (memory + tool logs):\n{hay}\n\n"
        f"Task: {item['question']}\nWrite a ```python``` block; assign `result`.")
print("NEEDLE VALUE:", item["needle_value"])
print("QUESTION:", item["question"])
print("SYSTEM PROMPT len:", len(sp))
with LLMClient("http://127.0.0.1:8090/v1", "m", think=False, timeout=300) as c:
    resp = c.chat([{"role": "system", "content": sp},
                   {"role": "user", "content": user}], max_tokens=512)
print("\n=== RAW RESPONSE (first 900) ===\n", resp[:900])
code = extract_code(resp)
print("\n=== EXTRACTED CODE (first 400) ===\n", code[:400])
out = run_code_action(code)
print("\n=== SANDBOX === ok=", out.ok, "result=", repr(out.result)[:200], "err=", out.error[:200])
print("CHECK PASS:", check_result(out.result, item["check"]))
