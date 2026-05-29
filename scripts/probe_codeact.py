"""Diagnostic: show raw model output for codeact items under the configured system
prompt. Run on capsule with llama-server up on :8090. Not part of the suite."""
import json, sys
sys.path.insert(0, ".")
from lib.evals.base import LLMClient
from lib.agentic.base import load_system_prompt, extract_code
from lib.agentic.codeact import _build_messages, check_result
from lib.agentic.sandbox import run_code_action
from lib.config import get

items = [json.loads(l) for l in open("data/agentic/codeact.jsonl").read().splitlines() if l.strip()][:3]
sp = load_system_prompt(get("agentic.system_prompt"))
print("SYSTEM PROMPT len(chars):", len(sp))
with LLMClient("http://127.0.0.1:8090/v1", "m", think=True, timeout=300) as c:
    for item in items:
        resp = c.chat(_build_messages(sp, item["task"]), max_tokens=1536)
        code = extract_code(resp)
        out = run_code_action(code)
        ok = bool(out.ok and check_result(out.result, item["check"]))
        print("\n====", item["id"], "PASS" if ok else "FAIL", "====")
        print("TASK:", item["task"][:120])
        print("RAW RESP (first 600):", repr(resp[:600]))
        print("EXTRACTED CODE (first 250):", repr(code[:250]))
        print("SANDBOX ok=", out.ok, "result=", repr(out.result)[:150], "err=", out.error[:150])
