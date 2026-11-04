
import sys, os, json, sqlite3, time
sys.path.insert(0, "/home/witcheer/.hermes/hermes-agent")
import hermes_bootstrap
from agent.context_compressor import ContextCompressor
from agent.model_metadata import estimate_messages_tokens_rough
db = sys.argv[1]; T = int(sys.argv[2]); pfn = int(sys.argv[3]); nmax = int(sys.argv[4]) if len(sys.argv) > 4 else 999
c = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
rows = c.execute("select role,content,tool_calls,tool_call_id,tool_name,reasoning_content from messages where compacted=1 or active=0 order by id").fetchall()
rows = [r for r in rows][:nmax]
msgs = []
for role, content, tc, tcid, tname, rc in rows:
    m = {"role": role, "content": content or ""}
    if role == "assistant" and tc:
        m["tool_calls"] = json.loads(tc)
    if role == "assistant" and rc:
        m["reasoning_content"] = rc
    if role == "tool":
        m["tool_call_id"] = tcid
    msgs.append(m)
print("reasoning chars", sum(len(m.get("reasoning_content") or "") for m in msgs)); print("n_messages", len(msgs), "rough_tokens", estimate_messages_tokens_rough(msgs), "chars", sum(len(m.get("content") or "") + len(json.dumps(m.get("tool_calls") or "")) for m in msgs))
cc = ContextCompressor(model="Qwen3.8-27B-Q6_K.gguf", threshold_percent=0.5, protect_first_n=pfn, protect_last_n=20, quiet_mode=True,
                       config_context_length=65536, provider="custom", base_url="http://127.0.0.1:1/v1", api_key="x", threshold_tokens_cap=T or None)
cc.update_model("Qwen3.8-27B-Q6_K.gguf", 65536, base_url="http://127.0.0.1:1/v1", provider="custom")
print("trigger threshold_tokens", cc.threshold_tokens, "tail_budget", cc.tail_token_budget, "tail_mode", cc.tail_mode)
SUM = "## Goal\nbuild the lighthouse page\n## Progress\n" + ("- wrote and patched index.html; checks pass\n" * 60)
def fake_summary(*a, **k):
    return SUM
for name in ("_generate_summary", "_call_summary_llm", "generate_summary"):
    if hasattr(cc, name):
        setattr(cc, name, fake_summary); print("patched", name)
before = estimate_messages_tokens_rough(msgs)
try:
    out = cc.compress(msgs, current_tokens=before)
except Exception as e:
    print("compress raised", type(e).__name__, str(e)[:200]); raise
after = estimate_messages_tokens_rough(out)
print("RESULT n_after", len(out), "rough_tokens_after", after, "roles", [m["role"] for m in out][:30])
print("telemetry no-op?", getattr(cc, "_structural_no_op_backoff_until", None), "made_progress", getattr(cc, "_last_compression_made_progress", None))
