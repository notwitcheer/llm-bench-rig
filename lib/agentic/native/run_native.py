"""Run the native agentic bench for one served model. Assumes llama-server is already up
on :8090 (run_native_bench.sh handles the server + Donald drain). Writes agentic_native.json."""
import json
import sys
from pathlib import Path
from lib.agentic.native.schemas import to_openai_tools
from lib.agentic.native.client import LlamaClient
from lib.agentic.native.agent_loop import run_agent
from lib.agentic.native.tasks import TASKS, check
from lib.agentic.native.metrics import score_run, agentic_score
from lib.agentic.mock_tools import TOOLS

EXEC_PY = {"type": "function", "function": {"name": "execute_python",
           "description": "Run Python; assign `result`. mock tools (web_search, read_file, calc, ...) are in scope.",
           "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}


def run(slug: str):
    tools = to_openai_tools(TOOLS) + [EXEC_PY]
    client = LlamaClient()
    runs, total_tokens, details = [], 0, []
    for t in TASKS:
        res = run_agent(client, t["goal"], tools, max_steps=8)
        ok = check(t, res.final_text)
        runs.append(score_run(ok, res.n_tool_calls, t["opt_calls"], res.bad_calls, res.stalled))
        total_tokens += res.n_tokens
        details.append({"id": t["id"], "axis": t["axis"], "success": ok,
                        "tool_calls": res.n_tool_calls, "bad_calls": res.bad_calls,
                        "tokens": res.n_tokens, "stalled": res.stalled})
        print(f"[native] {t['id']:22} {'OK' if ok else 'XX'} calls={res.n_tool_calls} tok={res.n_tokens}")
    summ = agentic_score(runs, tokens_per_task=total_tokens / max(1, len(TASKS)))
    out = {"model": slug, **summ, "details": details}
    p = Path(f"results/{slug}")
    p.mkdir(parents=True, exist_ok=True)
    (p / "agentic_native.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(summ))


if __name__ == "__main__":
    run(sys.argv[1])
