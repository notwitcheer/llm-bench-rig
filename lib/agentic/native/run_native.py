"""Run the native agentic bench for one served model.
Modes: short (5 short-context axes -> agentic_native.json) | long32k | long128k
(long-context tier -> agentic_longctx_<tier>.json). llama-server must be up on :8090."""
import json, sys
from pathlib import Path
from lib.agentic.native.schemas import to_openai_tools
from lib.agentic.native.client import LlamaClient
from lib.agentic.native.agent_loop import run_agent
from lib.agentic.native.tasks import TASKS, check
from lib.agentic.native.metrics import score_run, agentic_score, longctx_summary
from lib.agentic.mock_tools import TOOLS
from lib.agentic.native.tools_ext import EXT_SHORT_TOOLS, EXT_LONG_TOOLS

EXEC_PY = {"type": "function", "function": {"name": "execute_python",
           "description": "Run Python; assign `result`. mock tools (web_search, read_file, calc, ...) are in scope.",
           "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}

SHORT_AXES = {"chain", "multistep", "coding", "error_recovery", "distractor"}


def _short_tools():
    return to_openai_tools(TOOLS) + [EXEC_PY] + to_openai_tools(EXT_SHORT_TOOLS)


def _long_tools():
    return to_openai_tools(TOOLS) + [EXEC_PY] + to_openai_tools(EXT_LONG_TOOLS)


def run(slug: str, mode: str = "short"):
    if mode == "short":
        tasks = [t for t in TASKS if t["axis"] in SHORT_AXES]
        tools, client = _short_tools(), LlamaClient()
    else:
        tier = "32k" if mode == "long32k" else "128k"
        tasks = [t for t in TASKS if t["axis"] == "long_context" and t["ctx_tier"] == tier]
        tools, client = _long_tools(), LlamaClient(timeout=600)
    runs, total_tokens, details = [], 0, []
    for t in tasks:
        try:
            res = run_agent(client, t["goal"], tools, max_steps=8)
            ok = check(t, res.final_text)
            runs.append(score_run(ok, res.n_tool_calls, t["opt_calls"], res.bad_calls, res.stalled))
            total_tokens += res.n_tokens
            details.append({"id": t["id"], "axis": t["axis"], "success": ok,
                            "tool_calls": res.n_tool_calls, "bad_calls": res.bad_calls,
                            "tokens": res.n_tokens, "stalled": res.stalled})
            print(f"[{mode}] {t['id']:24} {'OK' if ok else 'XX'} calls={res.n_tool_calls} tok={res.n_tokens}")
        except Exception as e:
            # one task erroring (e.g. a server-side context overflow) must not void the run
            runs.append(score_run(False, 0, t["opt_calls"], 0, True))
            details.append({"id": t["id"], "axis": t["axis"], "success": False, "tool_calls": 0,
                            "bad_calls": 0, "tokens": 0, "stalled": True, "error": f"{type(e).__name__}: {e}"})
            print(f"[{mode}] {t['id']:24} ERR {type(e).__name__}: {e}")
    p = Path(f"results/{slug}"); p.mkdir(parents=True, exist_ok=True)
    if mode == "short":
        summ = agentic_score(runs, tokens_per_task=total_tokens / max(1, len(tasks)))
        out = {"model": slug, **summ, "details": details}
        (p / "agentic_native.json").write_text(json.dumps(out, indent=2))
    else:
        tier = "32k" if mode == "long32k" else "128k"
        out = {"model": slug, "tier": tier, **longctx_summary(details), "details": details}
        (p / f"agentic_longctx_{tier}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "details"}))


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "short")
