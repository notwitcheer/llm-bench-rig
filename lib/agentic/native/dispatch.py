"""Execute a single tool call against the deterministic mock tools / sandbox.
Returns {"ok": bool, "result": <json-able>, "error": str}. `execute_python` runs
the model's code in the existing sandbox; everything else calls a mock tool directly."""
from lib.agentic.mock_tools import build_namespace
from lib.agentic.sandbox import run_code_action
from lib.agentic.native.tools_ext import EXT_NS

_NS = build_namespace()


def dispatch_tool(name: str, args: dict) -> dict:
    if name == "execute_python":
        out = run_code_action(args.get("code", ""))
        return {"ok": out.ok, "result": out.result, "error": out.error}
    fn = _NS.get(name) or EXT_NS.get(name)
    if fn is None:
        return {"ok": False, "result": None, "error": f"unknown tool '{name}'"}
    try:
        return {"ok": True, "result": fn(**args), "error": ""}
    except Exception as e:
        return {"ok": False, "result": None, "error": f"{type(e).__name__}: {e}"}
