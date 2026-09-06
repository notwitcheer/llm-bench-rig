"""Execute a model-written code action with mock tools injected, in an isolated
subprocess. The code is expected to assign a variable named `result`."""
import json, os, subprocess, sys, tempfile
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ActionOutput:
    ok: bool
    result: object = None
    error: str = ""

# mock_tools is loaded by file path on purpose: `from lib.agentic.mock_tools import`
# would run lib/agentic/__init__.py, which imports the evaluators and httpx, and
# that import chain alone can eat most of the action timeout on a loaded box.
_HARNESS = '''
import importlib.util, json, sys
_spec = importlib.util.spec_from_file_location("mock_tools", sys.argv[2])
_mt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mt)
_ns = _mt.build_namespace()
globals().update(_ns)
try:
    exec(compile(open(sys.argv[1]).read(), "<action>", "exec"), globals())
    if "result" not in globals():
        print(json.dumps({"ok": False, "error": "no `result` assigned"}))
    else:
        print(json.dumps({"ok": True, "result": globals()["result"]}, default=str))
except Exception as e:
    print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))
'''

def run_code_action(code: str, timeout: int = 10) -> ActionOutput:
    rig_root = Path(__file__).parent.parent.parent
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as cf:
        cf.write(code); cf.flush(); code_path = cf.name
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as hf:
        hf.write(_HARNESS); hf.flush(); harness_path = hf.name
    try:
        proc = subprocess.run(
            # the interpreter running the harness, so a venv install sees its own deps
            [sys.executable, harness_path, code_path,
             str(rig_root / "lib" / "agentic" / "mock_tools.py")],
            capture_output=True, text=True, timeout=timeout, cwd=str(rig_root),
            env={**os.environ, "PYTHONPATH": str(rig_root)},
        )
        line = (proc.stdout or "").strip().splitlines()[-1] if proc.stdout.strip() else ""
        if not line:
            return ActionOutput(False, error=(proc.stderr[-300:] or "no output"))
        data = json.loads(line)
        return ActionOutput(data["ok"], data.get("result"), data.get("error", ""))
    except subprocess.TimeoutExpired:
        return ActionOutput(False, error="timeout")
    except json.JSONDecodeError:
        return ActionOutput(False, error="bad harness output")
    finally:
        Path(code_path).unlink(missing_ok=True)
        Path(harness_path).unlink(missing_ok=True)
