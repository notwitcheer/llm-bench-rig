"""Execute a model-written code action with mock tools injected, in an isolated
subprocess. The code is expected to assign a variable named `result`."""
import json, subprocess, tempfile
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ActionOutput:
    ok: bool
    result: object = None
    error: str = ""

_HARNESS = '''
import json, sys
from lib.agentic.mock_tools import build_namespace
_ns = build_namespace()
globals().update(_ns)
try:
    exec(compile(open(sys.argv[1]).read(), "<action>", "exec"), globals())
    if "result" not in globals():
        print(json.dumps({"ok": False, "error": "no `result` assigned"}))
    else:
        print(json.dumps({"ok": True, "result": result}, default=str))
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
            ["python3", harness_path, code_path],
            capture_output=True, text=True, timeout=timeout, cwd=str(rig_root),
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
